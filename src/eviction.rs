//! Context window & eviction, "the RAM" (ported from `aios/eviction.py`).
//!
//! Fixed-size. When it fills, items are **demoted**, not deleted (spec §1.1):
//! session messages get archived, details/summaries are simply unloaded (they
//! already live in the store). Eviction picks the highest-scoring slot, where a
//! higher score means "more evictable".

use crate::store::{MemoryStore, Timestamp};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotType {
    Identity,      // Level 0, never evicted
    BranchSummary, // Level 1, evicted on topic change
    Detail,        // Level 2, evicted under pressure
    Message,       // session messages, FIFO with scoring
    System,        // system instructions, never evicted
}

impl SlotType {
    pub fn label(&self) -> &'static str {
        match self {
            SlotType::Identity => "identity",
            SlotType::BranchSummary => "summary",
            SlotType::Detail => "detail",
            SlotType::Message => "message",
            SlotType::System => "system",
        }
    }
    fn type_bias(&self) -> f64 {
        match self {
            SlotType::Message => 10.0,      // messages evict easiest
            SlotType::Detail => 5.0,        // details next
            SlotType::BranchSummary => 0.0, // summaries evict last
            _ => 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ContextSlot {
    pub slot_type: SlotType,
    pub content: String,
    pub branch_name: String,
    pub pinned: bool,
    pub loaded_at: Timestamp,
    pub last_accessed: Timestamp,
    pub access_count: u32,
    pub relevance_score: f64,
    pub source_index: i64,
}

impl ContextSlot {
    pub fn new(slot_type: SlotType, content: String, branch_name: String, now: Timestamp) -> Self {
        ContextSlot {
            slot_type,
            content,
            branch_name,
            pinned: false,
            loaded_at: now,
            last_accessed: now,
            access_count: 0,
            relevance_score: 0.0,
            source_index: -1,
        }
    }

    pub fn token_estimate(&self) -> usize {
        self.content.len() / 4
    }

    pub fn touch(&mut self, now: Timestamp, relevance: f64) {
        self.last_accessed = now;
        self.access_count += 1;
        self.relevance_score = relevance;
    }

    /// Higher score = more likely to evict. `-inf` for pinned / identity / system.
    pub fn eviction_score(&self, current_time: Timestamp, current_topic: &str) -> f64 {
        if self.pinned || matches!(self.slot_type, SlotType::Identity | SlotType::System) {
            return f64::NEG_INFINITY;
        }
        let mut score = 0.0;

        // Recency: +1 per minute of staleness.
        let staleness = current_time - self.last_accessed;
        score += staleness / 60.0;

        // Frequency: more accesses = harder to evict.
        score -= self.access_count as f64 * 5.0;

        // Topic relevance.
        if !current_topic.is_empty() && !self.branch_name.is_empty() {
            if !self.branch_name.to_lowercase().contains(&current_topic.to_lowercase()) {
                score += 20.0; // off-topic: big penalty
            } else {
                score -= 10.0; // on-topic: bonus
            }
        }

        // Type priority.
        score += self.slot_type.type_bias();
        score
    }
}

#[derive(Debug, Clone)]
pub struct EvictionRecord {
    pub slot_type: String,
    pub branch: String,
    pub content_preview: String,
    pub access_count: u32,
}

/// A demoted item the owner must apply to its store: (branch, role, content).
/// Emitted when the window has no direct store reference, message-passing
/// instead of a held `&mut MemoryStore`, so long-lived sessions can own both.
pub type Demotion = (String, String, String);

pub struct ContextWindow<'a> {
    pub budget_tokens: usize,
    store: Option<&'a mut MemoryStore>,
    pub slots: Vec<ContextSlot>,
    pub eviction_log: Vec<EvictionRecord>,
    pub pending_demotions: Vec<Demotion>,
    pub evicted_summary: String,
    pub current_topic: String,
    pub total_evictions: usize,
    pub total_loads: usize,
    warn: f64,
    evict: f64,
    flush: f64,
    clock: Timestamp,
}

impl<'a> ContextWindow<'a> {
    pub fn new(budget_tokens: usize, store: Option<&'a mut MemoryStore>) -> Self {
        ContextWindow {
            budget_tokens,
            store,
            slots: Vec::new(),
            eviction_log: Vec::new(),
            pending_demotions: Vec::new(),
            evicted_summary: String::new(),
            current_topic: String::new(),
            total_evictions: 0,
            total_loads: 0,
            warn: 0.70,
            evict: 0.85,
            flush: 0.95,
            clock: 0.0,
        }
    }

    fn tick(&mut self) -> Timestamp {
        self.clock += 1.0;
        self.clock
    }

    // --- Capacity ---

    pub fn used_tokens(&self) -> usize {
        self.slots.iter().map(|s| s.token_estimate()).sum::<usize>() + self.evicted_summary.len() / 4
    }

    pub fn fill_ratio(&self) -> f64 {
        self.used_tokens() as f64 / self.budget_tokens.max(1) as f64
    }

    pub fn pressure_level(&self) -> &'static str {
        let r = self.fill_ratio();
        if r >= self.flush {
            "CRITICAL"
        } else if r >= self.evict {
            "HIGH"
        } else if r >= self.warn {
            "WARNING"
        } else {
            "OK"
        }
    }

    pub fn available_tokens(&self) -> usize {
        self.budget_tokens.saturating_sub(self.used_tokens())
    }

    // --- Load ---

    pub fn load_identity(&mut self, content: &str) {
        self.slots.retain(|s| s.slot_type != SlotType::Identity);
        let now = self.tick();
        let mut slot = ContextSlot::new(SlotType::Identity, content.to_string(), String::new(), now);
        slot.pinned = true;
        self.slots.insert(0, slot);
        self.total_loads += 1;
    }

    pub fn load_branch(&mut self, branch_name: &str, summary: &str) -> bool {
        let now = self.tick();
        for s in &mut self.slots {
            if s.slot_type == SlotType::BranchSummary && s.branch_name == branch_name {
                s.touch(now, 1.0);
                return true;
            }
        }
        let slot = ContextSlot::new(SlotType::BranchSummary, summary.to_string(), branch_name.to_string(), now);
        if !self.make_room(slot.token_estimate()) {
            return false;
        }
        self.slots.push(slot);
        self.total_loads += 1;
        true
    }

    pub fn load_detail(&mut self, branch_name: &str, detail: &str, source_index: i64) -> bool {
        let now = self.tick();
        let mut slot = ContextSlot::new(SlotType::Detail, detail.to_string(), branch_name.to_string(), now);
        slot.source_index = source_index;
        if !self.make_room(slot.token_estimate()) {
            return false;
        }
        self.slots.push(slot);
        self.total_loads += 1;
        true
    }

    pub fn load_message(&mut self, role: &str, content: &str, pinned: bool) -> bool {
        let now = self.tick();
        let mut slot = ContextSlot::new(SlotType::Message, format!("{role}: {content}"), String::new(), now);
        slot.pinned = pinned;
        if !self.make_room(slot.token_estimate()) {
            return false;
        }
        self.slots.push(slot);
        self.total_loads += 1;
        true
    }

    // --- Eviction ---

    fn make_room(&mut self, needed: usize) -> bool {
        let mut attempts = 0;
        while self.used_tokens() + needed > self.budget_tokens {
            if !self.evict_one() {
                return false;
            }
            attempts += 1;
            if attempts > 50 {
                return false;
            }
        }
        true
    }

    fn evict_one(&mut self) -> bool {
        let now = self.clock;
        let topic = self.current_topic.clone();
        let mut best: Option<(usize, f64)> = None;
        for (i, slot) in self.slots.iter().enumerate() {
            let score = slot.eviction_score(now, &topic);
            if score == f64::NEG_INFINITY {
                continue;
            }
            match best {
                Some((_, bs)) if score <= bs => {}
                _ => best = Some((i, score)),
            }
        }
        let Some((idx, _)) = best else { return false };
        let slot = self.slots.remove(idx);
        self.demote(&slot);
        self.total_evictions += 1;
        true
    }

    pub fn evict_branch(&mut self, branch_name: &str) {
        let mut i = 0;
        while i < self.slots.len() {
            if self.slots[i].branch_name == branch_name && !self.slots[i].pinned {
                let slot = self.slots.remove(i);
                self.demote(&slot);
                self.total_evictions += 1;
            } else {
                i += 1;
            }
        }
    }

    /// Evict old messages keeping the last N, folding them into a recursive summary.
    pub fn evict_messages(&mut self, keep_last: usize) {
        let msg_positions: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.slot_type == SlotType::Message && !s.pinned)
            .map(|(i, _)| i)
            .collect();
        if msg_positions.len() <= keep_last {
            return;
        }
        let to_evict = &msg_positions[..msg_positions.len() - keep_last];
        let mut evicted_content = Vec::new();
        for &idx in to_evict.iter().rev() {
            let slot = self.slots.remove(idx);
            evicted_content.push(slot.content.clone());
            self.demote(&slot);
            self.total_evictions += 1;
        }
        if !evicted_content.is_empty() {
            evicted_content.reverse();
            let new_summary = compress_messages(&evicted_content);
            if self.evicted_summary.is_empty() {
                self.evicted_summary = new_summary;
            } else {
                self.evicted_summary = format!("{} {}", self.evicted_summary, new_summary);
            }
        }
    }

    pub fn flush(&mut self, keep_pinned: bool) {
        let mut kept = Vec::new();
        for slot in std::mem::take(&mut self.slots) {
            if keep_pinned && slot.pinned {
                kept.push(slot);
            } else {
                self.demote(&slot);
                self.total_evictions += 1;
            }
        }
        self.slots = kept;
    }

    fn demote(&mut self, slot: &ContextSlot) {
        self.eviction_log.push(EvictionRecord {
            slot_type: slot.slot_type.label().to_string(),
            branch: slot.branch_name.clone(),
            content_preview: slot.content.chars().take(80).collect(),
            access_count: slot.access_count,
        });
        // Only raw session messages need demotion to the archive; details and
        // summaries were loaded FROM the store, so unloading is enough.
        if slot.slot_type == SlotType::Message && !slot.branch_name.is_empty() {
            let (role, content) = match slot.content.split_once(": ") {
                Some((r, c)) => (r, c),
                None => ("unknown", slot.content.as_str()),
            };
            match self.store.as_deref_mut() {
                Some(store) => store.add_archive(&slot.branch_name, role, content, self.clock),
                None => self.pending_demotions.push((
                    slot.branch_name.clone(),
                    role.to_string(),
                    content.to_string(),
                )),
            }
        }
    }

    /// Drain demotions accumulated while the window had no store reference.
    /// The caller applies them: `for (b, r, c) in cw.drain_demotions() { store.add_archive(&b, &r, &c, now) }`.
    pub fn drain_demotions(&mut self) -> Vec<Demotion> {
        std::mem::take(&mut self.pending_demotions)
    }

    // --- Topic ---

    pub fn set_topic(&mut self, topic: &str) {
        self.current_topic = topic.to_string();
        let now = self.clock;
        let tl = topic.to_lowercase();
        for slot in &mut self.slots {
            if !slot.branch_name.is_empty() && slot.branch_name.to_lowercase().contains(&tl) {
                slot.touch(now, 1.0);
            }
        }
    }

    // --- Render ---

    pub fn build(&self) -> String {
        let mut sections = Vec::new();
        if !self.evicted_summary.is_empty() {
            sections.push(format!("[PREVIOUS CONTEXT]\n{}", self.evicted_summary));
        }
        for st in [SlotType::Identity, SlotType::BranchSummary, SlotType::Detail, SlotType::Message] {
            for slot in self.slots.iter().filter(|s| s.slot_type == st) {
                match st {
                    SlotType::Identity => sections.push(format!("[IDENTITY]\n{}", slot.content)),
                    SlotType::BranchSummary => {
                        sections.push(format!("[BRANCH: {}]\n{}", slot.branch_name, slot.content))
                    }
                    SlotType::Detail => sections.push(format!("[DETAIL: {}]\n{}", slot.branch_name, slot.content)),
                    SlotType::Message => sections.push(slot.content.clone()),
                    _ => {}
                }
            }
        }
        sections.join("\n\n")
    }

    pub fn loaded_branches(&self) -> Vec<String> {
        let mut set: Vec<String> = self
            .slots
            .iter()
            .filter(|s| s.slot_type == SlotType::BranchSummary && !s.branch_name.is_empty())
            .map(|s| s.branch_name.clone())
            .collect();
        set.sort();
        set.dedup();
        set
    }
}

fn compress_messages(messages: &[String]) -> String {
    if messages.is_empty() {
        return String::new();
    }
    let user_msgs: Vec<&String> = messages.iter().filter(|m| m.starts_with("user:")).collect();
    if !user_msgs.is_empty() {
        let start = user_msgs.len().saturating_sub(5);
        let compressed: Vec<String> = user_msgs[start..]
            .iter()
            .filter_map(|m| m.split_once(": ").map(|(_, c)| c.chars().take(60).collect::<String>()))
            .collect();
        format!("[Earlier: {}]", compressed.join("; "))
    } else {
        format!("[Earlier: {} messages exchanged]", messages.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_and_identity_never_evict() {
        let mut cw = ContextWindow::new(1000, None);
        cw.load_identity("I am the user");
        let id_slot = &cw.slots[0];
        assert_eq!(id_slot.eviction_score(100.0, ""), f64::NEG_INFINITY);
    }

    #[test]
    fn off_topic_evicts_before_on_topic() {
        let now = 100.0;
        let mut on = ContextSlot::new(SlotType::Detail, "x".into(), "adoption".into(), 0.0);
        let mut off = ContextSlot::new(SlotType::Detail, "y".into(), "painting".into(), 0.0);
        on.last_accessed = now;
        off.last_accessed = now;
        let s_on = on.eviction_score(now, "adoption");
        let s_off = off.eviction_score(now, "adoption");
        assert!(s_off > s_on, "off-topic ({s_off}) should outscore on-topic ({s_on})");
    }

    #[test]
    fn budget_enforced_via_eviction() {
        let mut cw = ContextWindow::new(50, None); // ~200 chars
        for i in 0..40 {
            cw.load_message("user", &format!("message number {i} with some padding text"), false);
        }
        assert!(cw.used_tokens() <= cw.budget_tokens, "used {} > budget {}", cw.used_tokens(), cw.budget_tokens);
        assert!(cw.total_evictions > 0);
    }

    #[test]
    fn eviction_is_demotion_message_archived() {
        let mut store = MemoryStore::new();
        store.create_branch("adoption", "", "user", 0.0);
        {
            let mut cw = ContextWindow::new(20, Some(&mut store));
            let mut slot = ContextSlot::new(SlotType::Message, "user: I researched agencies".into(), "adoption".into(), 0.0);
            slot.last_accessed = 0.0;
            cw.slots.push(slot);
            cw.flush(false); // evict everything -> should archive
        }
        assert_eq!(store.get_branch("adoption").unwrap().archive.len(), 1);
    }
}
