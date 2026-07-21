//! Continuum Memory Store, the "disk" in the OS analogy.
//!
//! Four-level hierarchy with temporal versioning, ported from `continuum/store.py`:
//!   Level 0: Identity , who the user is, always loaded
//!   Level 1: Branches , topic summaries, the routing layer
//!   Level 2: Details  , specific facts per branch
//!   Level 3: Archive  , raw conversation pairs
//!
//! Every value is versioned (copy-on-write). Nothing is ever destroyed.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Monotonic-ish logical clock. The Python port used wall-clock `time.time()`;
/// for a deterministic kernel we use a simple counter so version ordering and
/// eviction staleness are reproducible in tests.
pub type Timestamp = f64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Version {
    pub value: String,
    pub timestamp: Timestamp,
    #[serde(default)]
    pub source: String,
}

/// A value that remembers all previous states (one cell in the 3D memory tree:
/// spatial = level + branch, temporal = version history).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VersionedValue {
    pub history: Vec<Version>,
}

impl VersionedValue {
    pub fn new(initial: &str, source: &str, now: Timestamp) -> Self {
        let mut vv = VersionedValue::default();
        if !initial.is_empty() {
            vv.history.push(Version { value: initial.to_string(), timestamp: now, source: source.to_string() });
        }
        vv
    }

    pub fn current(&self) -> &str {
        self.history.last().map(|v| v.value.as_str()).unwrap_or("")
    }

    pub fn last_updated(&self) -> Timestamp {
        self.history.last().map(|v| v.timestamp).unwrap_or(0.0)
    }

    pub fn version_count(&self) -> usize {
        self.history.len()
    }

    /// Copy-on-write: old value preserved, new value appended.
    pub fn update(&mut self, new_value: &str, source: &str, now: Timestamp) {
        self.history.push(Version { value: new_value.to_string(), timestamp: now, source: source.to_string() });
    }

    /// What was this value at time `t`?
    pub fn at_time(&self, t: Timestamp) -> Option<&str> {
        let mut result = None;
        for v in &self.history {
            if v.timestamp <= t {
                result = Some(v.value.as_str());
            } else {
                break;
            }
        }
        result
    }

    /// (previous, current) if there has been at least one update.
    pub fn diff(&self) -> Option<(&str, &str)> {
        let n = self.history.len();
        if n < 2 {
            return None;
        }
        Some((self.history[n - 2].value.as_str(), self.history[n - 1].value.as_str()))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub role: String,
    pub content: String,
    pub timestamp: Timestamp,
    #[serde(default)]
    pub session_id: String,
}

/// One topic branch: a Level-1 summary, Level-2 details, Level-3 archive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Branch {
    pub name: String,
    pub summary: VersionedValue,
    pub details: Vec<VersionedValue>,
    pub archive: Vec<ArchiveEntry>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub created_at: Timestamp,
}

impl Branch {
    pub fn new(name: &str, summary: &str, source: &str, now: Timestamp) -> Self {
        Branch {
            name: name.to_string(),
            summary: VersionedValue::new(summary, source, now),
            details: Vec::new(),
            archive: Vec::new(),
            tags: Vec::new(),
            created_at: now,
        }
    }

    pub fn add_detail(&mut self, text: &str, source: &str, now: Timestamp) {
        self.details.push(VersionedValue::new(text, source, now));
    }

    pub fn update_summary(&mut self, new_summary: &str, source: &str, now: Timestamp) {
        self.summary.update(new_summary, source, now);
    }

    pub fn add_archive(&mut self, role: &str, content: &str, session_id: &str, now: Timestamp) {
        self.archive.push(ArchiveEntry {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: now,
            session_id: session_id.to_string(),
        });
    }

    pub fn add_tag(&mut self, tag: &str) {
        let t = tag.to_lowercase();
        if !self.tags.contains(&t) {
            self.tags.push(t);
        }
    }

    /// All searchable text in this branch (for the TF-IDF matcher).
    pub fn all_text(&self) -> String {
        let mut parts = vec![self.name.clone(), self.summary.current().to_string()];
        parts.extend(self.details.iter().map(|d| d.current().to_string()));
        parts.extend(self.tags.iter().cloned());
        parts.join(" ")
    }

    /// Rough token count (~4 chars per token) for a given hierarchy level.
    pub fn token_estimate(&self, level: u8) -> usize {
        let mut chars = self.summary.current().len();
        if level >= 2 {
            chars += self.details.iter().map(|d| d.current().len()).sum::<usize>();
        }
        if level >= 3 {
            chars += self.archive.iter().map(|a| a.content.len()).sum::<usize>();
        }
        chars / 4
    }

    pub fn total_versions(&self) -> usize {
        self.summary.version_count() + self.details.iter().map(|d| d.version_count()).sum::<usize>()
    }
}

/// The complete four-level memory hierarchy.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MemoryStore {
    pub identity: VersionedValue,
    /// key (`to_key(name)`) -> branch. BTreeMap keeps a stable iteration order.
    pub branches: BTreeMap<String, Branch>,
}

impl MemoryStore {
    pub fn new() -> Self {
        MemoryStore::default()
    }

    // --- Level 0: Identity ---

    pub fn set_identity(&mut self, text: &str, source: &str, now: Timestamp) {
        if self.identity.current().is_empty() {
            self.identity = VersionedValue::new(text, source, now);
        } else {
            self.identity.update(text, source, now);
        }
    }

    pub fn get_identity(&self) -> &str {
        self.identity.current()
    }

    // --- Level 1: Branches ---

    pub fn create_branch(&mut self, name: &str, summary: &str, source: &str, now: Timestamp) {
        let key = to_key(name);
        match self.branches.get_mut(&key) {
            Some(b) => {
                if !summary.is_empty() {
                    b.update_summary(summary, source, now);
                }
            }
            None => {
                self.branches.insert(key, Branch::new(name, summary, source, now));
            }
        }
    }

    pub fn get_branch(&self, name: &str) -> Option<&Branch> {
        self.branches.get(&to_key(name))
    }

    pub fn get_branch_mut(&mut self, name: &str) -> Option<&mut Branch> {
        self.branches.get_mut(&to_key(name))
    }

    /// Fuzzy find: checks key, then substring of name, then tags.
    pub fn find_branch(&self, query: &str) -> Option<&Branch> {
        let key = to_key(query);
        if let Some(b) = self.branches.get(&key) {
            return Some(b);
        }
        let ql = query.to_lowercase();
        for b in self.branches.values() {
            if b.name.to_lowercase().contains(&ql) || b.tags.iter().any(|t| t == &ql) {
                return Some(b);
            }
        }
        None
    }

    pub fn list_branches(&self) -> Vec<String> {
        self.branches.values().map(|b| b.name.clone()).collect()
    }

    pub fn all_branches(&self) -> impl Iterator<Item = &Branch> {
        self.branches.values()
    }

    // --- Level 2: Details ---

    pub fn add_detail(&mut self, branch_name: &str, text: &str, source: &str, now: Timestamp) {
        if self.get_branch(branch_name).is_none() {
            self.create_branch(branch_name, "", source, now);
        }
        if let Some(b) = self.get_branch_mut(branch_name) {
            b.add_detail(text, source, now);
        }
    }

    // --- Level 3: Archive ---

    pub fn add_archive(&mut self, branch_name: &str, role: &str, content: &str, now: Timestamp) {
        if let Some(b) = self.get_branch_mut(branch_name) {
            b.add_archive(role, content, "", now);
        }
    }

    // --- Stats & persistence ---

    pub fn stats(&self) -> StoreStats {
        let details: usize = self.branches.values().map(|b| b.details.len()).sum();
        let archive: usize = self.branches.values().map(|b| b.archive.len()).sum();
        let versions: usize = self.identity.version_count()
            + self.branches.values().map(|b| b.total_versions()).sum::<usize>();
        StoreStats {
            identity_chars: self.identity.current().len(),
            branches: self.branches.len(),
            details,
            archive_entries: archive,
            total_versions: versions,
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let store = serde_json::from_str(&data)?;
        Ok(store)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StoreStats {
    pub identity_chars: usize,
    pub branches: usize,
    pub details: usize,
    pub archive_entries: usize,
    pub total_versions: usize,
}

/// `"Adoption Journey"` -> `"adoption_journey"`.
pub fn to_key(name: &str) -> String {
    name.to_lowercase().trim().replace(' ', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versioned_value_cow_and_time_travel() {
        let mut vv = VersionedValue::new("v1", "init", 1.0);
        vv.update("v2", "update", 2.0);
        vv.update("v3", "update", 3.0);
        assert_eq!(vv.current(), "v3");
        assert_eq!(vv.version_count(), 3);
        assert_eq!(vv.at_time(1.5), Some("v1"));
        assert_eq!(vv.at_time(2.0), Some("v2"));
        assert_eq!(vv.at_time(0.5), None);
        assert_eq!(vv.diff(), Some(("v2", "v3")));
    }

    #[test]
    fn store_hierarchy_and_keys() {
        let mut s = MemoryStore::new();
        s.set_identity("Abhirama, builds operating systems", "user", 1.0);
        s.create_branch("Adoption Journey", "researching agencies", "user", 2.0);
        s.add_detail("Adoption Journey", "contacted 3 agencies", "user", 3.0);
        assert_eq!(s.get_identity(), "Abhirama, builds operating systems");
        assert!(s.get_branch("adoption journey").is_some());
        assert_eq!(s.get_branch("Adoption Journey").unwrap().details.len(), 1);
        assert_eq!(to_key("Adoption Journey"), "adoption_journey");
    }

    #[test]
    fn store_roundtrip_serialization() {
        let mut s = MemoryStore::new();
        s.set_identity("id", "user", 1.0);
        s.create_branch("t1", "sum", "user", 2.0);
        let json = serde_json::to_string(&s).unwrap();
        let back: MemoryStore = serde_json::from_str(&json).unwrap();
        assert_eq!(back.get_identity(), "id");
        assert_eq!(back.list_branches(), vec!["t1".to_string()]);
    }
}
