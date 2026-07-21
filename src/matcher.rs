//! TF-IDF branch matcher, "the TLB" (ported from `continuum/matcher.py`).
//!
//! No LLM, no embeddings. Each branch builds a term-frequency profile from its
//! name (3x weight), tags (2x), summary, and details. Queries match via cosine
//! similarity over TF-IDF vectors, with an exact name-overlap boost. Used as the
//! kernel's keyword routing / fallback signal.

use std::collections::{HashMap, HashSet};

use crate::store::MemoryStore;

const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
    "with", "this", "that", "from", "by", "be", "was", "were", "been", "are", "am", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "can", "may", "might", "shall",
    "i", "you", "he", "she", "we", "they", "my", "your", "his", "her", "our", "their", "me", "him",
    "us", "them", "its", "what", "which", "who", "whom", "how", "when", "where", "why", "if", "then",
    "so", "no", "yes", "about", "up", "out", "just", "also", "very", "some", "any", "all", "each",
    "into", "over", "after", "before", "between", "through", "more", "than", "too", "here", "there",
    "now", "being", "going",
];

pub fn tokenize(text: &str) -> Vec<String> {
    let stops: HashSet<&str> = STOP_WORDS.iter().copied().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            if cur.len() > 1 && !stops.contains(cur.as_str()) {
                out.push(cur.clone());
            }
            cur.clear();
        }
    }
    if cur.len() > 1 && !stops.contains(cur.as_str()) {
        out.push(cur);
    }
    out
}

struct BranchProfile {
    branch_name: String,
    terms: HashMap<String, f64>,
    total: f64,
}

impl BranchProfile {
    fn tf(&self, term: &str) -> f64 {
        if self.total == 0.0 {
            0.0
        } else {
            self.terms.get(term).copied().unwrap_or(0.0) / self.total
        }
    }
}

pub struct Matcher {
    profiles: Vec<BranchProfile>,
    idf: HashMap<String, f64>,
}

impl Matcher {
    pub fn build(store: &MemoryStore) -> Self {
        let mut profiles = Vec::new();
        let mut doc_freq: HashMap<String, f64> = HashMap::new();

        for branch in store.all_branches() {
            let mut terms: HashMap<String, f64> = HashMap::new();
            let bump = |t: &str, w: f64, terms: &mut HashMap<String, f64>| {
                *terms.entry(t.to_string()).or_insert(0.0) += w;
            };
            for t in tokenize(&branch.name) {
                bump(&t, 3.0, &mut terms);
            }
            for tag in &branch.tags {
                for t in tokenize(tag) {
                    bump(&t, 2.0, &mut terms);
                }
            }
            for t in tokenize(branch.summary.current()) {
                bump(&t, 1.0, &mut terms);
            }
            for d in &branch.details {
                for t in tokenize(d.current()) {
                    bump(&t, 1.0, &mut terms);
                }
            }
            for term in terms.keys() {
                *doc_freq.entry(term.clone()).or_insert(0.0) += 1.0;
            }
            let total: f64 = terms.values().sum();
            profiles.push(BranchProfile { branch_name: branch.name.clone(), terms, total });
        }

        let n = profiles.len().max(1) as f64;
        let idf = doc_freq
            .into_iter()
            .map(|(term, df)| (term, (n / df).ln() + 1.0))
            .collect();

        Matcher { profiles, idf }
    }

    pub fn match_query(&self, query: &str, top_k: usize, threshold: f64) -> Vec<(String, f64)> {
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }
        let mut query_tf: HashMap<String, f64> = HashMap::new();
        for t in &query_terms {
            *query_tf.entry(t.clone()).or_insert(0.0) += 1.0;
        }
        let query_total = query_terms.len() as f64;
        let query_lower = query.to_lowercase();

        let mut scores: Vec<(String, f64)> = Vec::new();
        for profile in &self.profiles {
            let mut score = self.cosine_tfidf(&query_tf, query_total, profile);

            // Name-overlap boost (strong signal when a branch name is in the query).
            let name_tokens = tokenize(&profile.branch_name);
            if !name_tokens.is_empty() {
                let overlap = name_tokens.iter().filter(|t| query_lower.contains(t.as_str())).count();
                let ratio = overlap as f64 / name_tokens.len() as f64;
                if ratio >= 0.5 {
                    score += 0.3 * ratio;
                }
            }
            if score >= threshold {
                scores.push((profile.branch_name.clone(), score));
            }
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    fn cosine_tfidf(&self, query_tf: &HashMap<String, f64>, query_total: f64, profile: &BranchProfile) -> f64 {
        let mut all_terms: HashSet<&String> = HashSet::new();
        all_terms.extend(query_tf.keys());
        all_terms.extend(profile.terms.keys());

        let mut dot = 0.0;
        let mut qn = 0.0;
        let mut pn = 0.0;
        for term in all_terms {
            let idf = self.idf.get(term).copied().unwrap_or(1.0);
            let q = query_tf.get(term).map(|c| (c / query_total) * idf).unwrap_or(0.0);
            let p = profile.tf(term) * idf;
            dot += q * p;
            qn += q * q;
            pn += p * p;
        }
        if qn == 0.0 || pn == 0.0 {
            0.0
        } else {
            dot / (qn.sqrt() * pn.sqrt())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_strips_stops_and_punct() {
        assert_eq!(tokenize("What did Caroline research?"), vec!["caroline", "research"]);
    }

    #[test]
    fn matcher_routes_to_right_branch() {
        let mut s = MemoryStore::new();
        s.create_branch("Adoption Journey", "researching adoption agencies for a child", "user", 1.0);
        s.create_branch("Art Projects", "painting a sunrise and sculpture work", "user", 1.0);
        let m = Matcher::build(&s);
        let hits = m.match_query("which adoption agencies did we look at", 3, 0.02);
        assert_eq!(hits[0].0, "Adoption Journey");
    }
}
