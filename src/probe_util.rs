//! Shared labeling helper for the deterministic retrieval probes
//! (`examples/retrieval_probe.rs` and the Python attribution scripts mirror it).
//!
//! This exists to kill one specific recurring bug. The probes tag planted
//! facts as `"<case>.a"` / `"<case>.b"` (e.g. `"drive.a"`) and distractors as
//! `"d<index>"` (e.g. `"d17"`). To hide distractors the first cut filtered
//! `!label.starts_with('d')` — which also silently drops `"drive.a"` and
//! `"drive.b"`, because "drive" starts with 'd'. That bug produced a wrong
//! conclusion about the drive case twice (a display artifact once, a
//! `rank=None` once), caught both times by suspicion rather than a test.
//!
//! The correct discriminator is the dot: planted facts contain one, `d17`
//! does not. Route every probe filter through [`is_planted_fact_label`] so a
//! third recurrence is impossible — the regression test below asserts the
//! drive labels survive.

/// True for a planted-fact label (`"drive.a"`, `"eng.b"`), false for a
/// distractor label (`"d0"`, `"d17"`). The one predicate probes must filter
/// on. Never reintroduce a `starts_with('d')` test: "drive" starts with 'd'.
pub fn is_planted_fact_label(label: &str) -> bool {
    label.contains('.')
}

#[cfg(test)]
mod tests {
    use super::is_planted_fact_label;

    #[test]
    fn drive_labels_survive_filtering() {
        // The exact regression: "drive.*" must be kept, not eaten by a
        // leading-'d' filter. Assert every crux case, both slots.
        for case in ["drive", "storage", "eng", "laptop", "api", "flat", "grant", "gift"] {
            assert!(is_planted_fact_label(&format!("{case}.a")), "{case}.a dropped");
            assert!(is_planted_fact_label(&format!("{case}.b")), "{case}.b dropped");
        }
    }

    #[test]
    fn distractor_labels_are_filtered_out() {
        for i in 0..8 {
            assert!(!is_planted_fact_label(&format!("d{i}")), "d{i} kept");
        }
        assert!(!is_planted_fact_label("distractor"));
    }
}
