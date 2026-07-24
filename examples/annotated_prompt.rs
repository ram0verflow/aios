//! Dump the literal annotated working set for the drive and eng cases, so the
//! text the model actually receives can be inspected rather than described.
//!
//! Run: `cargo run --release --example annotated_prompt` (no model calls, no
//! network beyond the local embedder; runs keyword-only if Ollama is absent).

use continuum::driver::MemoryIndexDriver;
use continuum::hierarchical::HierarchicalTopicDriver;

const DRIVE: &[(&str, &str)] = &[
    ("The external drive holds 500 gigabytes.", "9:15 am on 2 March, 2023"),
    ("I'm currently keeping 140 gigabytes up there.", "10:02 am on 3 March, 2023"),
    ("The basic tier caps at 100 gigabytes.", "10:05 am on 3 March, 2023"),
    ("My photo library weighs in at 620 gigabytes.", "6:40 pm on 12 August, 2023"),
];

const ENG: &[(&str, &str)] = &[
    ("We budgeted for 12 engineers this year.", "11:00 am on 14 February, 2023"),
    ("There are 15 people on the platform team now.", "3:30 pm on 20 August, 2023"),
];

fn dump(title: &str, facts: &[(&str, &str)], annotate: bool) {
    let mut d = HierarchicalTopicDriver::new("/social");
    let mut idxs = Vec::new();
    for (text, ts) in facts {
        idxs.push(d.ingest_turn("user", text, ts));
    }
    d.route_cfg.annotate_values = annotate;
    let (ctx, tokens) = d.load_messages(&idxs, 4000);
    println!("\n===== {title} (annotate={annotate}, ~{tokens} tokens) =====");
    println!("{ctx}");
}

fn main() {
    dump("DRIVE CASE", DRIVE, false);
    dump("DRIVE CASE", DRIVE, true);
    dump("ENG CASE", ENG, false);
    dump("ENG CASE", ENG, true);
}
