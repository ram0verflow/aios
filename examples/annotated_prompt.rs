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

/// The discriminate harness plants EVERY case's mentions into one transcript,
/// so the working set for any single question also holds the other cases'
/// numbers. Reproduced faithfully here, because an isolated two-message version
/// has no same-type rival and selectivity correctly annotates nothing, which
/// hides the mechanism entirely.
const DSHIFT: &[(&str, &str)] = &[
    ("My dentist appointment is on October 14th.", "9:00 am on 1 October, 2023"),
    ("Heads up: I'm pushing everything in my calendar back by exactly one week.", "9:00 am on 2 October, 2023"),
    ("My rent is 1800 a month.", "9:00 am on 3 October, 2023"),
    ("My landlord told me everything goes up by 200 starting next month.", "9:00 am on 4 October, 2023"),
    ("My API plan allows 50 thousand requests per month.", "9:00 am on 5 October, 2023"),
    ("I've burned through about 62 thousand calls so far this month.", "9:00 am on 6 October, 2023"),
];

fn main() {
    dump("DATE-SHIFT CASE", DSHIFT, false);
    dump("DATE-SHIFT CASE", DSHIFT, true);
    dump("DRIVE CASE", DRIVE, false);
    dump("DRIVE CASE", DRIVE, true);
    dump("ENG CASE", ENG, false);
    dump("ENG CASE", ENG, true);
}
