//! Infinite-chat stress test: one merged store holding ALL 10 LoCoMo
//! conversations (~4,200 messages, ~10× the single-conv eval), interrogated
//! with conv-26's questions. Degradation vs the single-conv baseline is the
//! interference cost, the thing that actually limits lifelong memory.
//!
//! Modes:
//!   --mode preload  conv-26 keeps its pre-built tree; convs 1-9 are appended
//!                   as embedded distractors (isolates read-path interference)
//!   --mode online   driver starts EMPTY; every message of all 10 convs is
//!                   ingested turn-by-turn through the incremental-indexing
//!                   path, tree grown from nothing (the future-state test)
//!
//! Usage: stress [--mode preload|online] [--limit N] [--model M] [--no-judge]

use std::time::Instant;

use aios::driver::{MemoryIndexDriver, Message};
use aios::hierarchical::HierarchicalTopicDriver;
use aios::kernel::{Kernel, KernelConfig};
use aios::metrics::{judge, rouge_l, rouge_n};
use aios::ollama::Ollama;

fn main() {
    let mut mode = "preload".to_string();
    let mut limit = 199usize;
    let mut model = "llama3.1:8b".to_string();
    let mut use_judge = true;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => { mode = args.get(i + 1).cloned().unwrap_or(mode); i += 2; }
            "--limit" => { limit = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(limit); i += 2; }
            "--model" => { model = args.get(i + 1).cloned().unwrap_or(model); i += 2; }
            "--no-judge" => { use_judge = false; i += 1; }
            _ => i += 1,
        }
    }

    let locomo: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("data/locomo10.json").expect("read locomo"),
    )
    .expect("json");
    let convs = locomo.as_array().expect("array");

    let ollama = Ollama::new(&model, "nomic-embed-text");
    if !ollama.healthy() {
        eprintln!("ERROR: Ollama not reachable.");
        std::process::exit(1);
    }
    let judge_handle = Ollama::new("llama3.1:8b", "nomic-embed-text");

    // ---- Build the merged store ----
    let t0 = Instant::now();
    let mut driver = match mode.as_str() {
        "preload" => build_preload(convs, &ollama),
        "online" => build_online(convs, &ollama),
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(1);
        }
    };
    driver.set_embedder(ollama.clone());
    let build_secs = t0.elapsed().as_secs_f32();
    let tree_stats = driver
        .tree()
        .map(|t| (t.leaf_count(), t.depth(), t.message_count()))
        .unwrap_or((0, 0, 0));
    eprintln!(
        "[store built in {build_secs:.0}s: {} messages | tree: {} leaves, depth {}, {} in-tree]",
        driver.message_len(),
        tree_stats.0,
        tree_stats.1,
        tree_stats.2
    );

    let n_messages = driver.message_len();
    let mut kernel = Kernel::new(ollama, KernelConfig::default());
    kernel.mount(Box::new(driver));

    // ---- Interrogate with conv-26 (conv index 0) questions ----
    let qa = convs[0].get("qa").and_then(|v| v.as_array()).expect("qa");
    let mut n = 0usize;
    let mut sum_r1 = 0.0;
    let mut sum_rl = 0.0;
    let mut judged = 0usize;
    let mut judge_y = 0usize;
    let mut faults = 0usize;
    let mut lat_ms: Vec<f64> = Vec::new();
    let mut qid = 0usize;

    for q in qa.iter() {
        let question = q.get("question").and_then(|v| v.as_str()).unwrap_or("");
        let gold = match q.get("answer") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(v) => v.to_string(),
            None => continue, // unanswerable adversarial; not this test
        };
        qid += 1;
        if qid > limit {
            break;
        }
        let cat = q.get("category").map(|c| c.to_string()).unwrap_or_default();

        let tq = Instant::now();
        let result = kernel.query(question, &[]);
        lat_ms.push(tq.elapsed().as_secs_f64() * 1000.0);
        let answer = result.response.trim().to_string();
        n += 1;

        let r1 = rouge_n(&answer, &gold, 1);
        let rl = rouge_l(&answer, &gold);
        sum_r1 += r1;
        sum_rl += rl;
        if result.page_faulted {
            faults += 1;
        }
        let mut jmark = "-";
        if use_judge {
            if let Some(ok) = judge(&judge_handle, question, &gold, &answer) {
                judged += 1;
                if ok {
                    judge_y += 1;
                    jmark = "Y";
                } else {
                    jmark = "n";
                }
            }
        }
        eprintln!(
            "[{qid:>3}] cat{cat} RL={rl:.2} J={jmark} loaded={:>3} fault={} | {}",
            result.messages_loaded,
            if result.page_faulted { "Y" } else { "." },
            &question.chars().take(60).collect::<String>()
        );
    }

    lat_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = lat_ms.get(lat_ms.len() / 2).copied().unwrap_or(0.0);
    let p95 = lat_ms.get(lat_ms.len() * 95 / 100).copied().unwrap_or(0.0);

    println!("\n========== INFINITE-CHAT STRESS ({mode}) ==========");
    println!("store: {n_messages} messages (10 conversations merged)");
    println!("build time: {build_secs:.0}s");
    println!("questions (conv-26, answerable): {n}");
    println!("ROUGE-1/L: {:.4} / {:.4}", sum_r1 / n.max(1) as f64, sum_rl / n.max(1) as f64);
    if judged > 0 {
        println!("LLM-judge: {judge_y}/{judged} ({:.1}%)", 100.0 * judge_y as f64 / judged as f64);
    }
    println!("page faults: {faults}/{n}");
    println!("query latency p50/p95: {p50:.0}ms / {p95:.0}ms");
    println!("single-conv baseline: judge 78.9% | RL 0.443 | 14/154 faults");
    println!("====================================================");
}

/// Mode A: conv-26 with its pre-built tree; convs 1-9 as embedded distractors.
fn build_preload(convs: &[serde_json::Value], ollama: &Ollama) -> HierarchicalTopicDriver {
    let tree_data: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("data/conv_0.json").expect("read tree"),
    )
    .expect("json");
    let mut driver = HierarchicalTopicDriver::from_conv_json("/social", &tree_data, Some(ollama));
    let mut next_idx = driver.message_len();

    let mut batch = Vec::new();
    for conv in &convs[1..] {
        for (speaker, text, ts) in conv_turns(conv) {
            batch.push(Message {
                idx: next_idx,
                speaker,
                text: text.clone(),
                timestamp: ts,
                embedding: ollama.embed(&text).ok(),
            });
            next_idx += 1;
        }
    }
    eprintln!("[embedding+appending {} distractor messages…]", batch.len());
    driver.ingest_messages(&batch);
    driver
}

/// Mode B: everything ingested online, tree grown from nothing.
fn build_online(convs: &[serde_json::Value], ollama: &Ollama) -> HierarchicalTopicDriver {
    let mut driver = HierarchicalTopicDriver::new("/social");
    driver.set_embedder(ollama.clone());
    let mut total = 0usize;
    for conv in convs.iter() {
        for (speaker, text, ts) in conv_turns(conv) {
            driver.ingest_turn(&speaker, &text, &ts);
            total += 1;
            if total % 500 == 0 {
                eprintln!("[ingested {total} turns…]");
            }
        }
    }
    driver
}

/// Extract (speaker, text, session_timestamp) turns in order from one conv.
fn conv_turns(conv: &serde_json::Value) -> Vec<(String, String, String)> {
    let c = conv.get("conversation").and_then(|v| v.as_object()).expect("conversation");
    let mut session_nums: Vec<u32> = c
        .keys()
        .filter_map(|k| {
            k.strip_prefix("session_")
                .filter(|r| !r.contains("date"))
                .and_then(|r| r.parse().ok())
        })
        .collect();
    session_nums.sort_unstable();

    let mut out = Vec::new();
    for sn in session_nums {
        let ts = c
            .get(&format!("session_{sn}_date_time"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if let Some(turns) = c.get(&format!("session_{sn}")).and_then(|v| v.as_array()) {
            for t in turns {
                let speaker = t.get("speaker").and_then(|v| v.as_str()).unwrap_or("").to_lowercase();
                let text = t.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                if !text.is_empty() {
                    out.push((speaker, text, ts.clone()));
                }
            }
        }
    }
    out
}
