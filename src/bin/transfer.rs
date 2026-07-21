//! Cross-domain transfer test: does the page-fault fine-tune (trained ONLY on
//! /social conversation data) work on /workspace CODE blocks with NO retraining?
//!
//! Hypothesis: the fine-tune taught a FORMAT-level protocol (answer from the
//! [MEMORY_BLOCK] or emit CONTEXT_NEEDED), not conversation content. If so, the
//! same behavior should hold when the block contains code instead of chat.
//!
//! Indexes this very crate's Rust source via CodeGraphDriver, routes code
//! questions through BM25, assembles a /workspace block, and checks:
//!   ANSWERABLE , the relevant symbol is loaded  -> should ANSWER
//!   FAULT      , ask about code not in the repo  -> should CONTEXT_NEEDED
//!
//! Usage: transfer [--model M]

use continuum::codegraph::CodeGraphDriver;
use continuum::driver::MemoryIndexDriver;
use continuum::kernel::{detect_page_fault, Kernel, KernelConfig};
use continuum::llamaserver::LlamaServer;
use continuum::ollama::{ChatMessage, Ollama};

fn main() {
    let mut model = "aios-ft-r2-full".to_string();
    let mut use_server = false;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model = args.get(i + 1).cloned().unwrap_or(model); i += 2; }
            "--server" => { use_server = true; i += 1; }
            _ => i += 1,
        }
    }

    // Index the crate's own source.
    let mut driver = CodeGraphDriver::new("/workspace");
    let mut files = 0;
    for entry in std::fs::read_dir("src").unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let src = std::fs::read_to_string(&path).unwrap();
            driver.ingest_file(path.to_str().unwrap(), &src);
            files += 1;
        }
    }
    eprintln!("[indexed {files} files, {} symbols from src/]", driver.symbol_count());

    // (question, is_answerable_in_repo)
    let cases: &[(&str, bool)] = &[
        ("What does the eviction_score function compute?", true),
        ("How does the TreeNode enum represent a leaf versus a branch?", true),
        ("What does parse_msg_date do with a timestamp string?", true),
        ("How does the BM25 index score a query?", true),
        ("What Redis connection pool settings does this codebase use?", false),
        ("How does the GraphQL subscription resolver handle backpressure?", false),
        ("Where is the Kubernetes deployment manifest applied?", false),
    ];

    let ollama = Ollama::new(&model, "nomic-embed-text");
    let mut kernel = Kernel::new(ollama.clone(), KernelConfig::default());
    if use_server {
        let s = LlamaServer::new(8080);
        if s.healthy() { kernel.set_kv_backend(s); }
    }

    // We drive the driver + prompt directly (kernel.query embeds for /social
    // routing; /workspace is BM25-only, so we assemble here and call the model).
    let system_tmpl = continuum::kernel::SYSTEM_TEMPLATE;
    let (mut ans_ok, mut ans_total, mut fault_ok, mut fault_total) = (0, 0, 0, 0);

    for (q, answerable) in cases {
        let hits = driver.route_query(q, &[]);
        let (block, _) = driver.load_messages(&hits, 2000);
        let ctx = format!("[MEMORY_BLOCK: /workspace/src]\n{block}");
        let system = system_tmpl.replace("{context}", &ctx);
        let msgs = [ChatMessage::new("system", system), ChatMessage::new("user", *q)];
        let resp = ollama.chat(&msgs, 4096, 200).unwrap_or_default();
        let faulted = detect_page_fault(&resp).is_some();

        let correct = if *answerable { !faulted } else { faulted };
        if *answerable { ans_total += 1; if correct { ans_ok += 1; } }
        else { fault_total += 1; if correct { fault_ok += 1; } }

        eprintln!(
            "[{}] {} | symbols={} | {}",
            if *answerable { "ANSₐ" } else { "FAULT" },
            if correct { "PASS" } else { "FAIL" },
            hits.len(),
            &q[..q.len().min(52)]
        );
        eprintln!("     -> {}", resp.replace('\n', " ").chars().take(90).collect::<String>());
    }

    println!("\n===== CROSS-DOMAIN TRANSFER (model: {model}) =====");
    println!("Trained on /social chat ONLY. Tested on /workspace code, no retraining.");
    println!("Answerable code Qs answered : {ans_ok}/{ans_total}");
    println!("Absent code Qs page-faulted : {fault_ok}/{fault_total}");
    let total = ans_ok + fault_ok;
    let n = ans_total + fault_total;
    println!("Overall protocol transfer    : {total}/{n} ({:.0}%)", 100.0 * total as f64 / n as f64);
    println!("=================================================");
}
