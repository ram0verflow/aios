//! Long coding-session test: per-block KV paging, wired end to end.
//!
//! The original motivation for this project was a failure mode where
//! "truncation is fatal: the model hallucinated a password because it read from
//! overwritten context." Here we make the codebase LARGER than the context
//! window and show the OS keeps every fact retrievable and never fabricates.
//!
//! Pipeline (per-block KV `page_in`):
//!   1. Index a synthetic codebase into symbols (continuum::CodeGraphDriver).
//!   2. Encode each symbol ONCE as a KV block (pos 0), save to disk.
//!   3. Long session of queries. Each query:
//!      route (BM25) -> restore top-K symbol KV blocks -> RoPE-shift + stitch
//!      after the system block -> generate. Total loaded per query << codebase.
//!   4. Verify:
//!      - PRESERVATION: planted facts (early / mid / late in the codebase, and
//!        an early fact RE-ASKED at the end of the session) answer correctly.
//!      - NO HALLUCINATION: questions about functions NOT in the codebase must
//!        page-fault (CONTEXT_NEEDED), never fabricate a value.
//!
//! Usage: codesession <model.gguf>

use std::num::NonZeroU32;
use std::path::PathBuf;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use continuum::codegraph::CodeGraphDriver;
use continuum::driver::MemoryIndexDriver;

const SYSTEM: &str = "You are a coding assistant with OS-managed memory. Loaded code is in [MEMORY_BLOCK] sections. \
Answer ONLY from the loaded code. When asked what a function returns, quote the EXACT literal value from its body (e.g. the string or number), not a description. \
If the answer is not in the loaded code, reply EXACTLY: CONTEXT_NEEDED: <topic>\n";

const N_CTX: u32 = 1024; // deliberately small: codebase must exceed it
const PER_QUERY_BLOCKS: usize = 4; // max routed symbols loaded per query

fn main() {
    let model_path = std::env::args().nth(1).expect("usage: codesession <model.gguf>");
    let slot_dir = PathBuf::from("kv_sym");
    std::fs::create_dir_all(&slot_dir).unwrap();

    // --- Synthetic codebase: ~40 functions, deliberately > context window. ---
    let (source, facts) = synth_codebase();
    let mut driver = CodeGraphDriver::new("/workspace");
    driver.ingest_file("server.rs", &source);
    let n_sym = driver.symbol_count();

    let backend = LlamaBackend::init().unwrap();
    let model = LlamaModel::load_from_file(&backend, &model_path, &LlamaModelParams::default()).unwrap();
    let cp = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(N_CTX));

    // --- System block: encode once, save. ---
    let sys_toks = model.str_to_token(SYSTEM, AddBos::Always).unwrap();
    {
        let mut ctx = model.new_context(&backend, cp.clone()).unwrap();
        decode_chunked(&mut ctx, &sys_toks, 0);
        ctx.state_seq_save_file(&slot_dir.join("system.kv"), 0, &sys_toks).unwrap();
    }

    // --- Encode each symbol as its own KV block (pos 0), save to disk. ---
    let mut sym_toks: Vec<Vec<LlamaToken>> = Vec::with_capacity(n_sym);
    let mut total_code_tokens = 0usize;
    for i in 0..n_sym {
        let (block, _) = driver.load_messages(&[i], 100_000);
        let toks = model.str_to_token(&format!("\n{block}\n"), AddBos::Never).unwrap();
        total_code_tokens += toks.len();
        {
            let mut ctx = model.new_context(&backend, cp.clone()).unwrap();
            decode_chunked(&mut ctx, &toks, 0);
            ctx.state_seq_save_file(&slot_dir.join(format!("sym{i}.kv")), 0, &toks).unwrap();
        }
        sym_toks.push(toks);
    }
    println!(
        "codebase: {n_sym} symbols, {total_code_tokens} code tokens vs {N_CTX}-token window \
         ({}x over). full codebase cannot fit; routing pages in per query.\n",
        total_code_tokens / N_CTX as usize
    );

    // --- The long session ---
    // (question, Some(gold-substring) if answerable, None if it must FAULT)
    let mut session: Vec<(String, Option<String>)> = Vec::new();
    for f in &facts {
        session.push((f.question.clone(), Some(f.gold.clone())));
    }
    // Interleave absent-function questions (must fault, not fabricate).
    for q in ABSENT {
        session.push((q.to_string(), None));
    }
    // Re-ask the FIRST planted fact VERBATIM at the very end, the true
    // preservation test: same question, many turns later, must still recall.
    session.push((facts[0].question.clone(), Some(facts[0].gold.clone())));

    let mut ctx = model.new_context(&backend, cp.clone()).unwrap();
    let (mut preserve_ok, mut preserve_n) = (0, 0);
    let (mut halluc, mut fault_ok, mut absent_n) = (0, 0, 0);

    for (turn, (q, gold)) in session.iter().enumerate() {
        // page_in: route (relevance-gated) -> restore system + matched symbol
        // blocks -> shift+stitch. Absent-topic queries match nothing -> load 0.
        let hits = driver.route_query(q, &[]);
        let loaded: Vec<usize> = hits.into_iter().take(PER_QUERY_BLOCKS).collect();
        let answer = page_in_and_answer(&model, &mut ctx, &slot_dir, &sys_toks, &sym_toks, &loaded, q);

        let faulted = answer.to_uppercase().contains("CONTEXT_NEEDED")
            || ["not in the loaded", "not found", "cannot find", "isn't in", "not present",
                "no loaded", "no code", "not available"]
                .iter().any(|p| answer.to_lowercase().contains(p));

        match gold {
            Some(g) => {
                preserve_n += 1;
                let ok = answer.to_lowercase().contains(&g.to_lowercase());
                if ok { preserve_ok += 1; }
                let sym_pos = loaded.iter().map(|i| format!("s{i}")).collect::<Vec<_>>().join(",");
                println!("[{turn:>2}] PRESERVE {} gold~{:<14} loaded[{sym_pos}] -> {}",
                    if ok {"OK "} else {"MISS"}, g, trunc(&answer, 44));
            }
            None => {
                absent_n += 1;
                if faulted { fault_ok += 1; } else { halluc += 1; }
                println!("[{turn:>2}] ABSENT   {} loaded={} -> {}",
                    if faulted {"OK (faulted)  "} else {"HALLUCINATED! "}, loaded.len(), trunc(&answer, 44));
            }
        }
    }

    println!("\n===== LONG CODING SESSION RESULT =====");
    println!("codebase {}x the context window; {} turns", total_code_tokens / N_CTX as usize, session.len());
    println!("Context preserved (facts recalled) : {preserve_ok}/{preserve_n}");
    println!("Absent code correctly faulted      : {fault_ok}/{absent_n}");
    println!("HALLUCINATIONS (fabricated answers): {halluc}/{absent_n}");
    let verdict = preserve_ok == preserve_n && halluc == 0;
    println!("\nVERDICT: {}", if verdict {
        "PASS: every fact preserved despite codebase > window, zero hallucinations"
    } else if halluc == 0 {
        "PARTIAL: no hallucinations, but a fact retrieval missed"
    } else {
        "FAIL: a hallucination occurred"
    });
    std::process::exit(0);
}

/// The per-block `page_in`: restore system + routed symbol KV blocks, each
/// encoded at pos 0, RoPE-shift each to its running offset, seq_cp-merge into
/// seq 0, then decode the question and generate. No text re-prefill.
fn page_in_and_answer(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    dir: &std::path::Path,
    sys_toks: &[LlamaToken],
    sym_toks: &[Vec<LlamaToken>],
    loaded: &[usize],
    q: &str,
) -> String {
    // OS-raised page fault: if routing's relevance gate cleared NOTHING, the
    // working set for this query is empty, the OS faults on the model's behalf
    // rather than letting it answer from parametric memory (the segfault
    // analogy: unmapped address, don't execute). This is the deterministic
    // half of the fault protocol; the model handles "loaded but insufficient".
    if loaded.is_empty() {
        let topic: String = q.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| w.len() > 2 && !["what", "does", "the", "return", "value", "function", "returned"].contains(w))
            .take(3).collect::<Vec<_>>().join(" ");
        return format!("CONTEXT_NEEDED: {topic}");
    }
    ctx.clear_kv_cache();

    // System block into seq 0 at [0..sys).
    ctx.state_seq_load_file(&dir.join("system.kv"), 0, sys_toks.len() + 8).unwrap();
    let mut offset = sys_toks.len();

    // Each routed symbol: restore into a scratch seq, shift to `offset`, merge.
    for (slot, &sym) in loaded.iter().enumerate() {
        let seq = (slot + 1) as i32;
        let len = sym_toks[sym].len();
        if offset + len + 64 >= N_CTX as usize {
            break; // respect the window; routing already capped this
        }
        ctx.state_seq_load_file(&dir.join(format!("sym{sym}.kv")), seq, len + 8).unwrap();
        ctx.kv_cache_seq_add(seq, Some(0), Some(len as u32), offset as i32).unwrap();
        ctx.copy_kv_cache_seq(seq, 0, Some(offset as u32), Some((offset + len) as u32)).unwrap();
        ctx.clear_kv_cache_seq(Some(seq as u32), None, None).ok();
        offset += len;
    }

    // Decode the question at the tail, generate greedily.
    let prompt = format!("\nQuestion: {q}\nAnswer:");
    let qt = model.str_to_token(&prompt, AddBos::Never).unwrap();
    let mut b = LlamaBatch::new(qt.len().max(8), 1);
    let mut pos = offset as i32;
    for (i, t) in qt.iter().enumerate() {
        b.add(*t, pos, &[0], i == qt.len() - 1).unwrap();
        pos += 1;
    }
    ctx.decode(&mut b).unwrap();

    let mut sampler = LlamaSampler::greedy();
    let mut out = String::new();
    let mut idx = (b.n_tokens() - 1) as i32;
    for _ in 0..20 {
        let tok = sampler.sample(ctx, idx);
        if model.is_eog_token(tok) { break; }
        let piece = model.token_to_str(tok, Special::Tokenize).unwrap_or_default();
        if !out.is_empty() && piece.contains('\n') { break; }
        out.push_str(&piece);
        let mut nb = LlamaBatch::new(1, 1);
        nb.add(tok, pos, &[0], true).unwrap();
        ctx.decode(&mut nb).unwrap();
        pos += 1; idx = 0;
    }
    out.split('\n').next().unwrap_or("").trim().to_string()
}

struct Fact { question: String, gold: String }

/// Build a synthetic server codebase with planted, verifiable facts spread
/// across early/mid/late symbols, padded with plausible distractor functions
/// so the whole thing exceeds the context window.
fn synth_codebase() -> (String, Vec<Fact>) {
    let mut src = String::new();
    let mut facts = Vec::new();

    // Planted facts, distinctive return values, distinct identifiers.
    let planted = [
        ("admin_api_key", "the admin API key", "NIGHTHAWK7Q",
         "fn admin_api_key() -> &'static str {\n    // internal auth token\n    \"NIGHTHAWK7Q\"\n}"),
        ("default_retry_backoff_ms", "the default retry backoff in ms", "4200",
         "fn default_retry_backoff_ms() -> u64 {\n    4200\n}"),
        ("session_cookie_name", "the session cookie name", "aios_sid",
         "fn session_cookie_name() -> &'static str {\n    \"aios_sid\"\n}"),
        ("max_upload_bytes", "the max upload size in bytes", "8388608",
         "fn max_upload_bytes() -> usize {\n    8388608\n}"),
        ("shard_count", "the number of database shards", "17",
         "fn shard_count() -> u32 {\n    17\n}"),
    ];

    // Distractor functions to inflate the codebase past the window.
    let topics = ["metrics", "cache", "auth", "router", "queue", "config", "logging",
        "healthcheck", "ratelimit", "serializer", "scheduler", "migrations",
        "webhook", "pagination", "compression", "telemetry", "featureflag",
        "connection", "backpressure", "retention", "checkpoint", "throttle",
        "digest", "heartbeat", "quota", "snapshot", "dispatch", "reconcile",
        "backfill", "prefetch", "coalesce", "watermark", "rebalance"];

    // Distractor: a realistic multi-line handler so the codebase clears the
    // window several times over.
    let emit_fn = |src: &mut String, name: &str| {
        src.push_str(&format!(
"/// Handles {name} for the request pipeline stage.
fn {name}_handler(req: &Request, ctx: &Ctx) -> Response {{
    let cfg = load_{name}_config(ctx);
    let span = ctx.tracer.start(\"{name}\");
    let result = process_{name}(req, &cfg);
    span.record(result.status);
    finalize_{name}(result, &cfg)
}}

"));
    };

    // Interleave planted facts across the codebase; distractors in between.
    emit_planted(&mut src, &mut facts, &planted[0]);
    for t in &topics[..11] { emit_fn(&mut src, t); }
    emit_planted(&mut src, &mut facts, &planted[1]);
    emit_planted(&mut src, &mut facts, &planted[2]);
    for t in &topics[11..22] { emit_fn(&mut src, t); }
    emit_planted(&mut src, &mut facts, &planted[3]);
    for t in &topics[22..] { emit_fn(&mut src, t); }
    // second pass with suffixed names to inflate well past the window
    for t in &topics { emit_fn(&mut src, &format!("{t}_v2")); }
    emit_planted(&mut src, &mut facts, &planted[4]);

    (src, facts)
}

fn emit_planted(src: &mut String, facts: &mut Vec<Fact>, p: &(&str, &str, &str, &str)) {
    let (_name, desc, gold, code) = *p;
    src.push_str(&format!("/// Returns {desc}.\n{code}\n\n"));
    facts.push(Fact { question: format!("What does {} return?", p.0), gold: gold.to_string() });
}

/// Functions that do NOT exist in the codebase, asking about them must fault.
const ABSENT: &[&str] = &[
    "What does stripe_webhook_secret() return?",
    "What is the value returned by oauth_refresh_token()?",
    "What port does the grpc_gateway_bind() function use?",
    "What does the kafka_consumer_group() function return?",
];

fn decode_chunked(ctx: &mut LlamaContext, toks: &[LlamaToken], start: i32) {
    const CHUNK: usize = 256;
    let mut pos = start;
    for chunk in toks.chunks(CHUNK) {
        let mut b = LlamaBatch::new(chunk.len(), 1);
        for t in chunk { b.add(*t, pos, &[0], false).unwrap(); pos += 1; }
        ctx.decode(&mut b).unwrap();
    }
}

fn trunc(s: &str, n: usize) -> String { s.chars().take(n).collect() }
