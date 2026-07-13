//! CacheBlend quality gate (design doc §3.2).
//!
//! Question: do KV blocks encoded INDEPENDENTLY (each at pos 0, no cross-block
//! attention) still answer correctly once restored, RoPE-shifted, and stitched?
//! Or do the missing cross-attention links at block boundaries corrupt answers?
//!
//! Method: build N memory blocks from real conv-26 message ranges. Ask a set
//! of questions two ways and compare:
//!   MONOLITHIC, all blocks prefilled contiguously (ground truth: full
//!                cross-attention, i.e. what text-paging produces today).
//!   STITCHED  , each block encoded at pos 0, saved, restored, shifted to its
//!                slot, seq_cp-merged. No re-prefill.
//! Agreement (stitched == monolithic on the gold fact) is the gate: high
//! agreement means block-independent encoding is safe at these sizes and the
//! CacheBlend boundary-recompute step is unnecessary.
//!
//! Usage: cacheblend <model.gguf> <conv_0.json>

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

const SYSTEM: &str = "You are an assistant with OS-managed memory. Answer ONLY from the loaded memory, in the shortest phrase possible.\n";

// (question, gold-substring the answer must contain)
const QA: &[(&str, &str)] = &[
    ("When did Caroline go to the LGBTQ support group?", "may"),
    ("What did Melanie paint?", "sunrise"),
    ("When did Melanie sign up for a pottery class?", "july"),
    ("What kind of books does Caroline like?", "child"),
    ("What has Caroline been researching?", "adopt"),
    ("What pet does Melanie have?", "dog"),
];

fn main() {
    let model_path = std::env::args().nth(1).expect("usage: cacheblend <model.gguf> <conv.json>");
    let conv_path = std::env::args().nth(2).expect("need conv.json");
    let slot_dir = PathBuf::from("kv_blocks");
    std::fs::create_dir_all(&slot_dir).unwrap();

    let conv: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&conv_path).unwrap()).unwrap();
    let msgs = conv["messages"].as_array().unwrap();
    let line = |m: &serde_json::Value| {
        format!("[{}] {}: {}",
            m["timestamp"].as_str().unwrap_or(""),
            m["speaker"].as_str().unwrap_or("").to_lowercase(),
            m["text"].as_str().unwrap_or(""))
    };

    // Three memory blocks from disjoint message ranges (the first ~150 msgs
    // cover support group / painting / pottery / books / adoption / pets).
    let ranges = [(0usize, 50usize), (50, 100), (100, 150)];
    let block_texts: Vec<String> = ranges.iter().map(|(s, e)| {
        let body: Vec<String> = msgs[*s..(*e).min(msgs.len())].iter().map(line).collect();
        format!("[MEMORY_BLOCK: /social/conv26_part{s}]\n{}", body.join("\n"))
    }).collect();

    let backend = LlamaBackend::init().unwrap();
    let model = LlamaModel::load_from_file(&backend, &model_path, &LlamaModelParams::default()).unwrap();
    let cp = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(8192));

    // Tokenize blocks: block 0 carries system+BOS, others are continuations.
    let mut block_toks: Vec<Vec<LlamaToken>> = Vec::new();
    for (i, bt) in block_texts.iter().enumerate() {
        let text = if i == 0 { format!("{SYSTEM}{bt}\n") } else { format!("{bt}\n") };
        let add = if i == 0 { AddBos::Always } else { AddBos::Never };
        block_toks.push(model.str_to_token(&text, add).unwrap());
    }
    let lens: Vec<usize> = block_toks.iter().map(|t| t.len()).collect();
    let offsets: Vec<usize> = lens.iter().scan(0, |a, l| { let o = *a; *a += l; Some(o) }).collect();
    let total: usize = lens.iter().sum();
    println!("blocks: {:?} tokens, total {total}", lens);

    // Save each block's KV, encoded at pos 0.
    for (i, toks) in block_toks.iter().enumerate() {
        let mut ctx = model.new_context(&backend, cp.clone()).unwrap();
        decode_chunked(&mut ctx, toks, 0);
        ctx.state_seq_save_file(&slot_dir.join(format!("cb_block{i}.kv")), 0, toks).unwrap();
    }

    // --- Build MONOLITHIC context (ground truth) ---
    let mut mono = model.new_context(&backend, cp.clone()).unwrap();
    {
        let all: Vec<LlamaToken> = block_toks.iter().flatten().copied().collect();
        decode_chunked(&mut mono, &all, 0);
    }

    // --- Build STITCHED context (per-block KV, shifted, merged) ---
    let mut stitch = model.new_context(&backend, cp.clone()).unwrap();
    for (i, _) in block_toks.iter().enumerate() {
        let seq = i as i32;
        stitch.state_seq_load_file(&slot_dir.join(format!("cb_block{i}.kv")), seq, lens[i] + 8).unwrap();
        if offsets[i] > 0 {
            // RoPE-shift block i to its slot, then merge into seq 0.
            stitch.kv_cache_seq_add(seq, Some(0), Some(lens[i] as u32), offsets[i] as i32).unwrap();
            stitch.copy_kv_cache_seq(seq, 0, Some(offsets[i] as u32), Some((offsets[i] + lens[i]) as u32)).unwrap();
            stitch.clear_kv_cache_seq(Some(seq as u32), None, None).ok();
        }
    }

    println!("\n{:<52} | {:^10} | {:^10} | match", "question", "monolithic", "stitched");
    println!("{}", "-".repeat(92));
    let (mut mono_ok, mut stitch_ok, mut agree) = (0, 0, 0);
    for (q, gold) in QA {
        let m = answer(&model, &mut mono, total, q);
        let s = answer(&model, &mut stitch, total, q);
        let mo = m.to_lowercase().contains(gold);
        let so = s.to_lowercase().contains(gold);
        let ag = norm(&m) == norm(&s);
        mono_ok += mo as i32; stitch_ok += so as i32; agree += ag as i32;
        println!("{:<52} | {:^10} | {:^10} | {}",
            &q[..q.len().min(52)],
            if mo {"OK"} else {"--"}, if so {"OK"} else {"--"},
            if ag {"="} else {"≠"});
        println!("    gold~{:<10} mono: {:<28} stitch: {}", gold, trunc(&m, 26), trunc(&s, 30));
    }
    let n = QA.len() as i32;
    println!("\n== GATE: mono {mono_ok}/{n} correct | stitched {stitch_ok}/{n} correct | agreement {agree}/{n} ==");
    println!("stitched retains {}% of monolithic correctness",
        if mono_ok > 0 { 100 * stitch_ok / mono_ok } else { 0 });
    std::process::exit(0);
}

/// Decode the question at the tail, greedy-generate, then clear the tail so the
/// shared block KV is reusable for the next question.
fn answer(model: &LlamaModel, ctx: &mut LlamaContext, base: usize, q: &str) -> String {
    let prompt = format!("\nQuestion: {q}\nAnswer:");
    let qt = model.str_to_token(&prompt, AddBos::Never).unwrap();
    let mut b = LlamaBatch::new(qt.len().max(8), 1);
    let mut pos = base as i32;
    for (i, t) in qt.iter().enumerate() {
        b.add(*t, pos, &[0], i == qt.len() - 1).unwrap();
        pos += 1;
    }
    ctx.decode(&mut b).unwrap();
    let mut sampler = LlamaSampler::greedy();
    let mut out = String::new();
    let mut idx = (b.n_tokens() - 1) as i32;
    for _ in 0..16 {
        let tok = sampler.sample(ctx, idx);
        if model.is_eog_token(tok) { break; }
        let piece = model.token_to_str(tok, Special::Tokenize).unwrap_or_default();
        // Stop at the first newline: one answer only, no run-on into the next Q.
        if !out.is_empty() && piece.contains('\n') { break; }
        out.push_str(&piece);
        let mut nb = LlamaBatch::new(1, 1);
        nb.add(tok, pos, &[0], true).unwrap();
        ctx.decode(&mut nb).unwrap();
        pos += 1; idx = 0;
    }
    // Clear the tail (question + generation) so blocks are reusable.
    ctx.clear_kv_cache_seq(Some(0), Some(base as u32), None).ok();
    out.split('\n').next().unwrap_or("").trim().to_string()
}

/// Prefill `toks` starting at `start_pos` into seq 0, in <=256-token ubatches
/// (a single decode cannot exceed n_ubatch=512).
fn decode_chunked(ctx: &mut LlamaContext, toks: &[LlamaToken], start_pos: i32) {
    const CHUNK: usize = 256;
    let mut pos = start_pos;
    for chunk in toks.chunks(CHUNK) {
        let mut b = LlamaBatch::new(chunk.len(), 1);
        for t in chunk {
            b.add(*t, pos, &[0], false).unwrap();
            pos += 1;
        }
        ctx.decode(&mut b).unwrap();
    }
}

fn norm(s: &str) -> String { s.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect() }
fn trunc(s: &str, n: usize) -> String { s.chars().take(n).collect() }
