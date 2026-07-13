//! Phase-2 PoC: per-block KV, encode once, place anywhere.
//!
//! The claim under test (the thing HTTP slots CANNOT do): a memory block
//! encoded at position 0, saved to disk, can later be restored and
//! RoPE-shifted to an arbitrary absolute offset, stitched next to *another*
//! independently-encoded block, and correctly attended to during generation -
//! WITHOUT re-prefilling either block's text.
//!
//! Test: two facts live in two separate blocks. We save each block's KV once
//! (encoded at pos 0). Then, in a fresh context, we assemble them -
//! block A at [0..a), block B RoPE-shifted to [a..a+b) via seq_cp+seq_add -
//! and ask a question whose answer is in block B. If the model answers from
//! the *shifted, restored* KV, per-block paging works.
//!
//! Usage: kvpoc <model.gguf>

use std::path::PathBuf;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

const BLOCK_A: &str = "[MEMORY /social] Caroline went to the LGBTQ support group on 7 May 2023. \
She found it welcoming and made new friends there.";
const BLOCK_B: &str = "[MEMORY /social] Melanie painted a lake sunrise in 2022. \
The password to the shed is NIGHTHAWK. Melanie signed up for pottery on 2 July 2023.";

fn main() {
    let model_path = std::env::args().nth(1).expect("usage: kvpoc <model.gguf>");
    let slot_dir = PathBuf::from("kv_blocks");
    std::fs::create_dir_all(&slot_dir).unwrap();

    let backend = LlamaBackend::init().unwrap();
    let model = LlamaModel::load_from_file(&backend, &model_path, &LlamaModelParams::default())
        .expect("load model");
    let ctx_params = LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(4096));

    // Tokenize both blocks once.
    let toks_a = model.str_to_token(BLOCK_A, AddBos::Always).unwrap();
    let toks_b = model.str_to_token(BLOCK_B, AddBos::Never).unwrap();
    let (a, b) = (toks_a.len(), toks_b.len());
    println!("block A: {a} tokens | block B: {b} tokens");

    // --- OFFLINE: encode each block at position 0, save its KV to disk ---
    for (name, toks) in [("blockA.kv", &toks_a), ("blockB.kv", &toks_b)] {
        let mut ctx = model.new_context(&backend, ctx_params.clone()).unwrap();
        let mut batch = LlamaBatch::new(toks.len().max(8), 1);
        for (i, t) in toks.iter().enumerate() {
            batch.add(*t, i as i32, &[0], false).unwrap();
        }
        ctx.decode(&mut batch).unwrap();
        let path = slot_dir.join(name);
        let bytes = ctx.state_seq_save_file(&path, 0, toks).unwrap();
        println!("saved {name}: {} tok, {} MB (encoded at pos 0)", toks.len(), bytes / 1_048_576);
    }

    // --- RUNTIME: assemble a fresh context from the two saved blocks ---
    // Block A into seq 0 at its native positions [0..a).
    // Block B into seq 1, then RoPE-shift +a so it occupies [a..a+b), then
    // copy seq 1 -> seq 0 so both live in one attention stream.
    let mut ctx = model.new_context(&backend, ctx_params.clone()).unwrap();

    let (loaded_a, _) = ctx
        .state_seq_load_file(&slot_dir.join("blockA.kv"), 0, a + 8)
        .expect("restore A");
    let (loaded_b, _) = ctx
        .state_seq_load_file(&slot_dir.join("blockB.kv"), 1, b + 8)
        .expect("restore B");
    println!("restored A={} into seq0, B={} into seq1", loaded_a.len(), loaded_b.len());

    // RoPE re-rotation: shift block B's cached positions by +a (this is the
    // "place anywhere" operation, llama_memory_seq_add updates K by the delta).
    ctx.kv_cache_seq_add(1, Some(0), Some(b as u32), a as i32).unwrap();
    // Merge seq 1 into seq 0 so one sequence holds A[0..a) + B[a..a+b).
    ctx.copy_kv_cache_seq(1, 0, Some(a as u32), Some((a + b) as u32)).unwrap();
    ctx.clear_kv_cache_seq(Some(1), None, None).ok();

    // Now decode a query at position a+b onward, attending to the stitched KV.
    let question = "\nQ: What is the password to the shed? Answer in one word.\nA:";
    let q_toks = model.str_to_token(question, AddBos::Never).unwrap();
    let mut batch = LlamaBatch::new(q_toks.len().max(8), 1);
    let mut pos = (a + b) as i32;
    for (i, t) in q_toks.iter().enumerate() {
        let last = i == q_toks.len() - 1;
        batch.add(*t, pos, &[0], last).unwrap();
        pos += 1;
    }
    ctx.decode(&mut batch).unwrap();

    // Greedy-generate the answer.
    let mut sampler = LlamaSampler::greedy();
    let mut out = String::new();
    let mut logits_idx = (batch.n_tokens() - 1) as i32;
    for _ in 0..12 {
        let tok = sampler.sample(&ctx, logits_idx);
        if model.is_eog_token(tok) {
            break;
        }
        out.push_str(&model.token_to_str(tok, Special::Tokenize).unwrap_or_default());
        let mut nb = LlamaBatch::new(1, 1);
        nb.add(tok, pos, &[0], true).unwrap();
        ctx.decode(&mut nb).unwrap();
        pos += 1;
        logits_idx = 0;
    }

    println!("\n=== stitched-KV answer ===");
    println!("Q: password to the shed?  (fact lives in block B, RoPE-shifted to [{a}..{}))", a + b);
    println!("A:{out}");
    let ok = out.to_uppercase().contains("NIGHTHAWK");
    println!("\nVERDICT: {}", if ok { "PASS: model read the shifted, restored block" } else { "FAIL: answer not from block B" });
    std::process::exit(if ok { 0 } else { 1 });
}
