//! llama-server backend, the KV-paging inference engine (Phase 1).
//!
//! Same GGUF weights as Ollama (we point llama-server at the Ollama blob),
//! but with the syscalls Ollama doesn't expose: per-slot KV state save /
//! restore / erase. This is what turns "page fault" from re-prefilling text
//! into mapping attention states back from disk.
//!
//! Server launch (slot files land in `kv_slots/`):
//! ```bash
//! llama-server -m ~/.ollama/models/blobs/<digest> --port 8080 -c 8192 \
//!   --slots --slot-save-path kv_slots/ -np 1
//! ```

use serde_json::json;

use crate::http;
use crate::ollama::ChatMessage;

pub struct LlamaServer {
    pub port: u16,
}

#[derive(Debug, Clone, Copy)]
pub struct SlotSave {
    pub tokens: u64,
    pub bytes: u64,
}

impl LlamaServer {
    pub fn new(port: u16) -> Self {
        LlamaServer { port }
    }

    pub fn healthy(&self) -> bool {
        http::get_json(self.port, "/health")
            .map(|v| v.get("status").and_then(|s| s.as_str()) == Some("ok"))
            .unwrap_or(false)
    }

    /// Chat completion via the OpenAI-compatible endpoint so the model's own
    /// chat template is applied server-side (the fine-tune was trained with
    /// it). `cache_prompt` keeps the slot's KV warm across turns.
    pub fn chat(&self, messages: &[ChatMessage], n_predict: usize) -> Result<String, String> {
        let body = json!({
            "messages": messages,
            "max_tokens": n_predict,
            "temperature": 0.0,
            "cache_prompt": true,
        });
        let resp = http::post_json(self.port, "/v1/chat/completions", &body)?;
        resp.get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| format!("unexpected chat response: {resp}"))
    }

    /// Persist slot KV to `<slot-save-path>/<filename>`. Returns tokens/bytes.
    pub fn save_slot(&self, slot: u32, filename: &str) -> Result<SlotSave, String> {
        let resp = http::post_json(
            self.port,
            &format!("/slots/{slot}?action=save"),
            &json!({ "filename": filename }),
        )?;
        Ok(SlotSave {
            tokens: resp.get("n_saved").and_then(|v| v.as_u64()).unwrap_or(0),
            bytes: resp.get("n_written").and_then(|v| v.as_u64()).unwrap_or(0),
        })
    }

    /// Map KV back from disk into the slot. Returns tokens restored.
    pub fn restore_slot(&self, slot: u32, filename: &str) -> Result<u64, String> {
        let resp = http::post_json(
            self.port,
            &format!("/slots/{slot}?action=restore"),
            &json!({ "filename": filename }),
        )?;
        resp.get("n_restored")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("unexpected restore response: {resp}"))
    }

    pub fn erase_slot(&self, slot: u32) -> Result<(), String> {
        http::post_json(self.port, &format!("/slots/{slot}?action=erase"), &json!({}))?;
        Ok(())
    }
}
