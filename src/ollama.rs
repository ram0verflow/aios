//! Minimal, dependency-free Ollama client.
//!
//! Talks HTTP/1.1 to a local Ollama daemon over a raw `TcpStream`. We only ever
//! hit `127.0.0.1:11434`, so there is no TLS and no need to pull in `reqwest`.
//! Requests are sent with `Connection: close` and the whole response is read to
//! EOF, then de-chunked if necessary.

use serde_json::{json, Value};

use crate::http;

const PORT: u16 = 11434;

#[derive(Clone)]
pub struct Ollama {
    pub chat_model: String,
    pub embed_model: String,
}

#[derive(Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        ChatMessage { role: role.to_string(), content: content.into() }
    }
}

impl Ollama {
    pub fn new(chat_model: &str, embed_model: &str) -> Self {
        Ollama { chat_model: chat_model.to_string(), embed_model: embed_model.to_string() }
    }

    /// One non-streamed chat completion. Returns the assistant text.
    pub fn chat(
        &self,
        messages: &[ChatMessage],
        num_ctx: usize,
        num_predict: usize,
    ) -> Result<String, String> {
        let body = json!({
            "model": self.chat_model,
            "messages": messages,
            "stream": false,
            "options": { "num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0.0 }
        });
        let resp = self.post("/api/chat", &body)?;
        resp.get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| format!("unexpected chat response: {resp}"))
    }

    /// Embed a single string via the embedding model.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let body = json!({ "model": self.embed_model, "input": text });
        let resp = self.post("/api/embed", &body)?;
        // /api/embed returns {"embeddings": [[...]]}
        let arr = resp
            .get("embeddings")
            .and_then(|e| e.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("unexpected embed response: {resp}"))?;
        Ok(arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
    }

    /// True if the daemon answers and the models load.
    pub fn healthy(&self) -> bool {
        self.embed("ping").is_ok()
    }

    fn post(&self, path: &str, body: &Value) -> Result<Value, String> {
        http::post_json(PORT, path, body)
    }
}
