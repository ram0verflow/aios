//! `continuum serve`: the companion as a local web app.
//!
//! A small blocking HTTP server, localhost only, serving an embedded single
//! file UI plus a JSON API. No frameworks on either side. The UI is a chat
//! pane next to a memory panel that shows what the kernel actually did on
//! each turn: what got paged in, whether it faulted, what it decided to
//! remember, and how full the context window is.
//!
//! Memory persists to companion/ so the companion survives restarts:
//!   companion/store.json   the four level store (write-back targets)
//!   companion/driver.json  the retrieval index (messages, embeddings, tree)

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use serde_json::{json, Value};

use crate::eviction::ContextWindow;
use crate::hierarchical::{today_timestamp, HierarchicalTopicDriver};
use crate::kernel::{Kernel, KernelConfig};
use crate::llamaserver::LlamaServer;
use crate::ollama::{ChatMessage, Ollama};
use crate::store::MemoryStore;

const UI_HTML: &str = include_str!("ui.html");
const DIR: &str = "companion";
const STORE_PATH: &str = "companion/store.json";
const DRIVER_PATH: &str = "companion/driver.json";

struct Companion {
    kernel: Kernel,
    ollama: Ollama,
    store: MemoryStore,
    window: ContextWindow<'static>,
    turn: u64,
    model: String,
    kv: bool,
    kv_restored: u64,
}

const KV_SESSION: &str = "companion_session.kv";

pub fn run(port: u16, model: &str) {
    std::fs::create_dir_all(DIR).ok();

    let ollama = Ollama::new(model, "nomic-embed-text");
    if !ollama.healthy() {
        eprintln!("ollama is not reachable on :11434 (or models missing). start it first.");
        std::process::exit(1);
    }

    let store = MemoryStore::load(STORE_PATH).unwrap_or_default();
    let mut driver = HierarchicalTopicDriver::load(DRIVER_PATH)
        .unwrap_or_else(|_| HierarchicalTopicDriver::new("/social"));
    driver.set_embedder(ollama.clone());
    let n_msgs = driver.message_len();

    let mut kernel = Kernel::new(ollama.clone(), KernelConfig::default());
    kernel.mount(Box::new(driver));

    // KV backend is optional: mounted only when llama-server is already up.
    // When it is, restore last session's attention states from disk.
    let kv_server = LlamaServer::new(8080);
    let kv = kv_server.healthy();
    let mut kv_restored = 0;
    if kv {
        kernel.set_kv_backend(kv_server);
        kv_restored = kernel.restore_kv(KV_SESSION).unwrap_or(0);
    }

    let mut c = Companion {
        kernel,
        ollama,
        store,
        window: ContextWindow::new(1200, None),
        turn: 0,
        model: model.to_string(),
        kv,
        kv_restored,
    };

    let listener = TcpListener::bind(("127.0.0.1", port))
        .unwrap_or_else(|e| { eprintln!("cannot bind port {port}: {e}"); std::process::exit(1) });
    println!("companion up: http://localhost:{port}  (model {model}, kv {}, {} messages remembered)",
        if kv { "on" } else { "off" }, n_msgs);

    for stream in listener.incoming().flatten() {
        handle(stream, &mut c);
    }
}

fn handle(mut stream: TcpStream, c: &mut Companion) {
    let Some((method, path, body)) = read_request(&mut stream) else { return };

    // The chat endpoint streams; it writes its own response.
    if method == "POST" && path == "/api/chat" {
        chat_stream_turn(&mut stream, c, &body);
        return;
    }

    let (status, ctype, out) = match (method.as_str(), path.as_str()) {
        ("GET", "/") => ("200 OK", "text/html; charset=utf-8", UI_HTML.to_string()),
        ("GET", "/api/status") => ("200 OK", "application/json", status_json(c).to_string()),
        ("GET", "/api/memory") => ("200 OK", "application/json", memory_json(c).to_string()),
        ("POST", "/api/kv/save") => {
            let out = match c.kernel.save_kv(KV_SESSION) {
                Ok(s) => json!({"saved_tokens": s.tokens, "bytes": s.bytes}),
                Err(e) => json!({"error": e}),
            };
            ("200 OK", "application/json", out.to_string())
        }
        _ => ("404 Not Found", "text/plain", "not found".to_string()),
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{out}",
        out.len()
    );
}

fn sse(stream: &mut TcpStream, v: Value) {
    let _ = write!(stream, "data: {v}\n\n");
    let _ = stream.flush();
}

/// One conversation turn as a server sent event stream. Tokens go out as they
/// arrive; the first few are held back so a CONTEXT_NEEDED opener can be
/// intercepted and retried before the user sees it.
fn chat_stream_turn(stream: &mut TcpStream, c: &mut Companion, body: &str) {
    let message = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["message"].as_str().map(|s| s.trim().to_string()))
        .unwrap_or_default();
    if message.is_empty() {
        let _ = write!(stream, "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
        return;
    }
    c.turn += 1;
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    );

    // Session context from the window, same as the terminal chat.
    let mut session: Vec<ChatMessage> = Vec::new();
    if !c.window.evicted_summary.is_empty() {
        session.push(ChatMessage::new("system", format!("[PREVIOUS CONTEXT] {}", c.window.evicted_summary)));
    }
    for slot in &c.window.slots {
        if let Some((role, content)) = slot.content.split_once(": ") {
            if role == "user" || role == "assistant" {
                session.push(ChatMessage::new(role, content));
            }
        }
    }

    let (messages, meta) = c.kernel.prepare(&message, &session);
    sse(stream, json!({"t": "route", "loaded": meta.messages_loaded, "namespace": meta.namespace}));

    // First pass. When the KV backend is mounted the kernel path is used
    // non streaming (llama-server SSE parsing is not wired yet); otherwise
    // stream straight from ollama.
    let first = generate(c, stream, &messages);
    let mut reply = first.clone();
    let mut faulted = false;
    let mut retried = false;

    if let Some(topic) = crate::kernel::detect_page_fault(&first) {
        faulted = true;
        sse(stream, json!({"t": "fault", "topic": topic}));
        if let Some(retry_msgs) = c.kernel.prepare_fault(&topic, &message, &session, meta.memory_budget_tokens) {
            let second = generate(c, stream, &retry_msgs);
            if crate::kernel::detect_page_fault(&second).is_none() && !second.trim().is_empty() {
                reply = second;
                retried = true;
            }
        }
        if !retried {
            // Nothing better came back; show the fault text itself.
            sse(stream, json!({"t": "tok", "v": reply}));
        }
    }
    let reply = reply.trim().to_string();

    // Memory formation and window bookkeeping, same as before.
    let remembered = c.kernel.write_back(&mut c.store, &message, &reply, &today_timestamp(), c.turn as f64);
    for w in &remembered {
        sse(stream, json!({"t": "mem", "kind": w.kind, "content": w.content, "branch": w.branch}));
    }
    c.window.load_message("user", &message, false);
    c.window.load_message("assistant", &reply, false);
    if c.window.pressure_level() != "OK" {
        let before = c.window.total_evictions;
        c.window.evict_messages(4);
        let evicted = c.window.total_evictions - before;
        if evicted > 0 {
            sse(stream, json!({"t": "evict", "n": evicted}));
        }
    }
    for (branch, role, content) in c.window.drain_demotions() {
        c.store.add_archive(&branch, &role, &content, c.turn as f64);
    }
    c.store.save(STORE_PATH).ok();
    if let Some(d) = c.kernel.driver() {
        d.persist(DRIVER_PATH).ok();
    }

    sse(stream, json!({
        "t": "done",
        "reply": reply,
        "turn": c.turn,
        "faulted": faulted,
        "retried": retried,
        "loaded": meta.messages_loaded,
        "namespace": meta.namespace,
        "pressure": pressure_json(c),
    }));
}

/// Generate one completion, streaming tokens out as SSE. Holds the first few
/// pieces back so a CONTEXT_NEEDED opener never reaches the user mid word.
fn generate(c: &Companion, stream: &mut TcpStream, messages: &[ChatMessage]) -> String {
    if c.kernel.has_kv_backend() {
        // Non streaming fallback through the kernel's mounted backend.
        return c.kernel.complete_messages(messages).unwrap_or_else(|e| format!("[ERROR: {e}]"));
    }
    let Ok(sock) = stream.try_clone() else {
        return c.kernel.complete_messages(messages).unwrap_or_else(|e| format!("[ERROR: {e}]"));
    };
    let sock = std::cell::RefCell::new(sock);
    let mut held = String::new();
    let mut flushed = false;
    let out = c.ollama.chat_stream(
        messages,
        c.kernel.config.num_ctx,
        c.kernel.config.max_response_tokens,
        |piece| {
            if flushed {
                sse(&mut sock.borrow_mut(), json!({"t": "tok", "v": piece}));
                return;
            }
            held.push_str(piece);
            if held.len() >= 24 || held.contains('\n') {
                if !held.to_uppercase().contains("CONTEXT_NEEDED") {
                    sse(&mut sock.borrow_mut(), json!({"t": "tok", "v": held.as_str()}));
                    flushed = true;
                }
                // A fault opener stays held; the caller handles the retry.
            }
        },
    );
    match out {
        Ok(full) => {
            if !flushed && crate::kernel::detect_page_fault(&full).is_none() && !full.is_empty() {
                sse(stream, json!({"t": "tok", "v": held.as_str()}));
            }
            full
        }
        Err(e) => format!("[ERROR: {e}]"),
    }
}

fn pressure_json(c: &Companion) -> Value {
    json!({
        "used": c.window.used_tokens(),
        "budget": c.window.budget_tokens,
        "level": c.window.pressure_level(),
        "evictions": c.window.total_evictions,
    })
}

fn status_json(c: &Companion) -> Value {
    let s = c.store.stats();
    json!({
        "model": c.model,
        "kv": c.kv,
        "kv_restored": c.kv_restored,
        "turn": c.turn,
        "store": {
            "branches": s.branches,
            "details": s.details,
            "archive": s.archive_entries,
            "versions": s.total_versions,
        },
        "pressure": pressure_json(c),
    })
}

fn memory_json(c: &Companion) -> Value {
    let branches: Vec<Value> = c.store.all_branches().map(|b| json!({
        "name": b.name,
        "summary": b.summary.current(),
        "details": b.details.iter().rev().take(6).map(|d| d.current()).collect::<Vec<_>>(),
        "archive": b.archive.len(),
    })).collect();
    json!({
        "identity": c.store.get_identity(),
        "branches": branches,
    })
}

/// Minimal HTTP request reader: request line, headers, Content-Length body.
fn read_request(stream: &mut TcpStream) -> Option<(String, String, String)> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end;
    loop {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = crate::http::find_subslice(&buf, b"\r\n\r\n") {
            header_end = pos;
            break;
        }
        if buf.len() > 1_048_576 {
            return None;
        }
    }
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.lines();
    let mut first = lines.next()?.split_whitespace();
    let method = first.next()?.to_string();
    let path = first.next()?.to_string();

    let content_length: usize = head
        .lines()
        .find(|l| l.to_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);

    let mut body_bytes = buf[header_end + 4..].to_vec();
    while body_bytes.len() < content_length {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            break;
        }
        body_bytes.extend_from_slice(&tmp[..n]);
    }
    Some((method, path, String::from_utf8_lossy(&body_bytes).to_string()))
}
