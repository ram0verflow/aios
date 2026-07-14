//! `aios serve`: the companion as a local web app.
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
    store: MemoryStore,
    window: ContextWindow<'static>,
    turn: u64,
    model: String,
    kv: bool,
}

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

    let mut kernel = Kernel::new(ollama, KernelConfig::default());
    kernel.mount(Box::new(driver));

    // KV backend is optional: mounted only when llama-server is already up.
    let kv_server = LlamaServer::new(8080);
    let kv = kv_server.healthy();
    if kv {
        kernel.set_kv_backend(kv_server);
    }

    let mut c = Companion {
        kernel,
        store,
        window: ContextWindow::new(1200, None),
        turn: 0,
        model: model.to_string(),
        kv,
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
    let (status, ctype, out) = match (method.as_str(), path.as_str()) {
        ("GET", "/") => ("200 OK", "text/html; charset=utf-8", UI_HTML.to_string()),
        ("GET", "/api/status") => ("200 OK", "application/json", status_json(c).to_string()),
        ("GET", "/api/memory") => ("200 OK", "application/json", memory_json(c).to_string()),
        ("POST", "/api/chat") => match chat_turn(c, &body) {
            Ok(v) => ("200 OK", "application/json", v.to_string()),
            Err(e) => ("500 Internal Server Error", "application/json", json!({"error": e}).to_string()),
        },
        _ => ("404 Not Found", "text/plain", "not found".to_string()),
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{out}",
        out.len()
    );
}

/// One conversation turn: page in, answer, remember, evict, persist.
fn chat_turn(c: &mut Companion, body: &str) -> Result<Value, String> {
    let req: Value = serde_json::from_str(body).map_err(|e| e.to_string())?;
    let message = req["message"].as_str().unwrap_or("").trim().to_string();
    if message.is_empty() {
        return Err("empty message".into());
    }
    c.turn += 1;

    // Session context: evicted summary first, then whatever is still in RAM.
    let mut session: Vec<ChatMessage> = Vec::new();
    if !c.window.evicted_summary.is_empty() {
        session.push(ChatMessage::new(
            "system",
            format!("[PREVIOUS CONTEXT] {}", c.window.evicted_summary),
        ));
    }
    for slot in &c.window.slots {
        if let Some((role, content)) = slot.content.split_once(": ") {
            if role == "user" || role == "assistant" {
                session.push(ChatMessage::new(role, content));
            }
        }
    }

    let result = c.kernel.query(&message, &session);
    let reply = result.response.trim().to_string();

    // Memory formation, then feed the exchange back into the index.
    let remembered = c.kernel.write_back(
        &mut c.store,
        &message,
        &reply,
        &today_timestamp(),
        c.turn as f64,
    );

    // Window bookkeeping and eviction under pressure.
    c.window.load_message("user", &message, false);
    c.window.load_message("assistant", &reply, false);
    let mut evicted = 0;
    if c.window.pressure_level() != "OK" {
        let before = c.window.total_evictions;
        c.window.evict_messages(4);
        evicted = c.window.total_evictions - before;
    }
    for (branch, role, content) in c.window.drain_demotions() {
        c.store.add_archive(&branch, &role, &content, c.turn as f64);
    }

    // Persist both halves of memory every turn. Cheap at personal scale.
    c.store.save(STORE_PATH).ok();
    if let Some(d) = c.kernel.driver() {
        d.persist(DRIVER_PATH).ok();
    }

    Ok(json!({
        "reply": reply,
        "turn": c.turn,
        "faulted": result.page_faulted,
        "fault_topic": result.fault_topic,
        "retried": result.fault_retried,
        "loaded": result.messages_loaded,
        "namespace": result.namespace,
        "remembered": remembered.iter().map(|w| json!({
            "kind": w.kind, "content": w.content, "branch": w.branch,
        })).collect::<Vec<_>>(),
        "evicted": evicted,
        "pressure": pressure_json(c),
    }))
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
