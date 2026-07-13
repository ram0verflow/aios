//! Minimal dependency-free HTTP/1.1 client over TcpStream, shared by the
//! Ollama and llama-server backends. Localhost only, no TLS.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use serde_json::Value;

const HOST: &str = "127.0.0.1";

pub fn post_json(port: u16, path: &str, body: &Value) -> Result<Value, String> {
    let payload = serde_json::to_vec(body).map_err(|e| e.to_string())?;
    request(port, &format!(
        "POST {path} HTTP/1.1\r\nHost: {HOST}:{port}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    ), Some(&payload))
}

pub fn get_json(port: u16, path: &str) -> Result<Value, String> {
    request(port, &format!(
        "GET {path} HTTP/1.1\r\nHost: {HOST}:{port}\r\nConnection: close\r\n\r\n"
    ), None)
}

fn request(port: u16, header: &str, payload: Option<&[u8]>) -> Result<Value, String> {
    let mut stream = TcpStream::connect((HOST, port)).map_err(|e| format!("connect {HOST}:{port}: {e}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(600))).map_err(|e| e.to_string())?;
    stream.set_write_timeout(Some(Duration::from_secs(60))).map_err(|e| e.to_string())?;

    stream.write_all(header.as_bytes()).map_err(|e| e.to_string())?;
    if let Some(p) = payload {
        stream.write_all(p).map_err(|e| e.to_string())?;
    }
    stream.flush().map_err(|e| e.to_string())?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).map_err(|e| e.to_string())?;

    let split = find_subslice(&raw, b"\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response (no header/body split)".to_string())?;
    let head = String::from_utf8_lossy(&raw[..split]);
    let body_bytes = &raw[split + 4..];

    let status_ok = head.lines().next().map(|l| l.contains(" 200")).unwrap_or(false);
    let chunked = head.to_lowercase().contains("transfer-encoding: chunked");
    let decoded = if chunked { dechunk(body_bytes) } else { body_bytes.to_vec() };

    if !status_ok {
        return Err(format!(
            "HTTP error: {}\n{}",
            head.lines().next().unwrap_or(""),
            String::from_utf8_lossy(&decoded)
        ));
    }
    serde_json::from_slice(&decoded)
        .map_err(|e| format!("json parse: {e}\nbody: {}", String::from_utf8_lossy(&decoded)))
}

pub fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Decode HTTP/1.1 chunked transfer encoding.
pub fn dechunk(mut data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let Some(nl) = find_subslice(data, b"\r\n") else { break };
        let size_str = String::from_utf8_lossy(&data[..nl]);
        let size = usize::from_str_radix(size_str.trim(), 16).unwrap_or(0);
        if size == 0 {
            break;
        }
        let start = nl + 2;
        let end = start + size;
        if end > data.len() {
            break;
        }
        out.extend_from_slice(&data[start..end]);
        data = &data[end + 2..];
    }
    out
}
