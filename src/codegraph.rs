//! `CodeGraphDriver` (`/workspace`), spec §3.2, the second VFS volume.
//!
//! Handles codebases. Uses lightweight structural parsing (functions, structs,
//! impls, classes, a "poor man's AST" over braces + signature lines) and
//! **exact BM25 sparse retrieval**. Dense embeddings are BANNED here (spec):
//! they wrongly match disparate code with similar syntax (`for i in 0..n` looks
//! like every other loop). Lexical identifiers, `parse_header`, `KvCache`,
//! `budget_tokens`, are the signal that actually locates code.
//!
//! Retrieval unit is a **symbol** (a function/type/impl block), not a line and
//! not a whole file: the granularity a "where is X / what does X do" query wants.

use std::collections::HashMap;
use std::path::Path;

use crate::driver::{Message, MemoryIndexDriver};

/// One indexed code unit, a function, struct, impl, class, etc.
#[derive(Clone, Debug)]
pub struct Symbol {
    pub id: usize,
    pub file: String,
    pub kind: SymbolKind,
    pub name: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Impl,
    Trait,
    Class,
    Other,
}

impl SymbolKind {
    fn label(&self) -> &'static str {
        match self {
            SymbolKind::Function => "fn",
            SymbolKind::Struct => "struct",
            SymbolKind::Enum => "enum",
            SymbolKind::Impl => "impl",
            SymbolKind::Trait => "trait",
            SymbolKind::Class => "class",
            SymbolKind::Other => "sym",
        }
    }
}

pub struct CodeGraphDriver {
    namespace: String,
    symbols: Vec<Symbol>,
    /// BM25 over symbol text, split on code-identifier boundaries.
    postings: HashMap<String, Vec<(usize, f32)>>,
    doc_len: Vec<f32>,
    avgdl: f32,
    pub last_path: std::cell::RefCell<String>,
}

const K1: f32 = 1.2;
const B: f32 = 0.75;
/// Keep symbols scoring >= REL_FLOOR × top-hit, and above an absolute MIN_SCORE.
/// Together they make weakly-matched (absent-topic) queries return few/no
/// blocks so the model faults rather than hallucinating from junk context.
const REL_FLOOR: f32 = 0.45;
const MIN_SCORE: f32 = 0.8;

impl CodeGraphDriver {
    pub fn new(namespace: &str) -> Self {
        CodeGraphDriver {
            namespace: namespace.to_string(),
            symbols: Vec::new(),
            postings: HashMap::new(),
            doc_len: Vec::new(),
            avgdl: 0.0,
            last_path: std::cell::RefCell::new(String::new()),
        }
    }

    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }

    pub fn symbol(&self, idx: usize) -> Option<&Symbol> {
        self.symbols.get(idx)
    }

    /// Index a source file: extract symbols, add each to BM25.
    pub fn ingest_file(&mut self, path: &str, source: &str) {
        let lang = Lang::of(path);
        for sym in extract_symbols(path, source, lang) {
            self.add_symbol(sym);
        }
        self.rebuild_bm25();
    }

    fn add_symbol(&mut self, mut sym: Symbol) {
        sym.id = self.symbols.len();
        self.symbols.push(sym);
    }

    fn rebuild_bm25(&mut self) {
        self.postings.clear();
        self.doc_len = Vec::with_capacity(self.symbols.len());
        for (pos, sym) in self.symbols.iter().enumerate() {
            // Containers (impl/trait/class) are indexed by SIGNATURE only, their
            // bodies are the methods, which are indexed as their own symbols;
            // counting the body here would let the container outrank the precise
            // method on the method's own terms.
            let body_for_index = if matches!(sym.kind, SymbolKind::Impl | SymbolKind::Trait | SymbolKind::Class) {
                sym.text.lines().next().unwrap_or("").to_string()
            } else {
                sym.text.clone()
            };
            // Weight the symbol NAME heavily, it's the strongest locator.
            let mut toks = code_tokenize(&body_for_index);
            for _ in 0..4 {
                toks.extend(code_tokenize(&sym.name));
            }
            self.doc_len.push(toks.len() as f32);
            let mut tf: HashMap<String, f32> = HashMap::new();
            for t in toks {
                *tf.entry(t).or_insert(0.0) += 1.0;
            }
            for (term, f) in tf {
                self.postings.entry(term).or_default().push((pos, f));
            }
        }
        let n = self.symbols.len().max(1) as f32;
        self.avgdl = self.doc_len.iter().sum::<f32>() / n;
    }

    fn bm25(&self, query: &str, k: usize) -> Vec<usize> {
        if self.symbols.is_empty() {
            return Vec::new();
        }
        let n = self.symbols.len() as f32;
        let mut scores: HashMap<usize, f32> = HashMap::new();
        for term in code_tokenize(query) {
            let Some(posts) = self.postings.get(&term) else { continue };
            let df = posts.len() as f32;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(pos, tf) in posts {
                let dl = self.doc_len[pos];
                let denom = tf + K1 * (1.0 - B + B * dl / self.avgdl.max(1.0));
                *scores.entry(pos).or_insert(0.0) += idf * tf * (K1 + 1.0) / denom;
            }
        }
        let mut ranked: Vec<(usize, f32)> = scores.into_iter().filter(|(_, s)| *s > 0.0).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);
        // Relevance gate: keep only symbols scoring within a band of the top
        // hit. A query about code that isn't here has no strong match, so this
        // returns FEW or ZERO symbols, the driver declines to page in junk,
        // which lets the model page-fault instead of hallucinating from
        // irrelevant context. Also sharpens answers (no dilution by off-topic
        // blocks).
        let top = ranked.first().map(|(_, s)| *s).unwrap_or(0.0);
        let floor = (top * REL_FLOOR).max(MIN_SCORE);
        ranked.retain(|(_, s)| *s >= floor);
        ranked.into_iter().map(|(p, _)| p).collect()
    }
}

impl MemoryIndexDriver for CodeGraphDriver {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    fn ingest_messages(&mut self, _messages: &[Message]) {
        // CodeGraphDriver indexes files, not chat turns; see ingest_file.
    }

    fn ingest_turn(&mut self, _speaker: &str, _text: &str, _timestamp: &str) -> usize {
        // Not applicable for a code volume.
        0
    }

    /// Route BY BM25 ONLY. `query_embedding` is deliberately ignored, dense
    /// retrieval is banned in /workspace (spec §3.2).
    fn route_query(&self, query_text: &str, _query_embedding: &[f32]) -> Vec<usize> {
        const TOP_SYMBOLS: usize = 6;
        let hits = self.bm25(query_text, TOP_SYMBOLS);
        *self.last_path.borrow_mut() = format!("bm25 symbols: {}", hits.len());
        hits
    }

    fn load_messages(&self, indices: &[usize], budget_tokens: usize) -> (String, usize) {
        let mut parts = Vec::new();
        let mut tokens = 0usize;
        for &idx in indices {
            let Some(sym) = self.symbols.get(idx) else { continue };
            let header = format!(
                "// {} {} — {}:{}-{}",
                sym.kind.label(),
                sym.name,
                sym.file,
                sym.start_line,
                sym.end_line
            );
            let unit = format!("{header}\n{}", sym.text);
            let t = unit.len() / 4;
            if tokens + t > budget_tokens {
                break;
            }
            parts.push(unit);
            tokens += t;
        }
        (parts.join("\n\n"), tokens)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Lang {
    Rust,
    Python,
    CLike,
    Other,
}

impl Lang {
    fn of(path: &str) -> Lang {
        match Path::new(path).extension().and_then(|e| e.to_str()) {
            Some("rs") => Lang::Rust,
            Some("py") => Lang::Python,
            Some("c") | Some("h") | Some("cpp") | Some("cc") | Some("hpp") | Some("js") | Some("ts") | Some("go") | Some("java") => Lang::CLike,
            _ => Lang::Other,
        }
    }
}

/// "Poor man's AST": scan for definition lines and capture their body by
/// brace balance (Rust/C-like) or indentation (Python). Not a real parser, but
/// it partitions a file into named symbol units, enough for retrieval.
fn extract_symbols(path: &str, source: &str, lang: Lang) -> Vec<Symbol> {
    let lines: Vec<&str> = source.lines().collect();
    let mut symbols = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if let Some((kind, name)) = def_signature(line, lang) {
            let (body, end) = match lang {
                Lang::Python => capture_by_indent(&lines, i),
                _ => capture_by_braces(&lines, i),
            };
            symbols.push(Symbol {
                id: 0,
                file: path.to_string(),
                kind,
                name,
                start_line: i + 1,
                end_line: end + 1,
                text: body,
            });
            // Containers (impl/trait/class) hold methods, descend into them so
            // each nested fn/def gets its own retrievable symbol. Leaf defs
            // (functions) skip their body to avoid re-parsing local blocks.
            let is_container = matches!(kind, SymbolKind::Impl | SymbolKind::Trait | SymbolKind::Class);
            if is_container {
                i += 1;
            } else {
                i = end + 1;
            }
        } else {
            i += 1;
        }
    }
    // Fallback: a file with no recognized symbols becomes one whole-file unit.
    if symbols.is_empty() && !source.trim().is_empty() {
        symbols.push(Symbol {
            id: 0,
            file: path.to_string(),
            kind: SymbolKind::Other,
            name: Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or("file").to_string(),
            start_line: 1,
            end_line: lines.len(),
            text: source.to_string(),
        });
    }
    symbols
}

fn def_signature(line: &str, lang: Lang) -> Option<(SymbolKind, String)> {
    let t = line.trim_start();
    let after = |kw: &str, s: &str| -> Option<String> {
        s.find(kw).map(|p| {
            let rest = &s[p + kw.len()..];
            rest.trim_start()
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .find(|w| !w.is_empty())
                .unwrap_or("")
                .to_string()
        })
    };
    match lang {
        Lang::Rust => {
            if t.contains("fn ") { return after("fn ", t).filter(|n| !n.is_empty()).map(|n| (SymbolKind::Function, n)); }
            if t.starts_with("struct ") { return after("struct ", t).map(|n| (SymbolKind::Struct, n)); }
            if t.starts_with("enum ") { return after("enum ", t).map(|n| (SymbolKind::Enum, n)); }
            if t.starts_with("trait ") { return after("trait ", t).map(|n| (SymbolKind::Trait, n)); }
            if t.starts_with("impl") { return Some((SymbolKind::Impl, impl_name(t))); }
        }
        Lang::Python => {
            if t.starts_with("def ") { return after("def ", t).map(|n| (SymbolKind::Function, n)); }
            if t.starts_with("class ") { return after("class ", t).map(|n| (SymbolKind::Class, n)); }
        }
        Lang::CLike => {
            if t.starts_with("function ") { return after("function ", t).map(|n| (SymbolKind::Function, n)); }
            if t.starts_with("class ") { return after("class ", t).map(|n| (SymbolKind::Class, n)); }
            if t.starts_with("struct ") { return after("struct ", t).map(|n| (SymbolKind::Struct, n)); }
            // C-style `type name(args) {`
            if t.contains('(') && t.trim_end().ends_with('{') && !t.starts_with("if") && !t.starts_with("for") && !t.starts_with("while") && !t.starts_with("switch") {
                if let Some(name) = c_func_name(t) {
                    return Some((SymbolKind::Function, name));
                }
            }
        }
        Lang::Other => {}
    }
    None
}

fn impl_name(line: &str) -> String {
    // `impl Foo` / `impl Trait for Foo` -> the concrete type after `for`, else after `impl`.
    let s = line.trim_start_matches("impl").trim();
    let target = s.split(" for ").last().unwrap_or(s);
    target.split(|c: char| !c.is_alphanumeric() && c != '_')
        .find(|w| !w.is_empty())
        .unwrap_or("impl")
        .to_string()
}

fn c_func_name(line: &str) -> Option<String> {
    let before_paren = line.split('(').next()?;
    before_paren.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .last()
        .map(|s| s.to_string())
}

fn capture_by_braces(lines: &[&str], start: usize) -> (String, usize) {
    let mut depth = 0i32;
    let mut seen_open = false;
    let mut end = start;
    for (k, line) in lines.iter().enumerate().skip(start) {
        for ch in line.chars() {
            match ch {
                '{' => { depth += 1; seen_open = true; }
                '}' => depth -= 1,
                _ => {}
            }
        }
        end = k;
        if seen_open && depth <= 0 {
            break;
        }
        // Signature-only line (e.g. trait fn `fn f();`), one line.
        if !seen_open && line.trim_end().ends_with(';') {
            break;
        }
        if k - start > 400 {
            break; // safety
        }
    }
    (lines[start..=end.min(lines.len() - 1)].join("\n"), end)
}

fn capture_by_indent(lines: &[&str], start: usize) -> (String, usize) {
    let base = indent(lines[start]);
    let mut end = start;
    for (k, line) in lines.iter().enumerate().skip(start + 1) {
        if line.trim().is_empty() {
            end = k;
            continue;
        }
        if indent(line) <= base {
            break;
        }
        end = k;
        if k - start > 400 {
            break;
        }
    }
    (lines[start..=end.min(lines.len() - 1)].join("\n"), end)
}

fn indent(line: &str) -> usize {
    line.chars().take_while(|c| *c == ' ' || *c == '\t').count()
}

/// Tokenize code: split identifiers on non-alnum AND on camelCase / snake_case
/// so `parseHeader` and `parse_header` both yield `parse` + `header`.
pub fn code_tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if cur.len() > 1 {
            out.push(std::mem::take(cur));
        } else {
            cur.clear();
        }
    };
    let chars: Vec<char> = text.chars().collect();
    for i in 0..chars.len() {
        let c = chars[i];
        if c.is_ascii_alphanumeric() {
            // camelCase boundary: lower->Upper
            if c.is_ascii_uppercase() && i > 0 && chars[i - 1].is_ascii_lowercase() {
                flush(&mut cur, &mut out);
            }
            cur.push(c.to_ascii_lowercase());
        } else {
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUST_SRC: &str = r#"
use std::collections::HashMap;

/// Parse the wire header.
fn parse_header(buf: &[u8]) -> Header {
    let magic = read_u32(buf);
    Header { magic }
}

struct KvCache {
    blocks: Vec<Block>,
}

impl KvCache {
    fn evict_block(&mut self, id: usize) {
        self.blocks.remove(id);
    }
}
"#;

    #[test]
    fn extracts_rust_symbols() {
        let mut d = CodeGraphDriver::new("/workspace");
        d.ingest_file("net.rs", RUST_SRC);
        let names: Vec<&str> = d.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"parse_header"), "got {names:?}");
        assert!(names.contains(&"KvCache"), "got {names:?}");
        assert!(names.contains(&"evict_block"), "got {names:?}");
    }

    #[test]
    fn bm25_routes_by_identifier_not_syntax() {
        let mut d = CodeGraphDriver::new("/workspace");
        d.ingest_file("net.rs", RUST_SRC);
        // Query with the exact identifier -> that symbol first.
        let hits = d.route_query("where do we evict a block from the cache", &[]);
        assert!(!hits.is_empty());
        let top = &d.symbols[hits[0]];
        assert_eq!(top.name, "evict_block", "top was {}", top.name);
    }

    #[test]
    fn camelcase_and_snake_tokenize_alike() {
        assert!(code_tokenize("parseHeader").contains(&"parse".to_string()));
        assert!(code_tokenize("parse_header").contains(&"header".to_string()));
    }

    #[test]
    fn python_indent_capture() {
        let src = "class Store:\n    def get(self, k):\n        return self.d[k]\n\ndef helper():\n    pass\n";
        let mut d = CodeGraphDriver::new("/workspace");
        d.ingest_file("store.py", src);
        let names: Vec<&str> = d.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Store"));
        assert!(names.contains(&"helper"));
    }

    #[test]
    fn load_messages_renders_symbol_headers() {
        let mut d = CodeGraphDriver::new("/workspace");
        d.ingest_file("net.rs", RUST_SRC);
        let hits = d.route_query("parse_header", &[]);
        let (ctx, _) = d.load_messages(&hits, 1000);
        assert!(ctx.contains("// fn parse_header"));
        assert!(ctx.contains("read_u32"));
    }
}
