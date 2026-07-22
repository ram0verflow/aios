//! Continuum, a Memory Operating System kernel for LLMs.
//!
//! Library root. The Rust realization of the Continuum Architecture Spec: a
//! domain-agnostic [`kernel`] over pluggable [`driver`]s (VFS volumes), backed
//! by a versioned four-level [`store`] and a demotion-based [`eviction`] policy.

pub mod codegraph;
pub mod driver;
pub mod eviction;
pub mod hierarchical;
pub mod http;
pub mod kernel;
pub mod llamaserver;
pub mod matcher;
pub mod metrics;
pub mod ollama;
pub mod probe_util;
pub mod server;
pub mod store;
