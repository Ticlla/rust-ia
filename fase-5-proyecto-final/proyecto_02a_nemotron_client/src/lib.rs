//! Crate del Cliente y REPL Interactivo para NVIDIA Nemotron en Rust.

pub mod client;
pub mod config;
pub mod metrics;
pub mod repl;
pub mod sse;
pub mod types;

pub use client::NemotronClient;
pub use config::{BackendMode, NemotronConfig};
pub use metrics::StreamTelemetry;
pub use repl::NemotronRepl;
pub use sse::{SseEvent, SseParser};
pub use types::{ChatMessage, Role};
