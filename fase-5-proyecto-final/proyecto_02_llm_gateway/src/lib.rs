//! High-Throughput LLM Serving Gateway (vLLM en Rust)
//! Servidor de Inferencia OpenAI-Compatible con Paged KV-Cache y Continuous Batching

pub mod api;
pub mod config;
pub mod model;
pub mod paged_cache;
pub mod scheduler;

pub use config::GatewayConfig;
pub use model::{GatewayTokenizer, GatewayTransformer};
pub use paged_cache::{BlockAllocator, PagedKvCacheManager};
pub use scheduler::{ContinuousBatchScheduler, Sequence};
