//! Módulo de Gestión de Memoria Paginada para KV-Cache (Estilo vLLM)

pub mod block_table;
pub mod paged_cache;

pub use block_table::BlockAllocator;
pub use paged_cache::PagedKvCacheManager;
