//! Capa de API Web compatible con la especificación de OpenAI

pub mod handlers;
pub mod openai_types;
pub mod sse;

pub use handlers::build_router;
