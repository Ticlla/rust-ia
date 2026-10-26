//! Módulo del motor de inferencia ONNX Runtime con pooling de sesiones.

pub mod session_pool;
pub mod types;

pub use session_pool::{PoolMetrics, SessionPool, SingleModelPool};
pub use types::{EngineError, ModelKind, TensorInput};
