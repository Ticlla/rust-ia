//! Módulo de colas con contrapresión (Backpressure) y Dynamic Batching.

pub mod backpressure;
pub mod batcher;

pub use backpressure::{BackpressureQueue, InferenceTask, QueueError, QueueMetrics};
pub use batcher::InferenceBatcher;
