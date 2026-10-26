//! # Proyecto 04: Edge Vision & Audio Real-Time Inference Microservice
//!
//! Microservicio industrial de inferencia en tiempo real para Visión Artificial y Audio en el Borde (Edge),
//! implementado en **Rust puro** con **ONNX Runtime (`ort`)**, pooling de sesiones, colas con contrapresión
//! (Backpressure) y Dynamic Batching.

pub mod api;
pub mod audio;
pub mod config;
pub mod engine;
pub mod queue;
pub mod vision;

pub use config::EdgeServiceConfig;
pub use engine::{ModelKind, SessionPool};
pub use queue::{BackpressureQueue, InferenceBatcher};
