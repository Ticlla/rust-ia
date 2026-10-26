//! Configuración central del microservicio de inferencia de visión y audio en tiempo real.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeServiceConfig {
    /// Dirección IP de escucha
    pub host: String,
    /// Puerto de escucha
    pub port: u16,
    /// Ruta del modelo ONNX de visión
    pub vision_model_path: String,
    /// Ruta del modelo ONNX de audio
    pub audio_model_path: String,
    /// Tamaño del pool de sesiones ONNX
    pub pool_size: usize,
    /// Capacidad máxima de la cola con contrapresión (backpressure)
    pub queue_capacity: usize,
    /// Tamaño máximo de micro-batching dinámico
    pub max_batch_size: usize,
    /// Ventana de timeout en ms para acumular batches
    pub batch_timeout_ms: u64,
    /// Número de hilos intra-op para ONNX Runtime
    pub intra_threads: usize,
    /// Número de hilos inter-op para ONNX Runtime
    pub inter_threads: usize,
}

impl Default for EdgeServiceConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8082),
            vision_model_path: "models/vision_classifier.onnx".to_string(),
            audio_model_path: "models/audio_classifier.onnx".to_string(),
            pool_size: 4,
            queue_capacity: 64,
            max_batch_size: 8,
            batch_timeout_ms: 10,
            intra_threads: 2,
            inter_threads: 1,
        }
    }
}
