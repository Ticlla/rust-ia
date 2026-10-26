//! Modelos de solicitud y respuesta para la API REST del microservicio.

use serde::{Deserialize, Serialize};

use crate::audio::{AudioKeywordPrediction, AudioTiming};
use crate::engine::PoolMetrics;
use crate::queue::QueueMetrics;
use crate::vision::{VisionClassPrediction, VisionTiming};

#[derive(Debug, Clone, Deserialize)]
pub struct VisionClassifyRequest {
    /// Imagen codificada en formato Base64 (PNG, JPEG, etc.)
    pub image_base64: String,
    /// Número de predicciones Top-K a retornar (por defecto 3)
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VisionClassifyResponse {
    pub success: bool,
    pub predictions: Vec<VisionClassPrediction>,
    pub timing: VisionTiming,
    pub original_dimensions: (u32, u32),
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioInferenceRequest {
    /// Archivo de audio WAV codificado en Base64
    pub audio_base64: String,
    /// Número de predicciones Top-K a retornar (por defecto 3)
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioInferenceResponse {
    pub success: bool,
    pub predictions: Vec<AudioKeywordPrediction>,
    pub timing: AudioTiming,
    pub sample_rate: u32,
    pub duration_seconds: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime_seconds: u64,
    pub pool: PoolMetrics,
    pub queue: QueueMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricsResponse {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub rejected_by_backpressure: usize,
    pub pool: PoolMetrics,
    pub queue: QueueMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub backpressure_active: bool,
}
