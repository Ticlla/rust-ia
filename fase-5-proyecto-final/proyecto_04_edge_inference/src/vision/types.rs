//! Tipos de datos para el pipeline de inferencia de visión artificial.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionClassPrediction {
    pub class_id: usize,
    pub label: String,
    pub probability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionTiming {
    pub decode_and_preprocess_ms: f64,
    pub inference_ms: f64,
    pub postprocess_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionInferenceResult {
    pub predictions: Vec<VisionClassPrediction>,
    pub timing: VisionTiming,
    pub original_dimensions: (u32, u32),
    pub target_dimensions: (u32, u32),
}

#[derive(Debug, Clone)]
pub struct LetterboxInfo {
    pub scale: f32,
    pub pad_x: u32,
    pub pad_y: u32,
    pub new_w: u32,
    pub new_h: u32,
}
