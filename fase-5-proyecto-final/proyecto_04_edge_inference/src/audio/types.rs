//! Tipos de datos para el pipeline de audio y extracción de espectrogramas Mel.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioKeywordPrediction {
    pub keyword_id: usize,
    pub keyword: String,
    pub probability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTiming {
    pub decode_and_resample_ms: f64,
    pub spectrogram_extraction_ms: f64,
    pub inference_ms: f64,
    pub postprocess_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInferenceResult {
    pub predictions: Vec<AudioKeywordPrediction>,
    pub timing: AudioTiming,
    pub sample_rate: u32,
    pub duration_seconds: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrogramConfig {
    /// Frecuencia de muestreo estándar (ej. 16,000 Hz para modelos de voz)
    pub sample_rate: u32,
    /// Tamaño de ventana para la Transformada Rápida de Fourier (ej. 400 = 25 ms)
    pub n_fft: usize,
    /// Salto entre ventanas consecutivas (ej. 160 = 10 ms)
    pub hop_length: usize,
    /// Número de bancos de filtros Mel (ej. 80 coeficientes)
    pub n_mels: usize,
    /// Número objetivo de tramas temporales (ej. 100 tramas = 1 segundo de audio)
    pub target_frames: usize,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,
            target_frames: 100,
        }
    }
}
