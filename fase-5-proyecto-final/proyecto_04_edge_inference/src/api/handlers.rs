//! Controladores Axum para las rutas de visión, audio, salud y métricas.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::api::models::{
    AudioInferenceRequest, AudioInferenceResponse, ErrorResponse, HealthResponse, MetricsResponse,
    VisionClassifyRequest, VisionClassifyResponse,
};
use crate::audio::{
    AudioPostprocessor, AudioResampler, AudioTiming, MelSpectrogramExtractor, SpectrogramConfig,
    WavDecoder,
};
use crate::engine::{ModelKind, SessionPool};
use crate::queue::{BackpressureQueue, QueueError};
use crate::vision::{VisionPostprocessor, VisionPreprocessor, VisionTiming};

#[derive(Clone)]
pub struct AppState {
    pub queue: BackpressureQueue,
    pub session_pool: SessionPool,
    pub vision_preprocessor: Arc<VisionPreprocessor>,
    pub vision_postprocessor: Arc<VisionPostprocessor>,
    pub audio_extractor: Arc<MelSpectrogramExtractor>,
    pub audio_postprocessor: Arc<AudioPostprocessor>,
    pub start_time: Instant,
    pub total_requests: Arc<AtomicUsize>,
    pub successful_requests: Arc<AtomicUsize>,
    pub rejected_requests: Arc<AtomicUsize>,
}

impl AppState {
    pub fn new(queue: BackpressureQueue, session_pool: SessionPool) -> Self {
        Self {
            queue,
            session_pool,
            vision_preprocessor: Arc::new(VisionPreprocessor::default()),
            vision_postprocessor: Arc::new(VisionPostprocessor::default()),
            audio_extractor: Arc::new(MelSpectrogramExtractor::new(SpectrogramConfig::default())),
            audio_postprocessor: Arc::new(AudioPostprocessor::default()),
            start_time: Instant::now(),
            total_requests: Arc::new(AtomicUsize::new(0)),
            successful_requests: Arc::new(AtomicUsize::new(0)),
            rejected_requests: Arc::new(AtomicUsize::new(0)),
        }
    }
}

/// Clasificación de imágenes en tiempo real
pub async fn classify_vision_handler(
    State(state): State<AppState>,
    Json(payload): Json<VisionClassifyRequest>,
) -> Response {
    state.total_requests.fetch_add(1, Ordering::Relaxed);
    let t_total_start = Instant::now();

    // 1. Decodificación y preprocesamiento de imagen
    let t_pre_start = Instant::now();
    let img_res = state
        .vision_preprocessor
        .decode_base64(&payload.image_base64);

    let (img, original_dims) = match img_res {
        Ok(img) => {
            let dims = (img.width(), img.height());
            (img, dims)
        }
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Error en preprocesamiento de imagen: {e}"),
                    backpressure_active: false,
                }),
            )
                .into_response();
        }
    };

    let (canvas, _letterbox_info) = state.vision_preprocessor.letterbox(&img);
    let tensor_data = state.vision_preprocessor.to_nchw_tensor(&canvas);
    let decode_and_preprocess_ms = t_pre_start.elapsed().as_secs_f64() * 1000.0;

    // 2. Envío a la cola con contrapresión (Backpressure)
    let shape = vec![
        1,
        3,
        state.vision_preprocessor.target_h as usize,
        state.vision_preprocessor.target_w as usize,
    ];
    let task_id = uuid::Uuid::new_v4().to_string();

    let inference_res = state
        .queue
        .submit(
            task_id,
            ModelKind::Vision,
            shape,
            tensor_data,
            Duration::from_secs(5),
        )
        .await;

    let (logits, inference_us) = match inference_res {
        Ok(res) => res,
        Err(QueueError::Saturated { depth, capacity }) => {
            state.rejected_requests.fetch_add(1, Ordering::Relaxed);
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: format!(
                        "Microservicio saturado: Backpressure activo ({depth}/{capacity} solicitudes)"
                    ),
                    backpressure_active: true,
                }),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Fallo durante el procesamiento: {e}"),
                    backpressure_active: false,
                }),
            )
                .into_response();
        }
    };

    // 3. Postprocesamiento (Softmax y Top-K)
    let t_post_start = Instant::now();
    let probabilities = VisionPostprocessor::softmax(&logits);
    let top_k = payload.top_k.unwrap_or(3);
    let predictions = state
        .vision_postprocessor
        .extract_top_k(&probabilities, top_k);
    let postprocess_ms = t_post_start.elapsed().as_secs_f64() * 1000.0;
    let inference_ms = (inference_us as f64) / 1000.0;
    let total_ms = t_total_start.elapsed().as_secs_f64() * 1000.0;

    state.successful_requests.fetch_add(1, Ordering::Relaxed);

    let timing = VisionTiming {
        decode_and_preprocess_ms,
        inference_ms,
        postprocess_ms,
        total_ms,
    };

    (
        StatusCode::OK,
        Json(VisionClassifyResponse {
            success: true,
            predictions,
            timing,
            original_dimensions: original_dims,
        }),
    )
        .into_response()
}

/// Inferencia de audio WAV / Detección de comandos de voz
pub async fn infer_audio_handler(
    State(state): State<AppState>,
    Json(payload): Json<AudioInferenceRequest>,
) -> Response {
    state.total_requests.fetch_add(1, Ordering::Relaxed);
    let t_total_start = Instant::now();

    // 1. Decodificar Base64 del archivo WAV
    use base64::Engine;
    let clean_b64 = if let Some(idx) = payload.audio_base64.find(";base64,") {
        &payload.audio_base64[idx + 8..]
    } else {
        &payload.audio_base64
    };

    let wav_bytes = match base64::engine::general_purpose::STANDARD.decode(clean_b64.trim()) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Base64 de audio inválido: {e}"),
                    backpressure_active: false,
                }),
            )
                .into_response();
        }
    };

    // 2. Decodificar audio y remuestrear a 16kHz
    let t_decode_start = Instant::now();
    let (raw_samples, sample_rate) = match WavDecoder::decode_bytes(&wav_bytes) {
        Ok(res) => res,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Error decodificando WAV: {e}"),
                    backpressure_active: false,
                }),
            )
                .into_response();
        }
    };

    let target_rate = state.audio_extractor.config.sample_rate;
    let resampled = AudioResampler::resample(&raw_samples, sample_rate, target_rate);
    let duration_seconds = resampled.len() as f32 / target_rate as f32;
    let decode_and_resample_ms = t_decode_start.elapsed().as_secs_f64() * 1000.0;

    // 3. Extracción de espectrograma Mel
    let t_spec_start = Instant::now();
    let mel_tensor = state.audio_extractor.extract(&resampled);
    let spectrogram_extraction_ms = t_spec_start.elapsed().as_secs_f64() * 1000.0;

    // 4. Envío a la cola de inferencia con Backpressure
    let shape = vec![
        1,
        1,
        state.audio_extractor.config.n_mels,
        state.audio_extractor.config.target_frames,
    ];
    let task_id = uuid::Uuid::new_v4().to_string();

    let inference_res = state
        .queue
        .submit(
            task_id,
            ModelKind::Audio,
            shape,
            mel_tensor,
            Duration::from_secs(5),
        )
        .await;

    let (logits, inference_us) = match inference_res {
        Ok(res) => res,
        Err(QueueError::Saturated { depth, capacity }) => {
            state.rejected_requests.fetch_add(1, Ordering::Relaxed);
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: format!(
                        "Microservicio saturado: Backpressure activo ({depth}/{capacity} solicitudes)"
                    ),
                    backpressure_active: true,
                }),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("Fallo en inferencia de audio: {e}"),
                    backpressure_active: false,
                }),
            )
                .into_response();
        }
    };

    // 5. Postprocesamiento (Softmax y Top-K)
    let t_post_start = Instant::now();
    let probabilities = AudioPostprocessor::softmax(&logits);
    let top_k = payload.top_k.unwrap_or(3);
    let predictions = state
        .audio_postprocessor
        .extract_top_k(&probabilities, top_k);
    let postprocess_ms = t_post_start.elapsed().as_secs_f64() * 1000.0;
    let inference_ms = (inference_us as f64) / 1000.0;
    let total_ms = t_total_start.elapsed().as_secs_f64() * 1000.0;

    state.successful_requests.fetch_add(1, Ordering::Relaxed);

    let timing = AudioTiming {
        decode_and_resample_ms,
        spectrogram_extraction_ms,
        inference_ms,
        postprocess_ms,
        total_ms,
    };

    (
        StatusCode::OK,
        Json(AudioInferenceResponse {
            success: true,
            predictions,
            timing,
            sample_rate: target_rate,
            duration_seconds,
        }),
    )
        .into_response()
}

/// Estado de salud y métricas de saturación
pub async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    let pool_metrics = state.session_pool.get_stats();
    let queue_metrics = state.queue.stats();
    let uptime_seconds = state.start_time.elapsed().as_secs();

    Json(HealthResponse {
        status: "healthy".to_string(),
        uptime_seconds,
        pool: pool_metrics,
        queue: queue_metrics,
    })
}

/// Métricas agregadas de rendimiento y contrapresión
pub async fn metrics_handler(State(state): State<AppState>) -> Json<MetricsResponse> {
    let pool_metrics = state.session_pool.get_stats();
    let queue_metrics = state.queue.stats();

    Json(MetricsResponse {
        total_requests: state.total_requests.load(Ordering::Relaxed),
        successful_requests: state.successful_requests.load(Ordering::Relaxed),
        rejected_by_backpressure: state.rejected_requests.load(Ordering::Relaxed),
        pool: pool_metrics,
        queue: queue_metrics,
    })
}
