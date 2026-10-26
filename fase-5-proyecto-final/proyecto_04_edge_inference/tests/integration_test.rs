//! Suite de pruebas de integración para el microservicio Edge Vision & Audio.

use std::time::Duration;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::Engine;
use serde_json::Value;
use tower::ServiceExt;

use proyecto_04_edge_inference::api::{create_router, AppState};
use proyecto_04_edge_inference::audio::{
    AudioPostprocessor, AudioResampler, MelSpectrogramExtractor, SpectrogramConfig, WavDecoder,
};
use proyecto_04_edge_inference::config::EdgeServiceConfig;
use proyecto_04_edge_inference::engine::{ModelKind, SessionPool};
use proyecto_04_edge_inference::queue::{BackpressureQueue, InferenceBatcher, QueueError};
use proyecto_04_edge_inference::vision::{VisionPostprocessor, VisionPreprocessor};

const VISION_MODEL_BYTES: &[u8] = include_bytes!("../models/vision_classifier.onnx");
const AUDIO_MODEL_BYTES: &[u8] = include_bytes!("../models/audio_classifier.onnx");

fn create_test_system() -> (AppState, axum::Router) {
    let mut config = EdgeServiceConfig::default();
    config.pool_size = 2;
    config.queue_capacity = 8;
    config.max_batch_size = 4;
    config.batch_timeout_ms = 5;

    let pool = SessionPool::new(&config, VISION_MODEL_BYTES, AUDIO_MODEL_BYTES).unwrap();
    let (queue, receiver) = BackpressureQueue::new(config.queue_capacity);
    let _worker = InferenceBatcher::spawn(
        receiver,
        pool.clone(),
        config.max_batch_size,
        Duration::from_millis(config.batch_timeout_ms),
    );

    let state = AppState::new(queue, pool);
    let router = create_router(state.clone());
    (state, router)
}

// -----------------------------------------------------------------------------
// TEST 1: Pipeline de Visión (Letterboxing, Normalización y NCHW)
// -----------------------------------------------------------------------------
#[test]
fn test_vision_pipeline_letterbox_and_nchw() {
    let preprocessor = VisionPreprocessor::new(224, 224);

    // Generar imagen PNG sintética de 320x240 (relación 4:3)
    let png_bytes = VisionPreprocessor::generate_synthetic_png(320, 240);
    assert!(!png_bytes.is_empty());

    let (tensor, orig_dims, letterbox) = preprocessor.process_bytes(&png_bytes).unwrap();
    assert_eq!(orig_dims, (320, 240));

    // Verificar dimensiones exactas NCHW: 3 canales * 224 * 224
    assert_eq!(tensor.len(), 3 * 224 * 224);
    assert_eq!(letterbox.new_w, 224);
    assert!(letterbox.pad_y > 0, "Debe existir padding vertical para mantener aspecto 4:3");

    // Verificar Softmax y Top-K
    let dummy_logits = vec![2.5f32, 1.0, -0.5, 3.2, 0.1];
    let probs = VisionPostprocessor::softmax(&dummy_logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);

    let postprocessor = VisionPostprocessor::default();
    let top_k = postprocessor.extract_top_k(&probs, 2);
    assert_eq!(top_k.len(), 2);
    assert_eq!(top_k[0].class_id, 3); // logit más alto es 3.2 en índice 3
    assert!(top_k[0].probability > top_k[1].probability);
}

// -----------------------------------------------------------------------------
// TEST 2: Pipeline de Audio (Decodificación WAV, Remuestreo y Mel Spectrogram)
// -----------------------------------------------------------------------------
#[test]
fn test_audio_pipeline_wav_and_mel_spectrogram() {
    // 1. Sintetizar tono senoidal de 440 Hz a 44.1 kHz (1 segundo)
    let wav_bytes = WavDecoder::synthesize_test_wav(44100, 1.0, 440.0);
    let (samples_44k, sample_rate) = WavDecoder::decode_bytes(&wav_bytes).unwrap();
    assert_eq!(sample_rate, 44100);
    assert_eq!(samples_44k.len(), 44100);

    // 2. Remuestrear a 16 kHz
    let samples_16k = AudioResampler::resample(&samples_44k, 44100, 16000);
    assert!((samples_16k.len() as i32 - 16000).abs() <= 2);

    // 3. Extracción de espectrograma Mel
    let spec_config = SpectrogramConfig::default();
    let extractor = MelSpectrogramExtractor::new(spec_config.clone());
    let mel_tensor = extractor.extract(&samples_16k);

    // Debe coincidir exactamente con [1, 1, 80, 100] = 8000 floats
    assert_eq!(mel_tensor.len(), 1 * 1 * 80 * 100);
    assert!(mel_tensor.iter().all(|x| x.is_finite()));

    // 4. Postprocesamiento de palabras clave
    let dummy_audio_logits = vec![0.1f32, 4.0, 0.2, 1.1, -1.0];
    let probs = AudioPostprocessor::softmax(&dummy_audio_logits);
    let audio_post = AudioPostprocessor::default();
    let preds = audio_post.extract_top_k(&probs, 3);
    assert_eq!(preds.len(), 3);
    assert_eq!(preds[0].keyword_id, 1); // logit más alto en índice 1
    assert_eq!(preds[0].keyword, "parada_emergencia");
}

// -----------------------------------------------------------------------------
// TEST 3: Pool Concurrente de Sesiones ONNX Runtime (Multi-Threaded)
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_session_pool_concurrent_execution() {
    let mut config = EdgeServiceConfig::default();
    config.pool_size = 4;
    let pool = SessionPool::new(&config, VISION_MODEL_BYTES, AUDIO_MODEL_BYTES).unwrap();

    let mut handles = Vec::new();

    // Lanzar 8 tareas concurrentes de visión y audio
    for i in 0..8 {
        let pool_clone = pool.clone();
        let handle = tokio::spawn(async move {
            if i % 2 == 0 {
                let shape = vec![1, 3, 224, 224];
                let dummy_data = vec![0.2f32; 3 * 224 * 224];
                let (logits, _us) = pool_clone.run(ModelKind::Vision, &shape, dummy_data).await.unwrap();
                assert_eq!(logits.len(), 5);
            } else {
                let shape = vec![1, 1, 80, 100];
                let dummy_data = vec![0.1f32; 8000];
                let (logits, _us) = pool_clone.run(ModelKind::Audio, &shape, dummy_data).await.unwrap();
                assert_eq!(logits.len(), 5);
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.await.unwrap();
    }

    let stats = pool.get_stats();
    assert_eq!(stats.vision_total_inferences + stats.audio_total_inferences, 8);
}

// -----------------------------------------------------------------------------
// TEST 4: Cola de Inferencia con Contrapresión (Backpressure Rejection)
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_backpressure_queue_rejection_and_metrics() {
    // Cola con capacidad estricta de 2 elementos
    let (queue, _receiver) = BackpressureQueue::new(2);

    let dummy_shape = vec![1, 3, 224, 224];
    let dummy_data = vec![0.0f32; 3 * 224 * 224];

    // Enviar 2 tareas que llenan la capacidad
    let q1 = queue.clone();
    let s1 = dummy_shape.clone();
    let d1 = dummy_data.clone();
    tokio::spawn(async move {
        let _ = q1.submit("task-1".into(), ModelKind::Vision, s1, d1, Duration::from_millis(50)).await;
    });

    let q2 = queue.clone();
    let s2 = dummy_shape.clone();
    let d2 = dummy_data.clone();
    tokio::spawn(async move {
        let _ = q2.submit("task-2".into(), ModelKind::Vision, s2, d2, Duration::from_millis(50)).await;
    });

    // Pequeño descanso para garantizar encolado
    tokio::time::sleep(Duration::from_millis(10)).await;

    // La 3ra tarea DEBE ser rechazada de inmediato por contrapresión (cola llena)
    let res = queue
        .submit("task-3".into(), ModelKind::Vision, dummy_shape, dummy_data, Duration::from_millis(50))
        .await;

    match res {
        Err(QueueError::Saturated { depth, capacity }) => {
            assert_eq!(depth, 2);
            assert_eq!(capacity, 2);
        }
        other => panic!("Se esperaba QueueError::Saturated, pero se obtuvo: {:?}", other),
    }

    let stats = queue.stats();
    assert!(stats.total_rejected >= 1);
}

// -----------------------------------------------------------------------------
// TEST 5: Worker de Dynamic Micro-Batching
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_dynamic_batching_worker() {
    let mut config = EdgeServiceConfig::default();
    config.pool_size = 2;
    config.max_batch_size = 4;
    config.batch_timeout_ms = 15;

    let pool = SessionPool::new(&config, VISION_MODEL_BYTES, AUDIO_MODEL_BYTES).unwrap();
    let (queue, receiver) = BackpressureQueue::new(16);
    let _worker = InferenceBatcher::spawn(
        receiver,
        pool.clone(),
        config.max_batch_size,
        Duration::from_millis(config.batch_timeout_ms),
    );

    let mut handles = Vec::new();

    // Enviar 4 tareas de visión casi simultáneamente para que se agrupen en batch
    for i in 0..4 {
        let q = queue.clone();
        let handle = tokio::spawn(async move {
            let shape = vec![1, 3, 224, 224];
            let data = vec![0.1f32 * (i as f32 + 1.0); 3 * 224 * 224];
            q.submit(format!("batch-{i}"), ModelKind::Vision, shape, data, Duration::from_secs(3))
                .await
        });
        handles.push(handle);
    }

    for h in handles {
        let res = h.await.unwrap();
        assert!(res.is_ok(), "Cada solicitud del batch debe recibir su resultado individual");
        let (logits, _us) = res.unwrap();
        assert_eq!(logits.len(), 5);
    }
}

// -----------------------------------------------------------------------------
// TEST 6: Endpoints Axum REST (/v1/vision/classify, /v1/audio/infer, /health, /metrics)
// -----------------------------------------------------------------------------
#[tokio::test]
async fn test_axum_api_endpoints() {
    let (_state, app) = create_test_system();

    // 1. Probar GET /health
    let req = Request::builder()
        .uri("/health")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
    let health_json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(health_json["status"], "healthy");

    // 2. Probar POST /v1/vision/classify
    let png_bytes = VisionPreprocessor::generate_synthetic_png(128, 128);
    let png_b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);

    let vision_payload = serde_json::json!({
        "image_base64": format!("data:image/png;base64,{png_b64}"),
        "top_k": 3
    });

    let req = Request::builder()
        .uri("/v1/vision/classify")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&vision_payload).unwrap()))
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
    let vision_json: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(vision_json["success"].as_bool().unwrap());
    assert_eq!(vision_json["predictions"].as_array().unwrap().len(), 3);
    assert!(vision_json["timing"]["inference_ms"].as_f64().unwrap() > 0.0);

    // 3. Probar POST /v1/audio/infer
    let wav_bytes = WavDecoder::synthesize_test_wav(16000, 0.5, 440.0);
    let wav_b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);

    let audio_payload = serde_json::json!({
        "audio_base64": wav_b64,
        "top_k": 2
    });

    let req = Request::builder()
        .uri("/v1/audio/infer")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&audio_payload).unwrap()))
        .unwrap();

    let res = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
    let audio_json: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(audio_json["success"].as_bool().unwrap());
    assert_eq!(audio_json["predictions"].as_array().unwrap().len(), 2);

    // 4. Probar GET /metrics
    let req = Request::builder()
        .uri("/metrics")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
    let metrics_json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(metrics_json["total_requests"], 2);
    assert_eq!(metrics_json["successful_requests"], 2);
}
