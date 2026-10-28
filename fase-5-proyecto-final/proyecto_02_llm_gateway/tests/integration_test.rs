//! Pruebas de Integración para el High-Throughput LLM Serving Gateway

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use candle_core::Device;
use proyecto_02_llm_gateway::api::handlers::{build_router, AppState};
use proyecto_02_llm_gateway::api::openai_types::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ModelListResponse};
use proyecto_02_llm_gateway::config::GatewayConfig;
use proyecto_02_llm_gateway::model::{GatewayTokenizer, GatewayTransformer};
use proyecto_02_llm_gateway::paged_cache::{BlockAllocator, PagedKvCacheManager};
use proyecto_02_llm_gateway::scheduler::{ContinuousBatchScheduler, Sequence};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tower::ServiceExt;

#[test]
fn test_paged_block_allocator() {
    let mut allocator = BlockAllocator::new(10, 4); // 10 bloques de 4 tokens = 40 tokens
    assert_eq!(allocator.num_free_blocks(), 10);
    assert_eq!(allocator.num_used_blocks(), 0);

    // Asignar bloques para 9 tokens (requiere 3 bloques de 4)
    let blocks = allocator.allocate_blocks_for_length(9).unwrap();
    assert_eq!(blocks.len(), 3);
    assert_eq!(allocator.num_used_blocks(), 3);
    assert_eq!(allocator.num_free_blocks(), 7);
    assert!((allocator.memory_utilization() - 0.3).abs() < 1e-4);

    // Liberar los bloques
    allocator.free_all(&blocks);
    assert_eq!(allocator.num_free_blocks(), 10);
    assert_eq!(allocator.num_used_blocks(), 0);
}

#[test]
fn test_paged_kv_cache_storage_and_gather() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let block_size = 4;
    let n_layers = 1;
    let n_heads = 2;
    let head_dim = 4;

    let mut manager = PagedKvCacheManager::new(8, block_size, n_layers, n_heads, head_dim, &dev);

    // Supongamos una secuencia que usa 2 bloques físicos: bloque 0 y bloque 1
    let block_table = vec![0, 1];
    let k_tok = candle_core::Tensor::ones((n_heads, head_dim), candle_core::DType::F32, &dev)?;
    let v_tok = candle_core::Tensor::ones((n_heads, head_dim), candle_core::DType::F32, &dev)?;

    // Escribir 6 tokens (4 en bloque 0, 2 en bloque 1)
    for i in 0..6 {
        let block_id = block_table[i / block_size];
        let offset = i % block_size;
        manager.write_token(0, block_id, offset, &k_tok, &v_tok)?;
    }

    // Gather de los 6 tokens
    let (gathered_k, gathered_v) = manager.gather_sequence_kv(0, &block_table, 6)?;
    assert_eq!(gathered_k.dims4()?, (1, n_heads, 6, head_dim));
    assert_eq!(gathered_v.dims4()?, (1, n_heads, 6, head_dim));

    // Limpieza
    manager.free_sequence(&block_table);
    assert_eq!(manager.allocator.num_used_blocks(), 0);
    Ok(())
}

#[tokio::test]
async fn test_continuous_batching_scheduler_admission_and_step() {
    let dev = Device::Cpu;
    let config = GatewayConfig {
        embed_dim: 16,
        n_heads: 2,
        n_layers: 1,
        vocab_size: 30,
        block_size: 4,
        total_blocks: 16,
        max_batch_size: 4,
        eos_token_id: 2,
        ..Default::default()
    };

    let model = Arc::new(GatewayTransformer::new(config.clone(), &dev).unwrap());
    let tokenizer = GatewayTokenizer::new_builtin();
    let paged_cache = Arc::new(Mutex::new(PagedKvCacheManager::new(
        config.total_blocks,
        config.block_size,
        config.n_layers,
        config.n_heads,
        config.embed_dim / config.n_heads,
        &dev,
    )));

    let scheduler = ContinuousBatchScheduler::new(
        model,
        tokenizer,
        paged_cache.clone(),
        config.max_batch_size,
        config.eos_token_id,
        config.block_size,
    );

    let (tx1, mut rx1) = mpsc::channel(16);
    let (tx2, mut rx2) = mpsc::channel(16);

    let seq1 = Sequence::new("seq-1".to_string(), vec![4, 5], 0.0, 2, tx1);
    let seq2 = Sequence::new("seq-2".to_string(), vec![7, 8], 0.0, 2, tx2);

    scheduler.add_request(seq1).await;
    scheduler.add_request(seq2).await;

    // Ejecutar iteraciones de batch continuo
    let mut step_count = 0;
    while step_count < 10 {
        let active = scheduler.step().await.unwrap();
        step_count += 1;
        if active == 0 {
            break;
        }
    }

    // Verificar que ambas secuencias recibieron chunks generados
    let mut chunks1 = Vec::new();
    while let Ok(chunk) = rx1.try_recv() {
        chunks1.push(chunk);
    }
    assert!(!chunks1.is_empty(), "Secuencia 1 debe haber generado tokens");

    let mut chunks2 = Vec::new();
    while let Ok(chunk) = rx2.try_recv() {
        chunks2.push(chunk);
    }
    assert!(!chunks2.is_empty(), "Secuencia 2 debe haber generado tokens");

    // Verificar que tras completar la inferencia, la memoria se liberó
    let cache = paged_cache.lock().await;
    assert_eq!(cache.allocator.num_used_blocks(), 0, "Todos los bloques deben haberse liberado");
}

#[tokio::test]
async fn test_openai_api_endpoints() {
    let dev = Device::Cpu;
    let config = GatewayConfig {
        model_id: "test-model".to_string(),
        embed_dim: 16,
        n_heads: 2,
        n_layers: 1,
        vocab_size: 30,
        block_size: 4,
        total_blocks: 16,
        max_batch_size: 4,
        eos_token_id: 2,
        ..Default::default()
    };

    let model = Arc::new(GatewayTransformer::new(config.clone(), &dev).unwrap());
    let tokenizer = GatewayTokenizer::new_builtin();
    let paged_cache = Arc::new(Mutex::new(PagedKvCacheManager::new(
        config.total_blocks,
        config.block_size,
        config.n_layers,
        config.n_heads,
        config.embed_dim / config.n_heads,
        &dev,
    )));

    let scheduler = Arc::new(ContinuousBatchScheduler::new(
        model,
        tokenizer,
        paged_cache,
        config.max_batch_size,
        config.eos_token_id,
        config.block_size,
    ));

    // Spawn scheduler background loop
    let scheduler_loop = scheduler.clone();
    let cancel_token = Arc::new(tokio::sync::Notify::new());
    let cancel_notify = cancel_token.clone();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancel_notify.notified() => break,
                res = scheduler_loop.step() => {
                    if let Ok(0) = res {
                        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                    }
                }
            }
        }
    });

    let state = Arc::new(AppState::new(config.clone(), scheduler));
    let app = build_router(state);

    // 1. Test GET /v1/models
    let res = app
        .clone()
        .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), 1024 * 1024).await.unwrap();
    let models: ModelListResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(models.data[0].id, "test-model");

    // 2. Test POST /v1/chat/completions (Non-streaming)
    let req_body = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Hola".to_string(),
        }],
        temperature: 0.0,
        max_tokens: 2,
        stream: false,
    };
    let json_bytes = serde_json::to_vec(&req_body).unwrap();

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(json_bytes))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), 1024 * 1024).await.unwrap();
    let completion: ChatCompletionResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(completion.object, "chat.completion");
    assert!(!completion.choices.is_empty());
    assert!(!completion.choices[0].message.content.is_empty());

    // 4. Test GET /metrics (Prometheus)
    let res = app
        .clone()
        .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), 1024 * 1024).await.unwrap();
    let metrics_text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(metrics_text.contains("llm_gateway_requests_total 1"));
    assert!(metrics_text.contains("llm_gateway_kv_cache_utilization_ratio"));
    assert!(metrics_text.contains("llm_gateway_kv_cache_free_blocks"));

    cancel_token.notify_one();
}
