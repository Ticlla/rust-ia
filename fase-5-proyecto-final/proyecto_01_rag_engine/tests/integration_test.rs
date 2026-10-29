//! Pruebas de Integración de Extremo a Extremo para Enterprise Hybrid RAG Engine

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use candle_core::Device;
use proyecto_01_rag_engine::api::handlers::{AppState, build_router};
use proyecto_01_rag_engine::api::models::{HealthResponse, SearchRequest, SearchResponse};
use proyecto_01_rag_engine::config::RagConfig;
use proyecto_01_rag_engine::engine::{EnterpriseRagEngine, SearchMode};
use proyecto_01_rag_engine::ingestion::{Document, TextChunker};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;

#[test]
fn test_chunker_sliding_window() {
    let chunker = TextChunker::new(50, 10);
    let doc = Document::new(
        "doc1",
        "Test Chunker",
        "Rust es un lenguaje de programación seguro y concurrente. Su sistema de tipos evita data races.",
    );

    let chunks = chunker.chunk_document(&doc);
    assert!(chunks.len() >= 2, "Debe generar al menos 2 chunks");
    assert_eq!(chunks[0].chunk_index, 0);
    assert_eq!(chunks[1].chunk_index, 1);
    assert!(chunks[0].text.len() <= 50);
}

#[test]
fn test_bm25_lexical_search() {
    let mut engine = EnterpriseRagEngine::new(RagConfig::default(), &Device::Cpu).unwrap();
    let docs = vec![
        Document::new(
            "doc_cuda",
            "Nvidia CUDA",
            "CUDA es una plataforma de computación paralela desarrollada por Nvidia para GPUs.",
        ),
        Document::new(
            "doc_rust",
            "Rust Lang",
            "Rust provee garantías de seguridad de memoria sin recolector de basura.",
        ),
    ];
    engine.ingest_documents(docs).unwrap();

    let hits = engine.search("Nvidia GPUs", 5, SearchMode::Sparse).unwrap();
    assert!(!hits.is_empty(), "Debe encontrar resultados léxicos");
    assert_eq!(hits[0].doc_id, "doc_cuda");
    assert!(hits[0].bm25_score.unwrap() > 0.0);
}

#[test]
fn test_dense_semantic_search() {
    let mut engine = EnterpriseRagEngine::new(RagConfig::default(), &Device::Cpu).unwrap();
    let docs = vec![
        Document::new(
            "doc_concurrency",
            "Concurrencia",
            "Sistemas multi-hilo y canales asíncronos para distribuir trabajo en paralelo.",
        ),
        Document::new(
            "doc_cooking",
            "Cocina Italiana",
            "Recetas tradicionales de pasta y pizza casera con masa madre.",
        ),
    ];
    engine.ingest_documents(docs).unwrap();

    let hits = engine
        .search("sistemas en paralelo", 5, SearchMode::Dense)
        .unwrap();
    assert!(!hits.is_empty(), "Debe encontrar resultados semánticos");
    assert_eq!(hits[0].doc_id, "doc_concurrency");
}

#[test]
fn test_hybrid_rrf_and_reranking() {
    let mut engine = EnterpriseRagEngine::new(RagConfig::default(), &Device::Cpu).unwrap();
    let docs = vec![
        Document::new(
            "doc_rag",
            "RAG Enterprise",
            "Los sistemas de Generación Aumentada por Recuperación combinan búsqueda con LLMs.",
        ),
        Document::new(
            "doc_llm",
            "Modelos LLM",
            "Los modelos de lenguaje grandes procesan texto de forma autorregresiva.",
        ),
    ];
    engine.ingest_documents(docs).unwrap();

    // Búsqueda Híbrida
    let hits_hybrid = engine
        .search("sistemas RAG y búsqueda", 5, SearchMode::Hybrid)
        .unwrap();
    assert!(!hits_hybrid.is_empty());
    assert!(hits_hybrid[0].rrf_score.is_some());

    // Búsqueda Híbrida con Reranker Neuronal
    let hits_rerank = engine
        .search("sistemas RAG y búsqueda", 5, SearchMode::HybridWithRerank)
        .unwrap();
    assert!(!hits_rerank.is_empty());
    assert!(hits_rerank[0].final_score > 0.0);
}

#[tokio::test]
async fn test_axum_api_endpoints() {
    let config = RagConfig::default();
    let dev = Device::Cpu;
    let mut engine = EnterpriseRagEngine::new(config, &dev).unwrap();

    let docs = vec![Document::new(
        "doc_api_test",
        "API Document",
        "Documento para verificar endpoints HTTP de Axum y serialización JSON.",
    )];
    engine.ingest_documents(docs).unwrap();

    let state = Arc::new(AppState::new(Arc::new(RwLock::new(engine))));
    let app = build_router(state);

    // 1. Test GET /health
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let health: HealthResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(health.status, "healthy");
    assert_eq!(health.stats.total_documents, 1);

    // 2. Test POST /api/v1/search
    let search_req = SearchRequest {
        query: "endpoints HTTP Axum".to_string(),
        top_k: 2,
        mode: SearchMode::Hybrid,
    };
    let req_body = serde_json::to_vec(&search_req).unwrap();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/search")
                .header("content-type", "application/json")
                .body(Body::from(req_body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let search_res: SearchResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(search_res.total_hits, 1);
    assert_eq!(search_res.hits[0].doc_id, "doc_api_test");

    // 4. Test GET /metrics (Prometheus)
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let metrics_text = String::from_utf8(body_bytes.to_vec()).unwrap();
    assert!(metrics_text.contains("rag_engine_uptime_seconds"));
    assert!(metrics_text.contains("rag_engine_documents_total"));
    assert!(metrics_text.contains("rag_engine_searches_total 1"));
}
