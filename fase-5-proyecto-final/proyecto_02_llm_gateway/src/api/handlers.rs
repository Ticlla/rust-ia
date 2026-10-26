//! Manejadores de Rutas de la API OpenAI-Compatible

use super::openai_types::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ModelCard,
    ModelListResponse, UsageInfo,
};
use super::sse::create_sse_response;
use crate::config::GatewayConfig;
use crate::scheduler::{ContinuousBatchScheduler, FinishReason, Sequence};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use uuid::Uuid;

pub struct AppState {
    pub config: GatewayConfig,
    pub scheduler: Arc<ContinuousBatchScheduler>,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state)
}

async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cache = state.scheduler.paged_cache.lock().await;
    let waiting = state.scheduler.waiting_queue.lock().await;
    let running = state.scheduler.running_sequences.lock().await;

    Json(serde_json::json!({
        "status": "healthy",
        "model": state.config.model_id,
        "waiting_requests": waiting.len(),
        "running_requests": running.len(),
        "memory_utilization": cache.allocator.memory_utilization(),
        "free_blocks": cache.allocator.num_free_blocks(),
        "used_blocks": cache.allocator.num_used_blocks(),
    }))
}

async fn models_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelCard {
            id: state.config.model_id.clone(),
            object: "model".to_string(),
            created: now,
            owned_by: "rust-ia-systems".to_string(),
        }],
    })
}

async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    // 1. Construir prompt a partir del historial de mensajes
    let mut prompt = String::new();
    for msg in &payload.messages {
        prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }
    prompt.push_str("assistant: ");

    let prompt_tokens = state
        .scheduler
        .tokenizer
        .encode(&prompt)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Error de tokenización: {}", e)))?;

    if prompt_tokens.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Prompt no puede estar vacío".to_string()));
    }

    let seq_id = format!("chatcmpl-{}", Uuid::new_v4());
    let (tx, mut rx) = mpsc::channel(64);

    let sequence = Sequence::new(
        seq_id.clone(),
        prompt_tokens.clone(),
        payload.temperature,
        payload.max_tokens,
        tx,
    );

    // Enviar secuencia al scheduler de continuous batching
    state.scheduler.add_request(sequence).await;

    // 2. Si el cliente solicitó Streaming vía SSE:
    if payload.stream {
        let sse = create_sse_response(seq_id, state.config.model_id.clone(), rx);
        return Ok(sse.into_response());
    }

    // 3. Si solicitó respuesta síncrona completa:
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut full_content = String::new();
    let mut finish_reason_str = "stop".to_string();
    let mut completion_tokens = 0;

    while let Some(chunk) = rx.recv().await {
        full_content.push_str(&chunk.text_delta);
        completion_tokens += 1;
        if let Some(reason) = chunk.finish_reason {
            finish_reason_str = match reason {
                FinishReason::Stop => "stop".to_string(),
                FinishReason::Length => "length".to_string(),
                FinishReason::Abort => "abort".to_string(),
            };
            break;
        }
    }

    let response = ChatCompletionResponse {
        id: seq_id,
        object: "chat.completion".to_string(),
        created: now,
        model: state.config.model_id.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: full_content.trim().to_string(),
            },
            finish_reason: Some(finish_reason_str),
        }],
        usage: UsageInfo {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens: prompt_tokens.len() + completion_tokens,
        },
    };

    Ok(Json(response).into_response())
}
