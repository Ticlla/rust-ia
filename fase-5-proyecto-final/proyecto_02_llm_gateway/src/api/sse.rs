//! Generador de Flujos SSE (Server-Sent Events) para Token Streaming

use super::openai_types::{ChatCompletionChunk, ChunkChoice, DeltaMessage};
use crate::scheduler::{FinishReason, SequenceOutputChunk};
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::stream::Stream;
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::Receiver;

pub fn create_sse_response(
    seq_id: String,
    model_id: String,
    mut rx: Receiver<SequenceOutputChunk>,
) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let stream = async_stream::stream! {
        // 1. Emitir el chunk inicial con role: "assistant"
        let first_chunk = ChatCompletionChunk {
            id: seq_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model_id.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: DeltaMessage {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        yield Ok(Event::default().data(serde_json::to_string(&first_chunk).unwrap()));

        // 2. Transmitir los deltas de texto a medida que el planificador continuo los genera
        while let Some(chunk) = rx.recv().await {
            let finish_reason_str = chunk.finish_reason.map(|r| match r {
                FinishReason::Stop => "stop".to_string(),
                FinishReason::Length => "length".to_string(),
                FinishReason::Abort => "abort".to_string(),
            });

            let event_chunk = ChatCompletionChunk {
                id: seq_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: now,
                model: model_id.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: None,
                        content: Some(chunk.text_delta),
                    },
                    finish_reason: finish_reason_str,
                }],
            };

            yield Ok(Event::default().data(serde_json::to_string(&event_chunk).unwrap()));
        }

        // 3. Emitir el marcador final canónico de OpenAI: [DONE]
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}
