//! Parser de eventos Server-Sent Events (SSE) para streaming continuo de tokens.

use crate::types::ChatCompletionChunk;

#[derive(Debug, Clone, PartialEq)]
pub enum SseEvent {
    Token(String),
    Reasoning(String),
    Done,
    Comment(String),
    Empty,
}

pub struct SseParser;

impl SseParser {
    /// Parsea una línea de texto SSE cruda proveniente del servidor
    pub fn parse_line(line: &str) -> Result<SseEvent, String> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(SseEvent::Empty);
        }

        if trimmed.starts_with(':') {
            return Ok(SseEvent::Comment(trimmed[1..].trim().to_string()));
        }

        if let Some(data_str) = trimmed.strip_prefix("data:") {
            let data = data_str.trim();
            if data == "[DONE]" {
                return Ok(SseEvent::Done);
            }

            match serde_json::from_str::<ChatCompletionChunk>(data) {
                Ok(chunk) => {
                    if let Some(choice) = chunk.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            return Ok(SseEvent::Token(content.clone()));
                        } else if let Some(reasoning) = &choice.delta.reasoning_content {
                            return Ok(SseEvent::Reasoning(reasoning.clone()));
                        }
                    }
                    Ok(SseEvent::Empty)
                }
                Err(e) => Err(format!(
                    "Error deserializando chunk SSE: {e} | Contenido: {data}"
                )),
            }
        } else {
            Ok(SseEvent::Empty)
        }
    }
}
