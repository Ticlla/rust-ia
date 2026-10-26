//! Modelos de Secuencia y Estado de Petición

use tokio::sync::mpsc::Sender;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Abort,
}

#[derive(Debug, Clone)]
pub struct SequenceOutputChunk {
    pub seq_id: String,
    pub token_id: u32,
    pub text_delta: String,
    pub finish_reason: Option<FinishReason>,
}

/// Estado interno de una secuencia activa en el planificador
pub struct Sequence {
    pub id: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub block_table: Vec<usize>,
    pub temperature: f64,
    pub max_tokens: usize,
    pub is_prefilled: bool,
    pub sender: Sender<SequenceOutputChunk>,
}

impl Sequence {
    pub fn new(
        id: String,
        prompt_tokens: Vec<u32>,
        temperature: f64,
        max_tokens: usize,
        sender: Sender<SequenceOutputChunk>,
    ) -> Self {
        Self {
            id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            block_table: Vec::new(),
            temperature,
            max_tokens,
            is_prefilled: false,
            sender,
        }
    }

    pub fn total_len(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }
}
