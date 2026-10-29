//! Planificador de Inferencia con Continuous Batching en Tiempo Real

use super::request::{FinishReason, Sequence, SequenceOutputChunk};
use crate::model::{GatewayTokenizer, GatewayTransformer};
use crate::paged_cache::PagedKvCacheManager;
use candle_core::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct ContinuousBatchScheduler {
    pub model: Arc<GatewayTransformer>,
    pub tokenizer: GatewayTokenizer,
    pub paged_cache: Arc<Mutex<PagedKvCacheManager>>,
    pub waiting_queue: Arc<Mutex<VecDeque<Sequence>>>,
    pub running_sequences: Arc<Mutex<Vec<Sequence>>>,
    pub max_batch_size: usize,
    pub eos_token_id: u32,
    pub block_size: usize,
}

impl ContinuousBatchScheduler {
    pub fn new(
        model: Arc<GatewayTransformer>,
        tokenizer: GatewayTokenizer,
        paged_cache: Arc<Mutex<PagedKvCacheManager>>,
        max_batch_size: usize,
        eos_token_id: u32,
        block_size: usize,
    ) -> Self {
        Self {
            model,
            tokenizer,
            paged_cache,
            waiting_queue: Arc::new(Mutex::new(VecDeque::new())),
            running_sequences: Arc::new(Mutex::new(Vec::new())),
            max_batch_size,
            eos_token_id,
            block_size,
        }
    }

    /// Encola una nueva petición para ser procesada por el planificador dinámico
    pub async fn add_request(&self, sequence: Sequence) {
        let mut queue = self.waiting_queue.lock().await;
        queue.push_back(sequence);
    }

    /// Ejecuta una única iteración de Continuous Batching:
    /// 1. Admite nuevas secuencias si hay bloques libres y espacio en el batch.
    /// 2. Ejecuta 1 paso de prefill o decode para todas las secuencias activas concurrentes.
    /// 3. Emite los tokens generados a los canales de streaming correspondientes.
    /// 4. Retira y libera la memoria de las secuencias completadas.
    pub async fn step(&self) -> Result<usize> {
        let mut cache = self.paged_cache.lock().await;
        let mut waiting = self.waiting_queue.lock().await;
        let mut running = self.running_sequences.lock().await;

        // 1. Fase de Admisión: Mover secuencias de waiting a running si hay memoria
        while running.len() < self.max_batch_size && !waiting.is_empty() {
            let next_prompt_len = waiting.front().unwrap().prompt_tokens.len();
            // Estimamos bloques requeridos para prefill + al menos 1 bloque de decode
            let blocks_needed = (next_prompt_len + self.block_size - 1) / self.block_size + 1;

            if cache.allocator.num_free_blocks() >= blocks_needed {
                let mut seq = waiting.pop_front().unwrap();
                // Asignar bloques físicos iniciales
                if let Some(blocks) = cache.allocator.allocate_blocks_for_length(next_prompt_len) {
                    seq.block_table = blocks;
                    running.push(seq);
                } else {
                    waiting.push_front(seq);
                    break;
                }
            } else {
                break; // No hay suficiente VRAM/RAM en este ciclo
            }
        }

        if running.is_empty() {
            return Ok(0);
        }

        let mut completed_indices = Vec::new();

        // 2. Ejecución de la iteración de inferencia para cada secuencia activa
        for (idx, seq) in running.iter_mut().enumerate() {
            let next_token_result = if !seq.is_prefilled {
                // Modo PREFILL: Procesar prompt completo
                let logits = self.model.step_sequence(
                    &seq.prompt_tokens,
                    true,
                    &seq.block_table,
                    &mut cache,
                )?;
                seq.is_prefilled = true;
                GatewayTransformer::sample_token(&logits, seq.temperature)?
            } else {
                // Modo DECODE: Necesitamos verificar si rebasamos el tamaño de bloque actual
                let curr_len = seq.total_len();
                if curr_len % self.block_size == 0 {
                    // Requerimos un nuevo bloque físico
                    if let Some(new_block) = cache.allocator.allocate() {
                        seq.block_table.push(new_block);
                    } else {
                        // OOM Preventivo: Si no hay bloques, finalizar por aborto
                        let _ = seq
                            .sender
                            .send(SequenceOutputChunk {
                                seq_id: seq.id.clone(),
                                token_id: self.eos_token_id,
                                text_delta: String::new(),
                                finish_reason: Some(FinishReason::Abort),
                            })
                            .await;
                        completed_indices.push(idx);
                        continue;
                    }
                }

                // Pasar los tokens acumulados completos para el gather de attention
                let mut full_history = seq.prompt_tokens.clone();
                full_history.extend_from_slice(&seq.generated_tokens);

                let logits =
                    self.model
                        .step_sequence(&full_history, false, &seq.block_table, &mut cache)?;
                GatewayTransformer::sample_token(&logits, seq.temperature)?
            };

            seq.generated_tokens.push(next_token_result);
            let token_text = self
                .tokenizer
                .decode(&[next_token_result])
                .unwrap_or_else(|_| format!(" <tok_{}>", next_token_result));

            // Verificar si la secuencia terminó
            let finish_reason = if next_token_result == self.eos_token_id {
                Some(FinishReason::Stop)
            } else if seq.generated_tokens.len() >= seq.max_tokens {
                Some(FinishReason::Length)
            } else {
                None
            };

            let is_finished = finish_reason.is_some();

            // Emitir chunk al canal asíncrono SSE del cliente HTTP
            let _ = seq
                .sender
                .send(SequenceOutputChunk {
                    seq_id: seq.id.clone(),
                    token_id: next_token_result,
                    text_delta: format!(" {}", token_text),
                    finish_reason,
                })
                .await;

            if is_finished {
                completed_indices.push(idx);
            }
        }

        // 3. Limpieza: Liberar memoria Paged KV-Cache para las secuencias que terminaron
        // Ordenamos en reversa para poder hacer swap_remove/remove sin desfasar índices
        completed_indices.sort_by(|a, b| b.cmp(a));
        for idx in completed_indices {
            let finished_seq = running.remove(idx);
            cache.free_sequence(&finished_seq.block_table);
        }

        Ok(running.len())
    }
}
