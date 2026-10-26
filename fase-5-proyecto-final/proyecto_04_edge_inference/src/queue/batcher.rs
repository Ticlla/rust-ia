//! Worker de ejecución de inferencia con soporte para micro-batching dinámico.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

use crate::engine::SessionPool;
use crate::queue::backpressure::InferenceTask;

pub struct InferenceBatcher;

impl InferenceBatcher {
    /// Inicia el bucle de procesamiento en segundo plano
    pub fn spawn(
        mut receiver: mpsc::Receiver<InferenceTask>,
        pool: SessionPool,
        max_batch_size: usize,
        batch_timeout: Duration,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let pool = Arc::new(pool);

            while let Some(first_task) = receiver.recv().await {
                let mut batch = vec![first_task];

                // Si se habilita batching mayor a 1, intentar recolectar más tareas dentro del timeout
                if max_batch_size > 1 {
                    let deadline = tokio::time::Instant::now() + batch_timeout;
                    while batch.len() < max_batch_size {
                        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                        if remaining.is_zero() {
                            break;
                        }

                        match tokio::time::timeout(remaining, receiver.recv()).await {
                            Ok(Some(next_task)) => {
                                // Agrupar solo si coincide el tipo de modelo y forma unitaria
                                if next_task.kind == batch[0].kind && next_task.shape == batch[0].shape {
                                    batch.push(next_task);
                                } else {
                                    // Tarea con diferente forma/modelo: ejecutar de inmediato en tarea paralela
                                    let pool_clone = pool.clone();
                                    tokio::spawn(async move {
                                        let res = pool_clone
                                            .run(next_task.kind, &next_task.shape, next_task.data)
                                            .await;
                                        let _ = next_task.response_tx.send(res);
                                    });
                                }
                            }
                            _ => break,
                        }
                    }
                }

                // Ejecución: si batch.len() == 1, ejecución directa sin overhead de concatenación
                if batch.len() == 1 {
                    let task = batch.pop().unwrap();
                    let pool_clone = pool.clone();
                    tokio::spawn(async move {
                        let res = pool_clone.run(task.kind, &task.shape, task.data).await;
                        let _ = task.response_tx.send(res);
                    });
                } else {
                    // Micro-batching dinámico
                    let batch_size = batch.len();
                    let kind = batch[0].kind;
                    let mut batched_shape = batch[0].shape.clone();
                    batched_shape[0] = batch_size;

                    let sample_len: usize = batch[0].data.len();
                    let mut batched_data = Vec::with_capacity(batch_size * sample_len);
                    let mut senders = Vec::with_capacity(batch_size);

                    for task in batch {
                        batched_data.extend(task.data);
                        senders.push(task.response_tx);
                    }

                    let pool_clone = pool.clone();
                    tokio::spawn(async move {
                        match pool_clone.run(kind, &batched_shape, batched_data).await {
                            Ok((all_outputs, elapsed_us)) => {
                                let output_per_sample = all_outputs.len() / batch_size;
                                for (i, sender) in senders.into_iter().enumerate() {
                                    let start_idx = i * output_per_sample;
                                    let end_idx = start_idx + output_per_sample;
                                    let sample_slice = all_outputs[start_idx..end_idx].to_vec();
                                    let _ = sender.send(Ok((sample_slice, elapsed_us)));
                                }
                            }
                            Err(e) => {
                                for sender in senders {
                                    let _ = sender.send(Err(crate::engine::EngineError::InferenceFailed(
                                        format!("Fallo en micro-batching: {e}"),
                                    )));
                                }
                            }
                        }
                    });
                }
            }
        })
    }
}
