//! Cola de inferencia con control de contrapresión (Backpressure) y protección contra sobrecarga.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

use crate::engine::{EngineError, ModelKind};

#[derive(Debug)]
pub enum QueueError {
    Saturated { depth: usize, capacity: usize },
    Timeout(Duration),
    WorkerDropped,
    Engine(EngineError),
}

impl fmt::Display for QueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Saturated { depth, capacity } => {
                write!(f, "Cola saturada (Backpressure activo): {depth}/{capacity} solicitudes en espera")
            }
            Self::Timeout(dur) => write!(f, "Tiempo de espera en cola excedido ({dur:?})"),
            Self::WorkerDropped => write!(f, "El worker de inferencia finalizó inesperadamente"),
            Self::Engine(e) => write!(f, "Error en el motor de inferencia: {e}"),
        }
    }
}

impl std::error::Error for QueueError {}

pub struct InferenceTask {
    pub id: String,
    pub kind: ModelKind,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub response_tx: oneshot::Sender<Result<(Vec<f32>, u128), EngineError>>,
}

#[derive(Clone)]
pub struct BackpressureQueue {
    sender: mpsc::Sender<InferenceTask>,
    capacity: usize,
    submitted_count: Arc<AtomicUsize>,
    rejected_count: Arc<AtomicUsize>,
    completed_count: Arc<AtomicUsize>,
}

impl BackpressureQueue {
    pub fn new(capacity: usize) -> (Self, mpsc::Receiver<InferenceTask>) {
        let (sender, receiver) = mpsc::channel(capacity);
        let queue = Self {
            sender,
            capacity,
            submitted_count: Arc::new(AtomicUsize::new(0)),
            rejected_count: Arc::new(AtomicUsize::new(0)),
            completed_count: Arc::new(AtomicUsize::new(0)),
        };
        (queue, receiver)
    }

    /// Envía una tarea a la cola aplicando contrapresión inmediata si la capacidad está colmada.
    pub async fn submit(
        &self,
        id: String,
        kind: ModelKind,
        shape: Vec<usize>,
        data: Vec<f32>,
        timeout_dur: Duration,
    ) -> Result<(Vec<f32>, u128), QueueError> {
        let (response_tx, response_rx) = oneshot::channel();
        let task = InferenceTask {
            id,
            kind,
            shape,
            data,
            response_tx,
        };

        // 1. Verificación no bloqueante de capacidad (Backpressure)
        let _current_depth = self.capacity.saturating_sub(self.sender.capacity());
        match self.sender.try_send(task) {
            Ok(_) => {
                self.submitted_count.fetch_add(1, Ordering::Relaxed);
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.rejected_count.fetch_add(1, Ordering::Relaxed);
                return Err(QueueError::Saturated {
                    depth: self.capacity,
                    capacity: self.capacity,
                });
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                return Err(QueueError::WorkerDropped);
            }
        }

        // 2. Esperar respuesta del worker con timeout estricto
        match tokio::time::timeout(timeout_dur, response_rx).await {
            Ok(Ok(Ok(result))) => {
                self.completed_count.fetch_add(1, Ordering::Relaxed);
                Ok(result)
            }
            Ok(Ok(Err(engine_err))) => Err(QueueError::Engine(engine_err)),
            Ok(Err(_oneshot_closed)) => Err(QueueError::WorkerDropped),
            Err(_elapsed) => Err(QueueError::Timeout(timeout_dur)),
        }
    }

    pub fn stats(&self) -> QueueMetrics {
        let submitted = self.submitted_count.load(Ordering::Relaxed);
        let rejected = self.rejected_count.load(Ordering::Relaxed);
        let completed = self.completed_count.load(Ordering::Relaxed);
        let current_depth = self.capacity.saturating_sub(self.sender.capacity());

        QueueMetrics {
            capacity: self.capacity,
            current_depth,
            total_submitted: submitted,
            total_rejected: rejected,
            total_completed: completed,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct QueueMetrics {
    pub capacity: usize,
    pub current_depth: usize,
    pub total_submitted: usize,
    pub total_rejected: usize,
    pub total_completed: usize,
}
