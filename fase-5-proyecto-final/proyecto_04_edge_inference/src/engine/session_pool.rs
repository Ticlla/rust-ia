//! Pool de sesiones ONNX concurrentes sin contención de locks ni recompilación de grafos.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, Semaphore};

use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;

use crate::config::EdgeServiceConfig;
use crate::engine::types::{EngineError, ModelKind};

/// Entrada de un worker con su sesión dedicada
pub struct SessionEntry {
    pub session: Session,
    pub input_name: String,
    pub output_name: String,
}

/// Pool concurrente para un tipo específico de modelo
pub struct SingleModelPool {
    sessions: Arc<Mutex<Vec<SessionEntry>>>,
    semaphore: Arc<Semaphore>,
    capacity: usize,
    active_count: Arc<AtomicUsize>,
    total_inferences: Arc<AtomicUsize>,
}

impl SingleModelPool {
    pub fn new(
        env: &Arc<Environment>,
        model_bytes: &[u8],
        capacity: usize,
        intra_threads: usize,
        inter_threads: usize,
    ) -> Result<Self, EngineError> {
        let mut entries = Vec::with_capacity(capacity);

        for i in 0..capacity {
            let session = Session::builder(env)
                .map_err(|e| EngineError::SessionInit(format!("Fallo Session::builder: {e}")))?
                .with_intra_threads(intra_threads)
                .map_err(|e| EngineError::SessionInit(format!("Fallo with_intra_threads: {e}")))?
                .with_inter_threads(inter_threads)
                .map_err(|e| EngineError::SessionInit(format!("Fallo with_inter_threads: {e}")))?
                .commit_from_memory(model_bytes)
                .map_err(|e| {
                    EngineError::SessionInit(format!("Fallo commit_from_memory en sesión #{i}: {e}"))
                })?;

            let input_name = session
                .inputs()
                .first()
                .map(|inp| inp.name().to_string())
                .unwrap_or_else(|| "input".to_string());

            let output_name = session
                .outputs()
                .first()
                .map(|out| out.name().to_string())
                .unwrap_or_else(|| "output".to_string());

            entries.push(SessionEntry {
                session,
                input_name,
                output_name,
            });
        }

        Ok(Self {
            sessions: Arc::new(Mutex::new(entries)),
            semaphore: Arc::new(Semaphore::new(capacity)),
            capacity,
            active_count: Arc::new(AtomicUsize::new(0)),
            total_inferences: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Ejecuta una inferencia adquiriendo temporalmente una sesión libre del pool
    pub async fn run_inference(
        &self,
        shape: &[usize],
        data: Vec<f32>,
    ) -> Result<(Vec<f32>, u128), EngineError> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(EngineError::ShapeMismatch {
                expected: expected_len,
                got: data.len(),
            });
        }

        // 1. Adquirir permiso del semáforo (espera no bloqueante asíncrona)
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| EngineError::PoolExhausted)?;

        // 2. Extraer sesión del pool
        let mut entry = {
            let mut guard = self.sessions.lock().await;
            guard.pop().ok_or(EngineError::PoolExhausted)?
        };

        self.active_count.fetch_add(1, Ordering::Relaxed);
        let t0 = Instant::now();

        // 3. Crear tensor de entrada y ejecutar
        let tensor_result = Tensor::from_array((shape, data))
            .map_err(|e| EngineError::InferenceFailed(format!("Error creando Tensor: {e}")));

        let inference_result = match tensor_result {
            Ok(tensor) => {
                let inputs = ort::inputs![entry.input_name.as_str() => tensor];
                match entry.session.run(inputs) {
                    Ok(outputs) => match outputs.get(entry.output_name.as_str()) {
                        Some(out_val) => match out_val.try_extract_tensor::<f32>() {
                            Ok((_, slice)) => Ok(slice.to_vec()),
                            Err(e) => Err(EngineError::InferenceFailed(format!(
                                "Fallo extrayendo tensor de salida: {e}"
                            ))),
                        },
                        None => Err(EngineError::InferenceFailed(format!(
                            "Salida '{}' no encontrada en grafo",
                            entry.output_name
                        ))),
                    },
                    Err(e) => Err(EngineError::InferenceFailed(format!(
                        "Fallo en session.run: {e}"
                    ))),
                }
            }
            Err(e) => Err(e),
        };

        let elapsed_us = t0.elapsed().as_micros();
        self.active_count.fetch_sub(1, Ordering::Relaxed);
        self.total_inferences.fetch_add(1, Ordering::Relaxed);

        // 4. Devolver la sesión al pool (RAII manual seguro)
        {
            let mut guard = self.sessions.lock().await;
            guard.push(entry);
        }

        inference_result.map(|res| (res, elapsed_us))
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        let active = self.active_count.load(Ordering::Relaxed);
        let total = self.total_inferences.load(Ordering::Relaxed);
        (active, self.capacity, total)
    }
}

/// Orquestador central de pools para múltiples modalidades (Visión y Audio)
#[derive(Clone)]
pub struct SessionPool {
    #[allow(dead_code)]
    env: Arc<Environment>,
    vision_pool: Arc<SingleModelPool>,
    audio_pool: Arc<SingleModelPool>,
}

impl SessionPool {
    pub fn new(
        config: &EdgeServiceConfig,
        vision_bytes: &[u8],
        audio_bytes: &[u8],
    ) -> Result<Self, EngineError> {
        let env = Arc::new(
            ort::init()
                .with_name("edge_inference_engine")
                .build()
                .map_err(|e| EngineError::SessionInit(format!("Error inicializando ort: {e}")))?,
        );

        let vision_pool = Arc::new(SingleModelPool::new(
            &env,
            vision_bytes,
            config.pool_size,
            config.intra_threads,
            config.inter_threads,
        )?);

        let audio_pool = Arc::new(SingleModelPool::new(
            &env,
            audio_bytes,
            config.pool_size,
            config.intra_threads,
            config.inter_threads,
        )?);

        Ok(Self {
            env,
            vision_pool,
            audio_pool,
        })
    }

    pub async fn run(
        &self,
        kind: ModelKind,
        shape: &[usize],
        data: Vec<f32>,
    ) -> Result<(Vec<f32>, u128), EngineError> {
        match kind {
            ModelKind::Vision => self.vision_pool.run_inference(shape, data).await,
            ModelKind::Audio => self.audio_pool.run_inference(shape, data).await,
        }
    }

    pub fn get_stats(&self) -> PoolMetrics {
        let (v_act, v_cap, v_tot) = self.vision_pool.stats();
        let (a_act, a_cap, a_tot) = self.audio_pool.stats();

        PoolMetrics {
            vision_active_sessions: v_act,
            vision_capacity: v_cap,
            vision_total_inferences: v_tot,
            audio_active_sessions: a_act,
            audio_capacity: a_cap,
            audio_total_inferences: a_tot,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PoolMetrics {
    pub vision_active_sessions: usize,
    pub vision_capacity: usize,
    pub vision_total_inferences: usize,
    pub audio_active_sessions: usize,
    pub audio_capacity: usize,
    pub audio_total_inferences: usize,
}
