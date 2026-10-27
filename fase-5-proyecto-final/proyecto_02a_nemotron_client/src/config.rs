//! Configuración y perfiles de conexión para el cliente Nemotron.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendMode {
    /// Conexión oficial a NVIDIA NIM API Cloud
    NvidiaNim,
    /// Conexión a motor local compatible con OpenAI (Ollama, vLLM, llama.cpp)
    LocalEngine,
    /// Simulador offline con razonamiento simulado para pruebas locales inmediatas
    MockSimulator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NemotronConfig {
    pub backend_mode: BackendMode,
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: Option<usize>,
    pub stream: bool,
    pub system_prompt: String,
}

impl Default for NemotronConfig {
    fn default() -> Self {
        let env_key = std::env::var("NVIDIA_API_KEY").ok();
        let backend_mode = if env_key.is_some() {
            BackendMode::NvidiaNim
        } else {
            BackendMode::MockSimulator
        };

        Self {
            backend_mode,
            api_key: env_key,
            base_url: "https://integrate.api.nvidia.com/v1".to_string(),
            model: "nvidia/llama-3.1-nemotron-70b-instruct".to_string(),
            temperature: 0.6,
            top_p: 0.7,
            max_tokens: Some(2048),
            stream: true,
            system_prompt: "Eres Nemotron, un asistente de IA avanzado desarrollado por NVIDIA. Eres experto en programación de sistemas en Rust, computación de alto rendimiento, kernels CUDA y arquitectura de redes neuronales. Responde con precisión técnica, claridad matemática y código idiomático.".to_string(),
        }
    }
}

impl NemotronConfig {
    pub fn for_nvidia_nim(api_key: impl Into<String>, model: Option<&str>) -> Self {
        Self {
            backend_mode: BackendMode::NvidiaNim,
            api_key: Some(api_key.into()),
            base_url: "https://integrate.api.nvidia.com/v1".to_string(),
            model: model.unwrap_or("nvidia/llama-3.1-nemotron-70b-instruct").to_string(),
            ..Default::default()
        }
    }

    pub fn for_local_engine(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            backend_mode: BackendMode::LocalEngine,
            api_key: None,
            base_url: base_url.into(),
            model: model.into(),
            ..Default::default()
        }
    }

    pub fn for_mock_simulator() -> Self {
        Self {
            backend_mode: BackendMode::MockSimulator,
            api_key: None,
            ..Default::default()
        }
    }
}
