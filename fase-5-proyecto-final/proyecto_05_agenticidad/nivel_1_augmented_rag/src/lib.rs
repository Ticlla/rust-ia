//! Nivel 1 de Agenticidad: Context-Augmented (RAG / Memoria Semántica).
//! Recupera hechos reales del entorno e inyecta conocimiento antes de consultar al LLM.

pub mod retriever;

use crate::retriever::CodebaseRetriever;
use std::path::Path;
use std::process::Command;

pub struct AugmentedAgent {
    pub model: String,
    pub api_key: Option<String>,
}

impl Default for AugmentedAgent {
    fn default() -> Self {
        Self {
            model: std::env::var("NVIDIA_MODEL")
                .unwrap_or_else(|_| "nvidia/nemotron-3.5-bolt-e4b".to_string()),
            api_key: std::env::var("NVIDIA_API_KEY").ok(),
        }
    }
}

impl AugmentedAgent {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            ..Default::default()
        }
    }

    /// Ejecuta la consulta aumentada con RAG
    pub async fn query_with_rag(
        &self,
        sandbox_dir: &Path,
        goal: &str,
    ) -> Result<(String, String), String> {
        // 1. Recuperar contexto del código y documentación
        let augmented_prompt = CodebaseRetriever::build_augmented_prompt(sandbox_dir, goal)?;

        // 2. Consultar al LLM con el contexto fundamentado
        if let Some(key) = &self.api_key {
            let client = reqwest::Client::new();
            let body = serde_json::json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Eres un asistente experto en Rust. Utiliza el contexto provisto para responder con código Rust exacto y conciso."},
                    {"role": "user", "content": augmented_prompt}
                ],
                "max_tokens": 512,
                "temperature": 0.2
            });

            let res = client
                .post("https://integrate.api.nvidia.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {key}"))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| format!("Error conectando a NVIDIA API: {e}"))?;

            if res.status().is_success() {
                let json: serde_json::Value = res
                    .json()
                    .await
                    .map_err(|e| format!("Error parseando JSON: {e}"))?;

                if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
                    return Ok((augmented_prompt, content.to_string()));
                }
            }
        }

        // Simulación offline fiel a la respuesta con RAG
        let fallback_response =
            "```rust\npub fn calculate_tax(amount: f64) -> String {\n    format!(\"{:.2}\", amount * 0.18)\n}\n```"
                .to_string();

        Ok((augmented_prompt, fallback_response))
    }

    /// Verifica si el microservicio compila
    pub fn verify_sandbox(sandbox_path: &Path) -> (bool, String) {
        let output = Command::new("cargo")
            .arg("check")
            .current_dir(sandbox_path)
            .output();

        match output {
            Ok(out) => {
                let success = out.status.success();
                let err_msg = String::from_utf8_lossy(&out.stderr).to_string();
                (success, err_msg)
            }
            Err(e) => (false, format!("Fallo ejecutando cargo check: {e}")),
        }
    }
}
