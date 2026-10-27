//! Nivel 0 de Agenticidad: El LLM Pasivo (Zero-Shot).
//! Solo genera texto estático sin herramientas, sin bucle y sin actuar en el entorno.

use std::path::Path;
use std::process::Command;

pub struct ZeroShotAgent {
    pub model: String,
    pub api_key: Option<String>,
}

impl Default for ZeroShotAgent {
    fn default() -> Self {
        Self {
            model: std::env::var("NVIDIA_MODEL")
                .unwrap_or_else(|_| "nvidia/nemotron-3.5-bolt-e4b".to_string()),
            api_key: std::env::var("NVIDIA_API_KEY").ok(),
        }
    }
}

impl ZeroShotAgent {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            ..Default::default()
        }
    }

    /// Ejecuta la consulta pasiva al LLM sin herramientas ni permisos de modificación
    pub async fn query(&self, prompt: &str) -> Result<String, String> {
        if let Some(key) = &self.api_key {
            let client = reqwest::Client::new();
            let body = serde_json::json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Eres un asistente de IA. Responde con texto plano a la consulta del usuario."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 512,
                "temperature": 0.5
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
                    return Ok(content.to_string());
                }
            }
        }

        // Simulación offline si no hay conexión o API key
        Ok("Para solucionar el error de tipos E0308 en sandbox/broken_service/src/main.rs, \
           debes cambiar la función para que formatee el cálculo a un String: \
           utiliza `format!(\"{:.2}\", amount * 0.18)` en lugar de retornar directamente el float."
            .to_string())
    }

    /// Evalúa si el código en el sandbox fue realmente reparado
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
