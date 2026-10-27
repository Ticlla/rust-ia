//! Nivel 2 de Agenticidad: Uso Guiado de Herramientas (Tool Use / Function Calling).
//! Posee actuadores (manos) para escribir archivos y ejecutar comandos, pero opera en un solo paso (sin bucle ReAct).

pub mod tools;

use std::path::Path;
use std::process::Command;
use crate::tools::{ToolCall, ToolExecutor};

pub struct ToolCallerAgent {
    pub model: String,
    pub api_key: Option<String>,
}

impl Default for ToolCallerAgent {
    fn default() -> Self {
        Self {
            model: std::env::var("NVIDIA_MODEL")
                .unwrap_or_else(|_| "nvidia/nemotron-3.5-bolt-e4b".to_string()),
            api_key: std::env::var("NVIDIA_API_KEY").ok(),
        }
    }
}

impl ToolCallerAgent {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            ..Default::default()
        }
    }

    /// Ejecuta el ciclo de 1 solo paso de Function Calling (Nivel 2)
    pub async fn execute_single_step(
        &self,
        sandbox_dir: &Path,
        goal: &str,
    ) -> Result<(String, Option<String>), String> {
        let tools_schema = ToolExecutor::get_tools_schema();

        let system_msg = "Eres un agente con acceso a herramientas. Para resolver la tarea, \
                          debes invocar la herramienta 'write_file' con el código Rust completo \
                          corregido para src/main.rs.";

        if let Some(key) = &self.api_key {
            let client = reqwest::Client::new();
            let body = serde_json::json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": goal}
                ],
                "tools": tools_schema,
                "tool_choice": "auto",
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

                let choice = &json["choices"][0]["message"];
                if let Some(tool_calls_json) = choice.get("tool_calls") {
                    if let Ok(calls) = serde_json::from_value::<Vec<ToolCall>>(tool_calls_json.clone()) {
                        if let Some(first_call) = calls.first() {
                            let tool_name = &first_call.function.name;
                            let args = &first_call.function.arguments;

                            // Ejecutar la herramienta en el sandbox local
                            let exec_result = ToolExecutor::execute_tool(sandbox_dir, tool_name, args)?;
                            return Ok((
                                format!("Herramienta ejecutada: '{}' con argumentos: {}", tool_name, args),
                                Some(exec_result),
                            ));
                        }
                    }
                }

                // Si respondió texto en lugar de tool_call estructurado
                if let Some(content) = choice["content"].as_str() {
                    return Ok((format!("Respuesta en texto (sin llamada a herramienta): {content}"), None));
                }
            }
        }

        // Fallback simulado de Function Calling para Nivel 2
        let fixed_code = r#"//! Microservicio con error intencional reparado por Nivel 2

pub fn calculate_tax(amount: f64) -> String {
    format!("{:.2}", amount * 0.18)
}

fn main() {
    let tax = calculate_tax(100.0);
    println!("Impuesto: {}", tax);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_tax() {
        assert_eq!(calculate_tax(100.0), "18.00");
    }
}
"#;
        let args = serde_json::json!({
            "relative_path": "src/main.rs",
            "content": fixed_code
        })
        .to_string();

        let exec_result = ToolExecutor::execute_tool(sandbox_dir, "write_file", &args)?;
        Ok((
            "Herramienta ejecutada: 'write_file' en 'src/main.rs'".to_string(),
            Some(exec_result),
        ))
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

    /// Ejecuta las pruebas del microservicio
    pub fn run_sandbox_tests(sandbox_path: &Path) -> (bool, String) {
        let output = Command::new("cargo")
            .arg("test")
            .current_dir(sandbox_path)
            .output();

        match output {
            Ok(out) => {
                let success = out.status.success();
                let output_str = format!(
                    "{}\n{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                (success, output_str)
            }
            Err(e) => (false, format!("Fallo ejecutando cargo test: {e}")),
        }
    }
}
