//! Nivel 3 de Agenticidad: Bucle Autónomo ReAct y Autocorrección (Looping & Self-Correction).
//! Ejecuta un ciclo cerrado: Razonamiento -> Acción (Tool Call) -> Observación -> Reflexión -> Reintento.

pub mod tools;

use crate::tools::{ToolCall, ToolExecutor};
use serde_json::Value;
use std::path::Path;
use std::process::Command;

pub struct ReActAgent {
    pub model: String,
    pub api_key: Option<String>,
    pub max_steps: usize,
}

impl Default for ReActAgent {
    fn default() -> Self {
        Self {
            model: std::env::var("NVIDIA_MODEL")
                .unwrap_or_else(|_| "nvidia/nemotron-3.5-bolt-e4b".to_string()),
            api_key: std::env::var("NVIDIA_API_KEY").ok(),
            max_steps: 8,
        }
    }
}

pub struct StepTrace {
    pub step: usize,
    pub thought: Option<String>,
    pub action: String,
    pub observation: String,
}

impl ReActAgent {
    pub fn new(api_key: Option<String>, max_steps: usize) -> Self {
        Self {
            api_key,
            max_steps,
            ..Default::default()
        }
    }

    /// Ejecuta el bucle ReAct iterativo hasta completar la meta o agotar max_steps
    pub async fn run_react_loop(&self, sandbox_dir: &Path, goal: &str) -> (bool, Vec<StepTrace>) {
        let tools_schema = ToolExecutor::get_tools_schema();
        let mut traces = Vec::new();

        let system_prompt = "Eres un Agente Autónomo de Nivel 3 experto en Rust. \
            Tienes un bucle ReAct completo con herramientas para leer archivos, editarlos, \
            ejecutar 'cargo check' y 'cargo test'.\n\
            Tu objetivo es diagnosticar el problema en el sandbox, escribir el código correcto, \
            comprobar la compilación y asegurarte de que 'cargo test' pase exitosamente.\n\
            REGLA CRÍTICA: La función 'calculate_tax' debe retornar el valor formateado a 2 decimales usando format!(\"{:.2}\", amount * 0.18) para cumplir con la aserción de prueba. No modifiques los tests.\n\
            Si encuentras un error al compilar o al ejecutar las pruebas, lee la observación recibida, reflexiona sobre la causa y \
            corrige el archivo hasta que 'cargo test' pase al 100%. \
            Cuando 'cargo test' pase con éxito, concluye indicando que la tarea está completada.";

        let mut messages = vec![
            serde_json::json!({"role": "system", "content": system_prompt}),
            serde_json::json!({"role": "user", "content": goal}),
        ];

        let mut success = false;
        let mut step = 0;

        while step < self.max_steps && !success {
            step += 1;

            if let Some(key) = &self.api_key {
                let client = reqwest::Client::new();
                let body = serde_json::json!({
                    "model": self.model,
                    "messages": messages,
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
                    .await;

                if let Ok(resp) = res {
                    if resp.status().is_success() {
                        if let Ok(json) = resp.json::<Value>().await {
                            let choice = &json["choices"][0]["message"];
                            let content = choice["content"].as_str().map(|s| s.to_string());

                            if let Some(tool_calls_val) = choice.get("tool_calls") {
                                if let Ok(calls) =
                                    serde_json::from_value::<Vec<ToolCall>>(tool_calls_val.clone())
                                {
                                    if let Some(first_call) = calls.first() {
                                        let tool_name = &first_call.function.name;
                                        let args = &first_call.function.arguments;

                                        // Ejecutar la herramienta en el sandbox
                                        let observation = ToolExecutor::execute_tool(
                                            sandbox_dir,
                                            tool_name,
                                            args,
                                        );

                                        traces.push(StepTrace {
                                            step,
                                            thought: content.clone(),
                                            action: format!("{tool_name}({args})"),
                                            observation: observation.clone(),
                                        });

                                        // Guardar el tool_call en el historial del LLM
                                        messages.push(choice.clone());

                                        // Guardar la observación para el siguiente ciclo del bucle
                                        messages.push(serde_json::json!({
                                            "role": "tool",
                                            "tool_call_id": first_call.id,
                                            "content": observation
                                        }));

                                        if tool_name == "run_cargo_test"
                                            && observation.contains("EXITO")
                                        {
                                            success = true;
                                            break;
                                        }

                                        continue;
                                    }
                                }
                            }

                            // Si respondió solo texto sin invocar herramientas
                            if let Some(txt) = content {
                                traces.push(StepTrace {
                                    step,
                                    thought: Some(txt.clone()),
                                    action: "Respuesta conversacional".to_string(),
                                    observation: "Sin acción física".to_string(),
                                });

                                if txt.to_lowercase().contains("completada")
                                    || txt.to_lowercase().contains("exitoso")
                                {
                                    // Verificar objetivamente si compila
                                    let (check_ok, _) = crate::tools::ToolExecutor::execute_tool(
                                        sandbox_dir,
                                        "run_cargo_check",
                                        "{}",
                                    )
                                    .starts_with("EXITO")
                                    .then_some((true, ()))
                                    .unwrap_or((false, ()));
                                    if check_ok {
                                        success = true;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // Simulación pedagógica del bucle ReAct si no hay conexión de red
                match step {
                    1 => {
                        let obs = ToolExecutor::execute_tool(sandbox_dir, "run_cargo_check", "{}");
                        traces.push(StepTrace {
                            step,
                            thought: Some("Primero ejecuto 'cargo check' para inspeccionar el error del compilador.".to_string()),
                            action: "run_cargo_check()".to_string(),
                            observation: obs,
                        });
                    }
                    2 => {
                        let fixed_code = r#"//! Microservicio reparado exitosamente por Nivel 3 ReAct

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

                        let obs = ToolExecutor::execute_tool(sandbox_dir, "write_file", &args);
                        traces.push(StepTrace {
                            step,
                            thought: Some("El compilador reportó E0308. Reescribo calculate_tax retornando format!(\"{:.2}\", amount * 0.18).".to_string()),
                            action: "write_file('src/main.rs')".to_string(),
                            observation: obs,
                        });
                    }
                    3 => {
                        let obs = ToolExecutor::execute_tool(sandbox_dir, "run_cargo_test", "{}");
                        traces.push(StepTrace {
                            step,
                            thought: Some("Verifico el resultado ejecutando las pruebas unitarias con cargo test.".to_string()),
                            action: "run_cargo_test()".to_string(),
                            observation: obs.clone(),
                        });
                        if obs.contains("EXITO") {
                            success = true;
                        }
                    }
                    _ => break,
                }
            }
        }

        // Verificación objetiva final en el disco
        let final_check = Command::new("cargo")
            .arg("test")
            .current_dir(sandbox_dir)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        (success && final_check, traces)
    }
}
