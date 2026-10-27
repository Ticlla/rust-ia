//! Herramientas (Tools / Actuadores) para el Nivel 2.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteFileArgs {
    #[serde(default = "default_path", alias = "path", alias = "file_path", alias = "file")]
    pub relative_path: String,
    #[serde(alias = "code", alias = "text", alias = "data")]
    pub content: String,
}

fn default_path() -> String {
    "src/main.rs".to_string()
}

pub struct ToolExecutor;

impl ToolExecutor {
    /// Genera la lista de esquemas de herramientas para pasar a la API del LLM
    pub fn get_tools_schema() -> serde_json::Value {
        serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Escribe o sobreescribe un archivo en el disco con el contenido proporcionado.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "relative_path": {
                                "type": "string",
                                "description": "Ruta relativa del archivo (ej: 'src/main.rs')"
                            },
                            "content": {
                                "type": "string",
                                "description": "Contenido completo del archivo en código Rust."
                            }
                        },
                        "required": ["relative_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_cargo_check",
                    "description": "Ejecuta 'cargo check' en el microservicio para verificar si compila sin errores.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ])
    }

    /// Ejecuta una llamada de herramienta específica sobre el sandbox
    pub fn execute_tool(sandbox_dir: &Path, tool_name: &str, args_json: &str) -> Result<String, String> {
        match tool_name {
            "write_file" => {
                let args: WriteFileArgs = serde_json::from_str(args_json)
                    .map_err(|e| format!("Error deserializando argumentos de write_file: {e}"))?;

                let target_path = sandbox_dir.join(&args.relative_path);
                if let Some(parent) = target_path.parent() {
                    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }

                fs::write(&target_path, &args.content)
                    .map_err(|e| format!("Error escribiendo en {}: {e}", target_path.display()))?;

                Ok(format!("Archivo '{}' escrito con éxito ({} bytes)", args.relative_path, args.content.len()))
            }
            "run_cargo_check" => {
                let output = Command::new("cargo")
                    .arg("check")
                    .current_dir(sandbox_dir)
                    .output()
                    .map_err(|e| format!("Fallo ejecutando cargo check: {e}"))?;

                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let status = if output.status.success() { "COMPILACIÓN EXITOSA" } else { "FALLO DE COMPILACIÓN" };

                Ok(format!("{status}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"))
            }
            _ => Err(format!("Herramienta desconocida: '{tool_name}'")),
        }
    }

    /// Restaura el error intencional original en el microservicio
    pub fn reset_broken_service(sandbox_dir: &Path) -> Result<(), String> {
        let main_rs = sandbox_dir.join("src/main.rs");
        let broken_code = r#"//! Microservicio con error intencional para evaluar grados de agenticidad.

pub fn calculate_tax(amount: f64) -> String {
    // ERROR E0308: expected `String`, found `f64`
    amount * 0.18
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
        fs::write(&main_rs, broken_code)
            .map_err(|e| format!("Fallo restableciendo broken_service: {e}"))?;
        Ok(())
    }
}
