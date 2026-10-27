//! Ejecutor de herramientas con Control de Acceso Basado en Roles (RBAC).
//! Cada rol de agente solo puede ejecutar las herramientas para las cuales tiene privilegio explícito.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::process::Command;
use crate::agents::AgentRole;

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
pub struct ReadFileArgs {
    #[serde(default = "default_path", alias = "path", alias = "file_path", alias = "file")]
    pub relative_path: String,
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
    /// Genera el esquema de herramientas filtrado exclusivamente para el rol dado (RBAC)
    pub fn get_tools_schema_for_role(role: AgentRole) -> serde_json::Value {
        let mut tools = Vec::new();

        for tool_name in role.permitted_tools() {
            match tool_name {
                "read_file" => tools.push(serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Lee el código de un archivo para inspeccionar su contenido.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "relative_path": {
                                    "type": "string",
                                    "description": "Ruta relativa del archivo (ej: 'src/main.rs')"
                                }
                            },
                            "required": ["relative_path"]
                        }
                    }
                })),
                "write_file" => tools.push(serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Sobreescribe un archivo en el disco con el código corregido.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "relative_path": {
                                    "type": "string",
                                    "description": "Ruta relativa del archivo (ej: 'src/main.rs')"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Contenido completo en código Rust."
                                }
                            },
                            "required": ["relative_path", "content"]
                        }
                    }
                })),
                "run_cargo_check" => tools.push(serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": "run_cargo_check",
                        "description": "Ejecuta 'cargo check' para comprobar si compila y extraer los errores del compilador rustc.",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                })),
                "run_cargo_test" => tools.push(serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": "run_cargo_test",
                        "description": "Ejecuta 'cargo test' para validar si todas las pruebas unitarias pasan.",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                })),
                _ => {}
            }
        }

        serde_json::Value::Array(tools)
    }

    /// Ejecuta una herramienta validando previamente el permiso del rol solicitante (RBAC)
    pub fn execute_tool(
        sandbox_dir: &Path,
        role: AgentRole,
        tool_name: &str,
        args_json: &str,
    ) -> String {
        // 1. Verificación de Seguridad RBAC
        let allowed = role.permitted_tools();
        if !allowed.contains(&tool_name) {
            return format!(
                "⛔ VIOLACIÓN DE SEGURIDAD (RBAC): El rol {:?} no tiene permiso para invocar la herramienta '{}'. \
                Herramientas permitidas: {:?}",
                role, tool_name, allowed
            );
        }

        // 2. Normalización de rutas relativas
        let normalize_path = |p: &str| -> String {
            let s = p.trim_start_matches('/');
            let s = s.strip_prefix("sandbox/broken_service/").unwrap_or(s);
            let s = s.strip_prefix("broken_service/").unwrap_or(s);
            if s.is_empty() { "src/main.rs".to_string() } else { s.to_string() }
        };

        // 3. Despacho de la herramienta
        match tool_name {
            "read_file" => {
                let args: Result<ReadFileArgs, _> = serde_json::from_str(args_json);
                let raw_path = args.map(|a| a.relative_path).unwrap_or_else(|_| "src/main.rs".to_string());
                let rel_path = normalize_path(&raw_path);
                let target = sandbox_dir.join(&rel_path);

                match fs::read_to_string(&target) {
                    Ok(content) => format!("Contenido de '{}':\n{}", rel_path, content),
                    Err(e) => format!("Error leyendo '{}': {e}", rel_path),
                }
            }
            "write_file" => {
                let args: Result<WriteFileArgs, _> = serde_json::from_str(args_json);
                match args {
                    Ok(a) => {
                        let rel_path = normalize_path(&a.relative_path);
                        let target = sandbox_dir.join(&rel_path);
                        if let Some(parent) = target.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
                        let mut content = a.content;
                        if content.starts_with("r#\"") && content.ends_with("\"#") {
                            content = content[3..content.len() - 2].to_string();
                        }
                        match fs::write(&target, &content) {
                            Ok(_) => format!("Archivo '{}' modificado exitosamente ({} bytes escritos).", rel_path, content.len()),
                            Err(e) => format!("Error escribiendo en '{}': {e}", rel_path),
                        }
                    }
                    Err(e) => format!("Error deserializando argumentos de write_file: {e}"),
                }
            }
            "run_cargo_check" => {
                match Command::new("cargo").arg("check").current_dir(sandbox_dir).output() {
                    Ok(out) => {
                        let status = if out.status.success() { "EXITO: El código compila sin errores." } else { "FALLO: Errores de compilación:" };
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        format!("{status}\n{stderr}")
                    }
                    Err(e) => format!("Error ejecutando cargo check: {e}"),
                }
            }
            "run_cargo_test" => {
                match Command::new("cargo").arg("test").current_dir(sandbox_dir).output() {
                    Ok(out) => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        if out.status.success() {
                            format!("EXITO: Todas las pruebas unitarias pasaron.\n{stdout}")
                        } else {
                            format!("FALLO: Las pruebas no pasaron:\n{stdout}\n{stderr}")
                        }
                    }
                    Err(e) => format!("Error ejecutando cargo test: {e}"),
                }
            }
            _ => format!("Error: Herramienta desconocida '{}'", tool_name),
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
