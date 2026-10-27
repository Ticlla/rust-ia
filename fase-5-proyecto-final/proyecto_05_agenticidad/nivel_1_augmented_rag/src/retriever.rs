//! Módulo Retriever: Indexación y recuperación de contexto (RAG) para Nivel 1.

use std::fs;
use std::path::Path;

pub struct CodebaseRetriever;

impl CodebaseRetriever {
    /// Lee e indexa el código fuente relevante del microservicio
    pub fn retrieve_code_context(sandbox_dir: &Path) -> Result<String, String> {
        let main_rs = sandbox_dir.join("src/main.rs");
        if !main_rs.exists() {
            return Err(format!("Archivo no encontrado en {:?}", main_rs));
        }

        let content = fs::read_to_string(&main_rs)
            .map_err(|e| format!("Error leyendo {}: {e}", main_rs.display()))?;

        Ok(content)
    }

    /// Base de conocimiento RAG de errores comunes del compilador de Rust
    pub fn retrieve_rust_error_knowledge(error_code: &str) -> &'static str {
        match error_code {
            "E0308" => {
                "Documentación oficial de rustc [E0308]:\n\
                 Un tipo esperado no coincide con el tipo encontrado.\n\
                 En este caso: la función declara retornar `String`, pero la expresión final evalúa a `f64`.\n\
                 Solución idiomática: utilizar `format!(\"{:.2}\", amount * 0.18)` para convertir el cálculo a un String formateado."
            }
            _ => "Error de compilación genérico de Rust.",
        }
    }

    /// Construye el prompt aumentado con el contexto recuperado
    pub fn build_augmented_prompt(sandbox_dir: &Path, user_goal: &str) -> Result<String, String> {
        let code = Self::retrieve_code_context(sandbox_dir)?;
        let doc = Self::retrieve_rust_error_knowledge("E0308");

        Ok(format!(
            "=== CONTEXTO DEL CÓDIGO FUENTE (sandbox/broken_service/src/main.rs) ===\n\
             ```rust\n\
             {}\n\
             ```\n\n\
             === BASE DE CONOCIMIENTO TÉCNICO RECUPERADA (RAG) ===\n\
             {}\n\n\
             === TAREA ===\n\
             {}\n\n\
             Proporciona la función en Rust corregida.",
            code.trim(),
            doc,
            user_goal
        ))
    }
}
