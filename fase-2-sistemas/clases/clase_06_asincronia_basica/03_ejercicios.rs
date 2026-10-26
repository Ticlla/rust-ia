//! ============================================================================
//! 🎯 FASE 2 - CLASE 06: Retos Prácticos con Async / Await en IA
//! ============================================================================
//!
//! Ejecutar el reto:
//!     cargo run --bin c06_ejercicios
//!
//! Ejecutar las pruebas unitarias:
//!     cargo test --bin c06_ejercicios

use futures::executor::block_on;
use futures::join;

/// Simula una consulta asíncrona a un LLM en la nube.
pub async fn consultar_llm_externo(proveedor: &str, prompt: &str) -> Result<String, String> {
    if prompt.trim().is_empty() {
        return Err("Prompt vacío".to_string());
    }

    // Simula una respuesta exitosa del LLM:
    Ok(format!("[{proveedor}]: Respuesta generada para '{prompt}'"))
}

/// Consulta 3 proveedores de LLM de forma concurrente usando `futures::join!`.
pub async fn orquestar_consulta_triple(
    prompt: &str,
) -> (
    Result<String, String>,
    Result<String, String>,
    Result<String, String>,
) {
    // 💡 Las 3 consultas se disparan simultáneamente en cooperativo:
    join!(
        consultar_llm_externo("OpenAI-GPT4", prompt),
        consultar_llm_externo("Anthropic-Claude", prompt),
        consultar_llm_externo("Meta-Llama3", prompt)
    )
}

/// Filtra y consolida únicamente las respuestas exitosas de los proveedores.
pub fn filtrar_respuestas_exitosas(respuestas: Vec<Result<String, String>>) -> Vec<String> {
    respuestas.into_iter().filter_map(|r| r.ok()).collect()
}

fn main() {
    println!("============================================================");
    println!("🎯 RETO PRÁCTICO: Orquestador Asíncrono de Modelos de IA");
    println!("============================================================\n");

    let prompt = "¿Qué ventajas tiene Rust para infraestructura de LLMs?";
    println!("Prompt del usuario: \"{prompt}\"\n");

    println!("Consultando 3 proveedores en paralelo con async/join!...");
    let futuro = orquestar_consulta_triple(prompt);
    let (res_a, res_b, res_c) = block_on(futuro);

    println!("\n--- Respuestas Recibidas ---");
    println!("  1. {:?}", res_a);
    println!("  2. {:?}", res_b);
    println!("  3. {:?}", res_c);

    let consolidadas = filtrar_respuestas_exitosas(vec![res_a, res_b, res_c]);
    println!("\nTotal de respuestas exitosas recolectadas: {}", consolidadas.len());

    println!("\n✅ Reto ejecutado exitosamente.");
}

// =============================================================================
// PRUEBAS UNITARIAS AUTOMÁTICAS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orquestacion_triple_exitosa() {
        let (r1, r2, r3) = block_on(orquestar_consulta_triple("Explica Tokio"));

        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert!(r3.is_ok());

        assert!(r1.unwrap().contains("OpenAI"));
        assert!(r2.unwrap().contains("Anthropic"));
        assert!(r3.unwrap().contains("Meta"));
    }

    #[test]
    fn test_prompt_vacio_falla() {
        let res = block_on(consultar_llm_externo("TestProvider", "   "));
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), "Prompt vacío");
    }

    #[test]
    fn test_filtrar_exitosas() {
        let respuestas = vec![
            Ok("Respuesta 1".to_string()),
            Err("Fallo de red".to_string()),
            Ok("Respuesta 2".to_string()),
        ];

        let filtradas = filtrar_respuestas_exitosas(respuestas);
        assert_eq!(filtradas.len(), 2);
        assert_eq!(filtradas[0], "Respuesta 1");
        assert_eq!(filtradas[1], "Respuesta 2");
    }
}
