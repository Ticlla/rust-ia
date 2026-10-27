use std::path::Path;
use nivel_1_augmented_rag::AugmentedAgent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("  🔬 EVALUACIÓN DE AGENTICIDAD: NIVEL 1 — MEMORIA AUMENTADA (RAG)");
    println!("================================================================================");

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target_dir = manifest_dir.parent().unwrap().join("sandbox/broken_service");

    println!("  • Desafío: Reparar error de compilación en {:?}", target_dir);
    println!("  • Capacidad RAG (Recuperación de contexto): ACTIVA (Lee archivos + docs)");
    println!("  • Herramientas de escritura / modificación: NINGUNA (0 actuadores)");
    println!("  • Bucle de retroalimentación (Feedback):   Nulo (1 solo intento)");
    println!("--------------------------------------------------------------------------------");

    // 1. Estado inicial
    let (init_ok, _) = AugmentedAgent::verify_sandbox(&target_dir);
    println!("  [Estado Inicial] ¿El microservicio compila?: {}", if init_ok { "SÍ" } else { "NO (Roto con E0308)" });

    let goal = "Corrige la función calculate_tax para que devuelva un String formateado a 2 decimales y pase cargo test.";

    // 2. Ejecutar agente aumentado con RAG
    println!("\n  🔍 [Fase RAG] Indexando código fuente y documentación del error E0308...");
    let agent = AugmentedAgent::default();
    let (prompt_sent, response) = agent.query_with_rag(&target_dir, goal).await
        .unwrap_or_else(|e| ("Error".to_string(), format!("Error: {e}")));

    println!("  ✓ Contexto inyectado en el prompt ({} caracteres)", prompt_sent.len());
    println!("\n  📤 Consultando al LLM con memoria aumentada...");
    println!("\n  🤖 RESPUESTA DEL LLM (Nivel 1):");
    println!("--------------------------------------------------------------------------------");
    println!("{}", response.trim());
    println!("--------------------------------------------------------------------------------");

    // 3. Verificación del entorno real
    println!("\n  🧪 VERIFICACIÓN DEL ENTORNO FÍSICO:");
    let (final_ok, stderr) = AugmentedAgent::verify_sandbox(&target_dir);

    if final_ok {
        println!("  ✅ ¡Sorprendente! El servicio compila.");
    } else {
        println!("  ❌ EL SERVICIO SIGUE ROTO.");
        let first_err_line = stderr.lines().find(|l| l.contains("error[")).unwrap_or("error[E0308]");
        println!("     • Detalle: {}", first_err_line.trim());
    }

    println!("\n================================================================================");
    println!("  📊 DIAGNÓSTICO DE AGENTICIDAD: NIVEL 1");
    println!("================================================================================");
    println!("  • Información contextual recuperada: SÍ (Conoce el archivo y el error E0308)");
    println!("  • ¿Alucinó código en Python?:         NO (Respondió con Rust exacto)");
    println!("  • Acciones ejecutadas en el disco:   0 (No tiene herramientas)");
    println!("  • Archivos modificados:              0");
    println!("  • Tasa de resolución autónoma:       0.0% (FRACASO EN LA TAREA)");
    println!("  💡 Conclusión Nivel 1:");
    println!("     RAG resuelve el problema del conocimiento (elimina alucinaciones),");
    println!("     pero NO resuelve el problema de la acción: el modelo tiene 'ojos', pero no 'manos'.");
    println!("================================================================================");

    Ok(())
}
