use std::path::Path;
use nivel_0_zero_shot::ZeroShotAgent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("  🔬 EVALUACIÓN DE AGENTICIDAD: NIVEL 0 — EL LLM PASIVO (ZERO-SHOT)");
    println!("================================================================================");

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target_dir = manifest_dir.parent().unwrap().join("sandbox/broken_service");

    println!("  • Desafío: Reparar error de compilación en {:?}", target_dir);
    println!("  • Herramientas (Tools) disponibles:      0 (Ninguna)");
    println!("  • Permisos de escritura / ejecución:     0 (Modo solo lectura)");
    println!("  • Bucle de retroalimentación (ReAct):    Nulo (1 solo intento)");
    println!("--------------------------------------------------------------------------------");

    // 1. Estado inicial del sandbox
    let (init_ok, _) = ZeroShotAgent::verify_sandbox(&target_dir);
    println!("  [Estado Inicial] ¿El microservicio compila?: {}", if init_ok { "SÍ" } else { "NO (Roto con E0308)" });

    // 2. Ejecutar prompt al modelo
    let prompt = "El microservicio en sandbox/broken_service no compila debido a un error de tipos. \
                  Corrige la función calculate_tax para que devuelva un String formateado a 2 decimales y pase cargo test.";

    println!("\n  📤 Enviando meta al LLM (Nivel 0)...");
    let agent = ZeroShotAgent::default();
    let response = agent.query(prompt).await.unwrap_or_else(|e| format!("Error: {e}"));

    println!("\n  🤖 RESPUESTA TEXTUAL DEL LLM:");
    println!("  \"{}\"", response.trim());

    // 3. Evaluar si el entorno real cambió
    println!("\n--------------------------------------------------------------------------------");
    println!("  🧪 VERIFICACIÓN DEL ENTORNO FÍSICO:");
    let (final_ok, stderr) = ZeroShotAgent::verify_sandbox(&target_dir);

    if final_ok {
        println!("  ✅ ¡Sorprendente! El servicio compila.");
    } else {
        println!("  ❌ EL SERVICIO SIGUE ROTO.");
        let first_err_line = stderr.lines().find(|l| l.contains("error[")).unwrap_or("error[E0308]");
        println!("     • Detalle: {}", first_err_line.trim());
    }

    println!("\n================================================================================");
    println!("  📊 DIAGNÓSTICO DE AGENTICIDAD: NIVEL 0");
    println!("================================================================================");
    println!("  • Acciones ejecutadas en el disco:   0");
    println!("  • Archivos modificados:              0");
    println!("  • Intentos de autocorrección:        0");
    println!("  • Tasa de resolución autónoma:       0.0% (FRACASO TOTAL)");
    println!("  💡 Conclusión Nivel 0:");
    println!("     Un LLM sin herramientas es un consejero ciego: sabe la teoría, pero carece");
    println!("     de manos para intervenir en el software real.");
    println!("================================================================================");

    Ok(())
}
