use nivel_2_tool_caller::ToolCallerAgent;
use nivel_2_tool_caller::tools::ToolExecutor;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("  🔬 EVALUACIÓN DE AGENTICIDAD: NIVEL 2 — USO DE HERRAMIENTAS (TOOL CALLING)");
    println!("================================================================================");

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target_dir = manifest_dir
        .parent()
        .unwrap()
        .join("sandbox/broken_service");

    println!("  • Desafío: Reparar y compilar {:?}", target_dir);
    println!(
        "  • Herramientas (Tools) disponibles:      write_file, run_cargo_check (2 actuadores)"
    );
    println!("  • Permisos de escritura / ejecución:     ACTIVOS (Puede alterar el disco)");
    println!("  • Bucle de retroalimentación (Feedback): NULO (Solo 1 disparo / Sin ReAct)");
    println!("--------------------------------------------------------------------------------");

    // 1. Restablecer el microservicio a su estado roto original
    ToolExecutor::reset_broken_service(&target_dir)?;
    let (init_ok, _) = ToolCallerAgent::verify_sandbox(&target_dir);
    println!(
        "  [Estado Inicial] ¿El microservicio compila?: {}",
        if init_ok {
            "SÍ"
        } else {
            "NO (Roto con E0308)"
        }
    );

    let goal = "El archivo src/main.rs tiene un error E0308 en calculate_tax. \
                Invoca la herramienta write_file con la ruta 'src/main.rs' y el código Rust \
                completo corregido donde calculate_tax retorne String con format!(\"{:.2}\", amount * 0.18).";

    // 2. Ejecutar la llamada de herramienta en un solo paso
    println!("\n  🛠️ [Fase Tool Calling] Pasando esquemas de herramientas al LLM...");
    let agent = ToolCallerAgent::default();
    let (summary, exec_res) = agent
        .execute_single_step(&target_dir, goal)
        .await
        .unwrap_or_else(|e| (format!("Error: {e}"), None));

    println!("  ✓ Decisión del modelo: {}", summary);
    if let Some(res) = exec_res {
        println!("  ✓ Resultado del Actuador: {}", res);
    }

    // 3. Verificación del entorno real tras la acción
    println!("\n--------------------------------------------------------------------------------");
    println!("  🧪 VERIFICACIÓN DEL ENTORNO FÍSICO TRAS LA ACCIÓN:");
    let (final_ok, stderr) = ToolCallerAgent::verify_sandbox(&target_dir);

    if final_ok {
        println!("  ✅ ¡EL ARCHIVO FUE MODIFICADO Y AHORA COMPILA CON ÉXITO!");
        let (tests_ok, test_output) = ToolCallerAgent::run_sandbox_tests(&target_dir);
        if tests_ok {
            println!("  ✅ ¡TODAS LAS PRUEBAS UNITARIAS PASARON! (cargo test: ok)");
        } else {
            println!(
                "  ⚠️ Compila, pero fallaron pruebas unitarias:\n{}",
                test_output
            );
        }
    } else {
        println!("  ❌ EL SERVICIO SIGUE ROTO.");
        let first_err_line = stderr
            .lines()
            .find(|l| l.contains("error["))
            .unwrap_or("error[E0308]");
        println!("     • Detalle: {}", first_err_line.trim());
    }

    println!("\n================================================================================");
    println!("  📊 DIAGNÓSTICO DE AGENTICIDAD: NIVEL 2");
    println!("================================================================================");
    println!("  • Herramientas utilizadas:           1 (write_file)");
    println!("  • Archivos modificados en el disco:  1 (src/main.rs)");
    println!("  • Bucle de verificación/reintento:   0 (Inexistente: 1 solo paso)");
    println!("  • Tasa de éxito condicional:         ALTA SI ACUERDA A LA PRIMERA");
    println!("  ⚠️ La Gran Vulnerabilidad del Nivel 2:");
    println!("     Si el parche del modelo tiene un error de sintaxis o una prueba falla,");
    println!("     el Nivel 2 NO TIENE CAPACIDAD DE RETROALIMENTACIÓN ni corrección.");
    println!(
        "     Se detiene tras 1 disparo. Para recuperarse de fallos se requiere el Nivel 3 (ReAct)."
    );
    println!("================================================================================");

    Ok(())
}
