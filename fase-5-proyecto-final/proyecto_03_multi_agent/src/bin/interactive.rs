//! Modo Interactivo (REPL) para el Autonomous Multi-Agent Orchestration Framework.

use proyecto_03_multi_agent::agent::GatewayLlmBackend;
use proyecto_03_multi_agent::events::AgentEvent;
use proyecto_03_multi_agent::{create_default_system, create_system_with_backend};
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, BufReader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\x1b[1;36m================================================================================\x1b[0m");
    println!("\x1b[1;32m  🤖 AUTONOMOUS MULTI-AGENT ORCHESTRATION REPL (RUST PURE)\x1b[0m");
    println!("\x1b[1;36m================================================================================\x1b[0m");
    println!("  Comandos disponibles:");
    println!("    \x1b[1;33m:help\x1b[0m     - Muestra esta ayuda");
    println!("    \x1b[1;33m:tools\x1b[0m    - Muestra las herramientas registradas en el Sandbox");
    println!("    \x1b[1;33m:clear\x1b[0m    - Limpia la pantalla de la terminal");
    println!("    \x1b[1;33m:exit\x1b[0m     - Finaliza la sesión interactiva");
    println!("  Opciones de inicio:");
    println!("    \x1b[1;32m--gateway\x1b[0m - Conecta los agentes al LLM Gateway real (Proyecto 02 en :8081)");
    println!("  O simplemente escribe cualquier meta/requerimiento de ingeniería.");
    println!("\x1b[1;36m--------------------------------------------------------------------------------\x1b[0m\n");

    let use_gateway = std::env::args().any(|arg| arg == "--gateway")
        || std::env::var("USE_GATEWAY").is_ok();

    let (orchestrator, event_bus) = if use_gateway {
        let gateway_url = std::env::var("GATEWAY_URL")
            .unwrap_or_else(|_| "http://localhost:8081/v1/chat/completions".into());
        println!("\x1b[1;32m🚀 [MODO IA REAL ACTIVADO]\x1b[0m Conectando a LLM Gateway en: \x1b[33m{}\x1b[0m\n", gateway_url);
        let backend = Arc::new(GatewayLlmBackend::new(gateway_url, "smollm2-135m"));
        create_system_with_backend(backend)
    } else {
        println!("\x1b[1;33m⚡ [MODO DETERMINISTA LOCAL ACTIVADO]\x1b[0m (Para activar inferencia con Proyecto 02, usa '--gateway')\n");
        create_default_system()
    };

    // Suscriptor para streaming de eventos en vivo
    let mut receiver = event_bus.subscribe();
    tokio::spawn(async move {
        while let Ok(event) = receiver.recv().await {
            match event {
                AgentEvent::WorkflowStarted { workflow_id, goal, .. } => {
                    println!("\n\x1b[1;34m[WORKFLOW INICIADO]\x1b[0m ID: \x1b[33m{}\x1b[0m", workflow_id);
                    println!("\x1b[1;34m[OBJETIVO]\x1b[0m {}\n", goal);
                }
                AgentEvent::TaskStatusChanged { task_id, role, status } => {
                    println!("\x1b[1;30m📌 [DAG] {} ({}) -> {}\x1b[0m", task_id, role, status);
                }
                AgentEvent::AgentThinking { agent_role, thought } => {
                    println!("\x1b[1;35m💭 [{}]\x1b[0m \x1b[3m{}\x1b[0m", agent_role, thought);
                }
                AgentEvent::ToolCallStarted { agent_role, tool_name, arguments } => {
                    println!("\x1b[1;33m🛠️  [{}]\x1b[0m Invocando herramienta \x1b[1;32m`{}`\x1b[0m", agent_role, tool_name);
                    println!("    \x1b[2mParámetros: {}\x1b[0m", arguments);
                }
                AgentEvent::ToolCallCompleted { agent_role, tool_name, success, duration_ms, .. } => {
                    println!("\x1b[1;33m📋 [{}]\x1b[0m Tool `{}` completada en {}ms (Éxito: {})", agent_role, tool_name, duration_ms, success);
                }
                AgentEvent::ReviewOutcome { approved, score, feedback } => {
                    let color = if approved { "\x1b[1;32m" } else { "\x1b[1;31m" };
                    println!("{}⚖️  [CONSENSO & AUDITORÍA]\x1b[0m", color);
                    println!("    Aprobado: {} | Puntuación: {:.2} | Dictamen: {}\n", approved, score, feedback);
                }
                AgentEvent::WorkflowCompleted { total_duration_ms, .. } => {
                    println!("\x1b[1;32m✅ [WORKFLOW COMPLETADO CON ÉXITO]\x1b[0m Tiempo: {}ms\n", total_duration_ms);
                }
                AgentEvent::WorkflowFailed { error, .. } => {
                    println!("\x1b[1;31m❌ [WORKFLOW FALLIDO]\x1b[0m Error: {}\n", error);
                }
                _ => {}
            }
        }
    });

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut counter = 1;

    loop {
        print!("\x1b[1;32magent-rs>\x1b[0m ");
        io::stdout().flush()?;

        let line = match reader.next_line().await? {
            Some(l) => l.trim().to_string(),
            None => break, // EOF
        };

        if line.is_empty() {
            continue;
        }

        match line.as_str() {
            ":exit" | ":quit" | "exit" | "quit" => {
                println!("\x1b[1;33mSaliendo del modo interactivo. ¡Hasta pronto!\x1b[0m");
                break;
            }
            ":help" => {
                println!("  Comandos del REPL:");
                println!("    :help     - Muestra este menú");
                println!("    :tools    - Muestra la lista de herramientas en el sandbox");
                println!("    :clear    - Limpia la pantalla");
                println!("    :exit     - Salir");
                println!("  Escribe cualquier meta técnica para coordinar a los agentes.");
                continue;
            }
            ":tools" => {
                println!("\n\x1b[1;33mInventario de Herramientas Tipadas:\x1b[0m");
                println!("  1. \x1b[1;32mcalculator\x1b[0m: Evaluación de expresiones aritméticas (+, -, *, /, %, ^)");
                println!("  2. \x1b[1;32mrag_search\x1b[0m: Búsqueda híbrida de documentos técnicos y referencias");
                println!("  3. \x1b[1;32mrust_code_validator\x1b[0m: Auditoría de sintaxis, balance de delimitadores y detección de `unsafe`\n");
                continue;
            }
            ":clear" => {
                print!("\x1b[2J\x1b[1;1H");
                io::stdout().flush()?;
                continue;
            }
            _ => {}
        }

        let workflow_id = format!("wf-repl-{:03}", counter);
        counter += 1;

        let start = Instant::now();
        let result = orchestrator.run_workflow(&workflow_id, &line).await;

        // Breve pausa para permitir vaciar el canal de eventos en pantalla
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        match result {
            Ok(report) => {
                println!("\x1b[1;36m================================================================================\x1b[0m");
                println!("\x1b[1;32m📊 ENTREGABLE FINAL SINTETIZADO (Flujo {})\x1b[0m", workflow_id);
                println!("\x1b[1;36m================================================================================\x1b[0m");
                if let Some(code) = &report.generated_code {
                    println!("\x1b[1;33m[CÓDIGO RUST GENERADO Y AUDITADO]:\x1b[0m\n{}\n", code);
                } else {
                    println!("\x1b[1;32m💬 [RESPUESTA CONVERSACIONAL DEL ASISTENTE]:\x1b[0m\n{}\n", report.final_synthesis);
                }
                println!("\x1b[2mLatencia total: {:?}\x1b[0m\n", start.elapsed());
            }
            Err(e) => {
                println!("\x1b[1;31mError durante la ejecución del flujo: {}\x1b[0m\n", e);
            }
        }
    }

    Ok(())
}
