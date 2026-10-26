//! Demostración interactiva por terminal de la orquestación multi-agente en Rust.

use proyecto_03_multi_agent::create_default_system;
use proyecto_03_multi_agent::events::AgentEvent;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("🤖 Autonomous Multi-Agent Orchestration Framework (ReAct + Actor Model)");
    println!("================================================================================\n");

    let (orchestrator, event_bus) = create_default_system();

    // Tarea de escucha en segundo plano para el bus reactivo de eventos
    let mut receiver = event_bus.subscribe();
    let listener_handle = tokio::spawn(async move {
        while let Ok(event) = receiver.recv().await {
            match event {
                AgentEvent::WorkflowStarted { workflow_id, goal, .. } => {
                    println!("🚀 [WORKFLOW INICIADO] ID: {workflow_id}");
                    println!("🎯 [META]: {goal}\n");
                }
                AgentEvent::AgentThinking { agent_role, thought } => {
                    println!("💭 [{agent_role}] Razonamiento (Thought):");
                    println!("   {thought}\n");
                }
                AgentEvent::ToolCallStarted { agent_role, tool_name, arguments } => {
                    println!("🛠️  [{agent_role}] Invocando Tool (Action): `{tool_name}`");
                    println!("   Parámetros: {arguments}\n");
                }
                AgentEvent::ToolCallCompleted { agent_role, tool_name, success, duration_ms, .. } => {
                    println!("📋 [{agent_role}] Tool Completada: `{tool_name}` en {duration_ms}ms (Éxito: {success})\n");
                }
                AgentEvent::ReviewOutcome { approved, score, feedback } => {
                    println!("⚖️  [CONSENSO & AUDITORÍA]");
                    println!("   Aprobado: {approved} | Calidad: {score:.2} | Veredicto: {feedback}\n");
                }
                AgentEvent::TaskStatusChanged { task_id, role, status } => {
                    println!("📌 [DAG TASK] {task_id} ({role}) -> {status}");
                }
                AgentEvent::WorkflowCompleted { total_duration_ms, .. } => {
                    println!("\n✅ [WORKFLOW COMPLETADO] Tiempo total: {total_duration_ms}ms\n");
                }
                _ => {}
            }
        }
    });

    let goal = "Diseñar e implementar una estructura de caché concurrente segura en Rust con tipos genéricos y evaluación de RAG.";
    let workflow_id = "wf-demo-001";

    let start = Instant::now();
    let report = orchestrator.run_workflow(workflow_id, goal).await?;

    // Esperar un instante para vaciar el bus de eventos
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    listener_handle.abort();

    println!("================================================================================");
    println!("📊 RESUMEN FINAL DEL ENTREGABLE MULTI-AGENTE");
    println!("================================================================================");
    if let Some(research) = &report.research_summary {
        println!("\n[1. INVESTIGACIÓN TÉCNICA]\n{research}");
    }
    if let Some(code) = &report.generated_code {
        println!("\n[2. CÓDIGO GENERADO]\n{code}");
    }
    if let Some(review) = &report.review_feedback {
        println!("\n[3. AUDITORÍA DEL REVISOR]\n{review}");
    }
    println!("\nTiempo total medido: {:?}", start.elapsed());

    Ok(())
}
