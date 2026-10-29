use nivel_4_multi_agent::MultiAgentOrchestrator;
use nivel_4_multi_agent::agents::AgentRole;
use nivel_4_multi_agent::dag::{TaskDag, TaskNode};
use nivel_4_multi_agent::tools::ToolExecutor;
use std::path::Path;

#[test]
fn test_dag_cycle_detection() {
    let mut dag = TaskDag::new();
    dag.add_task(TaskNode::new(
        "A",
        "Tarea A",
        AgentRole::Analyzer,
        "",
        vec!["B".to_string()],
    ));
    dag.add_task(TaskNode::new(
        "B",
        "Tarea B",
        AgentRole::Coder,
        "",
        vec!["A".to_string()],
    ));

    let result = dag.topological_batches();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("ciclo"));
}

#[test]
fn test_rbac_permission_enforcement() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = manifest_dir
        .parent()
        .unwrap()
        .join("sandbox/broken_service");

    // Analyzer intentando escribir
    let res = ToolExecutor::execute_tool(&target, AgentRole::Analyzer, "write_file", "{}");
    assert!(res.contains("VIOLACIÓN DE SEGURIDAD"));

    // Coder intentando ejecutar pruebas (auto-aprobación prohibida)
    let res2 = ToolExecutor::execute_tool(&target, AgentRole::Coder, "run_cargo_test", "{}");
    assert!(res2.contains("VIOLACIÓN DE SEGURIDAD"));

    // Auditor intentando escribir
    let res3 = ToolExecutor::execute_tool(&target, AgentRole::Auditor, "write_file", "{}");
    assert!(res3.contains("VIOLACIÓN DE SEGURIDAD"));
}

#[tokio::test]
async fn test_multi_agent_orchestration_resolves_sandbox() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = manifest_dir
        .parent()
        .unwrap()
        .join("sandbox/broken_service");

    // 1. Restablecer el sandbox
    ToolExecutor::reset_broken_service(&target).unwrap();

    // 2. Ejecutar la orquestación en modo determinista
    let orchestrator = MultiAgentOrchestrator::new(None);
    let (success, events, dag) = orchestrator
        .run_orchestration(&target, "Reparar microservicio")
        .await;

    assert!(
        success,
        "La orquestación multi-agente debe completarse con éxito"
    );
    assert!(
        events.len() >= 3,
        "Deben haberse registrado al menos 3 eventos de coordinación"
    );
    assert_eq!(dag.nodes.len(), 3, "El DAG debe contener 3 tareas");

    // 3. Verificar que cargo test pasa
    let test_res = ToolExecutor::execute_tool(&target, AgentRole::Auditor, "run_cargo_test", "{}");
    assert!(test_res.contains("EXITO"));

    // 4. Limpiar el sandbox para futuras pruebas
    ToolExecutor::reset_broken_service(&target).unwrap();
}
