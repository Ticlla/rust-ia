//! Suite de pruebas de integración para el Autonomous Agent & Multi-Agent Orchestration Framework.

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use proyecto_03_multi_agent::agent::{
    AgentContext, AgentRole, AgentState, DeterministicLlmBackend, ReActEngine,
};
use proyecto_03_multi_agent::api::{AppState, build_router};
use proyecto_03_multi_agent::create_default_system;
use proyecto_03_multi_agent::events::EventBus;
use proyecto_03_multi_agent::orchestrator::dag::{TaskGraph, TaskNode};
use proyecto_03_multi_agent::tools::{
    CalculatorTool, RagSearchTool, RustCodeValidatorTool, ToolRegistry,
};
use std::sync::Arc;
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// TEST 1: Registro de Herramientas y Tool Calling Tipado
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_tool_registry_and_execution() {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool));
    registry.register(Arc::new(RagSearchTool));
    registry.register(Arc::new(RustCodeValidatorTool));

    assert_eq!(registry.len(), 3);
    assert!(registry.get("calculator").is_some());
    assert!(registry.get("rag_search").is_some());
    assert!(registry.get("rust_code_validator").is_some());

    // 1. Ejecución de la calculadora (128 * 16)
    let calc_res = registry
        .execute(
            "calculator",
            serde_json::json!({ "expression": "128 * 16" }),
        )
        .await
        .expect("Calculator execution failed");
    assert!(calc_res.success);
    assert_eq!(calc_res.result["value"], 2048.0);

    // 2. Ejecución de búsqueda RAG
    let rag_res = registry
        .execute("rag_search", serde_json::json!({ "query": "SIMD AVX2" }))
        .await
        .expect("RAG search execution failed");
    assert!(rag_res.success);
    assert!(rag_res.result["retrieved_count"].as_u64().unwrap() >= 1);

    // 3. Ejecución del validador de código Rust
    let code_sample = "pub fn add(a: i32, b: i32) -> i32 { a + b }";
    let val_res = registry
        .execute(
            "rust_code_validator",
            serde_json::json!({ "code": code_sample }),
        )
        .await
        .expect("Validator execution failed");
    assert!(val_res.success);
    assert_eq!(val_res.result["valid_syntax"], true);
    assert_eq!(val_res.result["has_unsafe_blocks"], false);

    // 4. Exportación de esquema OpenAI
    let schema = registry.to_openai_tools_schema();
    assert!(schema.is_array());
    assert_eq!(schema.as_array().unwrap().len(), 3);
}

// ---------------------------------------------------------------------------
// TEST 2: Bucle ReAct con Autocorrección y Memoria
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_react_engine_execution_loop() {
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(RagSearchTool));
    tools.register(Arc::new(RustCodeValidatorTool));

    let llm = Arc::new(DeterministicLlmBackend);
    let event_bus = EventBus::new(64);
    let engine = ReActEngine::new(llm, tools).with_event_bus(event_bus);

    let mut context = AgentContext::new(
        AgentRole::Researcher,
        "Investigar arquitecturas de KV-Cache.",
    );
    context.add_message(proyecto_03_multi_agent::agent::AgentMessage::user(
        "Paged KV-Cache vLLM Architecture",
    ));

    let result = engine
        .run(&mut context)
        .await
        .expect("ReAct engine execution failed");

    assert_eq!(result.state, AgentState::Completed);
    assert!(result.total_steps >= 2); // 1 paso de Tool Call + 1 paso de Final Answer
    assert!(result.final_answer.contains("Reporte de Investigación"));
}

// ---------------------------------------------------------------------------
// TEST 3: Grafo de Dependencias DAG y Detección de Ciclos
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_task_graph_dag_topological_resolution() {
    let mut graph = TaskGraph::new();

    graph.add_task(TaskNode::new(
        "t1",
        AgentRole::Researcher,
        "Investigar",
        vec![],
    ));
    graph.add_task(TaskNode::new(
        "t2",
        AgentRole::Coder,
        "Codificar",
        vec!["t1".into()],
    ));
    graph.add_task(TaskNode::new(
        "t3",
        AgentRole::Reviewer,
        "Revisar",
        vec!["t2".into()],
    ));

    // Inicialmente solo t1 debe estar lista
    assert!(!graph.has_cycle());
    let ready_1 = graph.get_ready_tasks();
    assert_eq!(ready_1.len(), 1);
    assert_eq!(ready_1[0].id, "t1");

    // Marcar t1 como completada
    graph.mark_completed("t1", "Resultados de investigación".into());
    let ready_2 = graph.get_ready_tasks();
    assert_eq!(ready_2.len(), 1);
    assert_eq!(ready_2[0].id, "t2");

    // Marcar t2 como completada
    graph.mark_completed("t2", "Código Rust escrito".into());
    let ready_3 = graph.get_ready_tasks();
    assert_eq!(ready_3.len(), 1);
    assert_eq!(ready_3[0].id, "t3");

    // Marcar t3 como completada
    graph.mark_completed("t3", "Aprobado".into());
    assert!(graph.is_all_completed());

    // Probar detección de ciclos en un grafo inválido
    let mut cyclic_graph = TaskGraph::new();
    cyclic_graph.add_task(TaskNode::new(
        "cA",
        AgentRole::Coder,
        "A",
        vec!["cB".into()],
    ));
    cyclic_graph.add_task(TaskNode::new(
        "cB",
        AgentRole::Coder,
        "B",
        vec!["cA".into()],
    ));
    assert!(cyclic_graph.has_cycle());
}

// ---------------------------------------------------------------------------
// TEST 4: Coordinación Multi-Agente con Modelo de Actores
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_multi_agent_actor_coordination() {
    let (orchestrator, event_bus) = create_default_system();

    // Suscriptor para monitorear eventos emitidos durante el workflow
    let mut rx = event_bus.subscribe();

    let goal = "Construir un módulo de memoria paginada para atención en Rust.";
    let workflow_id = "wf-test-actor-001";

    let report = orchestrator
        .run_workflow(workflow_id, goal)
        .await
        .expect("Workflow execution failed");

    assert!(report.success);
    assert_eq!(report.workflow_id, workflow_id);
    assert!(report.research_summary.is_some());
    assert!(report.generated_code.is_some());
    assert!(report.review_feedback.is_some());

    // Verificar que el bus recibió eventos
    let first_event = rx.recv().await.expect("Failed to receive first event");
    match first_event {
        proyecto_03_multi_agent::events::AgentEvent::WorkflowStarted {
            workflow_id: wid, ..
        } => {
            assert_eq!(wid, workflow_id);
        }
        _ => {} // Evento capturado
    }
}

// ---------------------------------------------------------------------------
// TEST 5: API REST y Streaming SSE de Eventos en Axum
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_axum_api_and_sse_streaming() {
    let (orchestrator, event_bus) = create_default_system();
    let app_state = Arc::new(AppState {
        orchestrator,
        event_bus,
    });
    let app = build_router(app_state);

    // 1. Health check
    let req = Request::builder()
        .uri("/health")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let response = app.clone().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let health: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["active_actors"], 3);

    // 2. Run workflow POST
    let run_payload = serde_json::json!({
        "goal": "Auditar patrones de concurrencia y channels en Tokio",
        "workflow_id": "wf-api-test-01"
    });

    let run_req = Request::builder()
        .uri("/v1/orchestrator/run")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(run_payload.to_string()))
        .unwrap();

    let run_resp = app.clone().oneshot(run_req).await.unwrap();
    assert_eq!(run_resp.status(), StatusCode::OK);

    let run_body = to_bytes(run_resp.into_body(), usize::MAX).await.unwrap();
    let run_result: serde_json::Value = serde_json::from_slice(&run_body).unwrap();
    assert_eq!(run_result["workflow_id"], "wf-api-test-01");
    assert_eq!(run_result["status"], "completed");
    assert!(run_result["report"]["success"].as_bool().unwrap());
}

// ---------------------------------------------------------------------------
// TEST 6: Enrutador Semántico Vectorial (Semantic Router) Conversacional vs Técnico
// ---------------------------------------------------------------------------
#[tokio::test]
async fn test_intent_routing_conversational_vs_technical() {
    use proyecto_03_multi_agent::orchestrator::SupervisorOrchestrator;
    use proyecto_03_multi_agent::orchestrator::semantic_router::{SemanticRouter, UserIntent};

    // 1. Pruebas de clasificación conversacional y opiniones
    assert_eq!(
        SupervisorOrchestrator::classify_intent("hola"),
        UserIntent::Conversational
    );
    assert_eq!(
        SupervisorOrchestrator::classify_intent("hi"),
        UserIntent::Conversational
    );
    assert_eq!(
        SupervisorOrchestrator::classify_intent("¿Quién eres?"),
        UserIntent::Conversational
    );
    assert_eq!(
        SupervisorOrchestrator::classify_intent("how are you?"),
        UserIntent::Conversational
    );

    // CASO CLAVE: "¿Qué opinas de Rust?" (Contiene "rust", pero es una pregunta de opinión)
    assert_eq!(
        SupervisorOrchestrator::classify_intent("que opinas de rust"),
        UserIntent::Conversational
    );
    assert_eq!(
        SupervisorOrchestrator::classify_intent("qué opinas de rust"),
        UserIntent::Conversational
    );
    assert_eq!(
        SupervisorOrchestrator::classify_intent("no me des codigo solo dime que opinas de rust"),
        UserIntent::Conversational
    );

    // 2. Pruebas de clasificación de tarea técnica de ingeniería
    assert_eq!(
        SupervisorOrchestrator::classify_intent("Crear un RingBuffer en Rust"),
        UserIntent::TechnicalTask
    );
    assert_eq!(
        SupervisorOrchestrator::classify_intent("make me an hello world rust"),
        UserIntent::TechnicalTask
    );
    assert_eq!(
        SupervisorOrchestrator::classify_intent("módulo SIMD AVX2 de producto punto"),
        UserIntent::TechnicalTask
    );

    // 3. Verificación de similitud semántica y confianza del SemanticRouter
    let router = SemanticRouter::new();
    let res_opinion = router.classify("que opinas de rust");
    assert_eq!(res_opinion.intent, UserIntent::Conversational);
    assert!(res_opinion.conv_similarity > res_opinion.tech_similarity);

    let res_buffer = router.classify("diseñar un RingBuffer continuo en Rust");
    assert_eq!(res_buffer.intent, UserIntent::TechnicalTask);
    assert!(res_buffer.tech_similarity > res_buffer.conv_similarity);

    // 4. Ejecución de flujo conversacional de punta a punta para "¿qué opinas de rust?"
    let (orchestrator, _) = create_default_system();
    let report = orchestrator
        .run_workflow("wf-chat-opinion", "que opinas de rust")
        .await
        .unwrap();

    assert!(report.success);
    assert!(report.generated_code.is_none()); // No se forzó generación de código
    assert!(
        report.final_synthesis.contains("Borrow Checker")
            || report.final_synthesis.contains("Rust")
    );
}
