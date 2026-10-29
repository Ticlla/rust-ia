use nivel_3_react_loop::ReActAgent;
use nivel_3_react_loop::tools::ToolExecutor;
use std::path::Path;

#[tokio::test]
async fn test_react_loop_resolves_broken_service_autonomously() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = manifest_dir
        .parent()
        .unwrap()
        .join("sandbox/broken_service");

    // 1. Restablecer el sandbox
    ToolExecutor::reset_broken_service(&target).unwrap();

    // 2. Ejecutar agente ReAct (en modo simulado o real)
    let agent = ReActAgent::new(None, 6);
    let (success, traces) = agent
        .run_react_loop(&target, "Corrige broken_service y corre cargo test")
        .await;

    // 3. Debe iterar múltiples pasos
    assert!(
        traces.len() >= 2,
        "El bucle ReAct debe ejecutar al menos 2 pasos"
    );
    assert!(
        success,
        "El agente ReAct debe lograr que el sandbox compile y pase tests"
    );

    // 4. Verificar que cargo test pasa de verdad
    let test_res = ToolExecutor::execute_tool(&target, "run_cargo_test", "{}");
    assert!(test_res.contains("EXITO"));

    // 5. Restablecer para futuras pruebas
    ToolExecutor::reset_broken_service(&target).unwrap();
}
