use nivel_2_tool_caller::ToolCallerAgent;
use nivel_2_tool_caller::tools::ToolExecutor;
use std::path::Path;

#[tokio::test]
async fn test_tool_calling_modifies_disk_and_can_be_reset() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = manifest_dir
        .parent()
        .unwrap()
        .join("sandbox/broken_service");

    // 1. Verificar esquema de herramientas
    let schemas = ToolExecutor::get_tools_schema();
    assert!(schemas.is_array());
    assert_eq!(schemas.as_array().unwrap().len(), 2);

    // 2. Restablecer estado roto
    ToolExecutor::reset_broken_service(&target).unwrap();
    let (broken, _) = ToolCallerAgent::verify_sandbox(&target);
    assert!(!broken, "Debe iniciar roto con E0308");

    // 3. Ejecutar acción de escritura
    let agent = ToolCallerAgent::new(None);
    let (summary, res) = agent
        .execute_single_step(&target, "Corrige calculate_tax")
        .await
        .unwrap();
    assert!(summary.contains("write_file"));
    assert!(res.is_some());

    // 4. Verificar que el entorno físico cambió y ahora compila
    let (fixed, _) = ToolCallerAgent::verify_sandbox(&target);
    assert!(fixed, "Tras write_file debe compilar");

    let (tests_pass, _) = ToolCallerAgent::run_sandbox_tests(&target);
    assert!(tests_pass, "Las pruebas deben pasar");

    // 5. Restablecer a roto para futuras pruebas
    ToolExecutor::reset_broken_service(&target).unwrap();
}
