//! Nivel 4 de Agenticidad: Coordinación Multi-Agente con DAGs y Separación de Roles.
//! Demuestra cómo la especialización y el principio de mínimo privilegio evitan la
//! degradación de contexto y el sesgo de auto-aprobación del agente monolítico.

pub mod agents;
pub mod dag;
pub mod tools;

use serde_json::Value;
use std::path::Path;
use std::process::Command;

use crate::agents::{AgentMessage, AgentRole};
use crate::dag::{TaskDag, TaskNode};
use crate::tools::{ToolCall, ToolExecutor};

#[derive(Debug, Clone)]
pub struct OrchestrationEvent {
    pub task_id: String,
    pub role: AgentRole,
    pub action: String,
    pub details: String,
}

pub struct MultiAgentOrchestrator {
    pub model: String,
    pub api_key: Option<String>,
}

impl Default for MultiAgentOrchestrator {
    fn default() -> Self {
        Self {
            model: std::env::var("NVIDIA_MODEL")
                .unwrap_or_else(|_| "nvidia/nemotron-3.5-bolt-e4b".to_string()),
            api_key: std::env::var("NVIDIA_API_KEY").ok(),
        }
    }
}

impl MultiAgentOrchestrator {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            ..Default::default()
        }
    }

    /// Construye el Task DAG estándar para el ciclo de ingeniería de software
    pub fn build_standard_dag(&self) -> TaskDag {
        let mut dag = TaskDag::new();

        dag.add_task(TaskNode::new(
            "1_diagnose",
            "Diagnóstico de Error y Causa Raíz",
            AgentRole::Analyzer,
            "Inspecciona el código en 'src/main.rs' y corre 'run_cargo_check' para identificar el fallo.",
            vec![],
        ));

        dag.add_task(TaskNode::new(
            "2_code_patch",
            "Implementación del Parche de Código",
            AgentRole::Coder,
            "Escribe la solución correcta con 'write_file' respetando los requisitos de tipo y formato.",
            vec!["1_diagnose".to_string()],
        ));

        dag.add_task(TaskNode::new(
            "3_audit_test",
            "Auditoría y Certificación de Pruebas",
            AgentRole::Auditor,
            "Ejecuta 'run_cargo_test' y verifica que las pruebas no hayan sido modificadas ni falseadas.",
            vec!["2_code_patch".to_string()],
        ));

        dag
    }

    /// Ejecuta la orquestación multi-agente a través del DAG topológico
    pub async fn run_orchestration(
        &self,
        sandbox_dir: &Path,
        _goal: &str,
    ) -> (bool, Vec<OrchestrationEvent>, TaskDag) {
        let mut dag = self.build_standard_dag();
        let mut events = Vec::new();

        let batches = match dag.topological_batches() {
            Ok(b) => b,
            Err(e) => {
                events.push(OrchestrationEvent {
                    task_id: "DAG_ERROR".to_string(),
                    role: AgentRole::Supervisor,
                    action: "Fallo de Topología".to_string(),
                    details: e,
                });
                return (false, events, dag);
            }
        };

        let mut messages_log: Vec<AgentMessage> = Vec::new();

        for batch in batches {
            for task_id in batch {
                let (role, title, description, deps) = {
                    let task = dag.get_task(&task_id).unwrap();
                    (
                        task.role,
                        task.title.clone(),
                        task.description.clone(),
                        task.dependencies.clone(),
                    )
                };

                events.push(OrchestrationEvent {
                    task_id: task_id.clone(),
                    role: AgentRole::Supervisor,
                    action: "Delegando Tarea".to_string(),
                    details: format!("Asignada tarea '{}' a {:?}", title, role.name()),
                });

                // Recopilar contexto de las tareas predecesoras
                let mut prior_context = String::new();
                for dep in &deps {
                    if let Some(dep_task) = dag.get_task(dep) {
                        if let Some(out) = &dep_task.output {
                            prior_context.push_str(&format!("\n[Salida de {}]: {}\n", dep, out));
                        }
                    }
                }

                // Ejecutar el agente asignado
                let task_output = self
                    .execute_agent_task(
                        sandbox_dir,
                        role,
                        &task_id,
                        &description,
                        &prior_context,
                        &mut events,
                    )
                    .await;

                dag.complete_task(&task_id, task_output.clone());

                messages_log.push(AgentMessage {
                    from: role,
                    to: AgentRole::Supervisor,
                    task_id: task_id.clone(),
                    payload: task_output,
                });
            }
        }

        // Verificación objetiva final en el disco y Bucle de Re-auditoría
        let mut retry_count = 0;
        let max_retries = 2;

        while retry_count < max_retries {
            let test_ok = Command::new("cargo")
                .arg("test")
                .current_dir(sandbox_dir)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);

            let auditor_passed = dag
                .get_task("3_audit_test")
                .and_then(|t| t.output.as_ref())
                .map(|out| out.contains("APROBADO") || out.contains("EXITO"))
                .unwrap_or(false);

            if test_ok && auditor_passed {
                return (true, events, dag);
            }

            retry_count += 1;
            let failure_feedback = dag
                .get_task("3_audit_test")
                .and_then(|t| t.output.as_ref())
                .cloned()
                .unwrap_or_else(|| "Fallo al ejecutar las pruebas".to_string());

            events.push(OrchestrationEvent {
                task_id: format!("retry_{retry_count}"),
                role: AgentRole::Supervisor,
                action: "Rechazo del Auditor: Solicitando Corrección".to_string(),
                details: format!(
                    "El Auditor rechazó el código. Supervisor reasigna corrección a CoderAgent (Intento {}/{}). Razón: {}",
                    retry_count, max_retries, failure_feedback.lines().next().unwrap_or("")
                ),
            });

            // Re-ejecutar CoderAgent con el feedback explícito del Auditor
            let coder_retry_output = self.execute_agent_task(
                sandbox_dir,
                AgentRole::Coder,
                &format!("2_code_patch_retry_{retry_count}"),
                "El Auditor reportó fallos. Sobreescribe 'src/main.rs' con código Rust válido donde calculate_tax retorne format!(\"{:.2}\", amount * 0.18).",
                &format!("Feedback del Auditor:\n{}", failure_feedback),
                &mut events,
            ).await;

            dag.complete_task("2_code_patch", coder_retry_output);

            // Re-ejecutar AuditorAgent para re-evaluación
            let audit_retry_output = self
                .execute_agent_task(
                    sandbox_dir,
                    AgentRole::Auditor,
                    &format!("3_audit_test_retry_{retry_count}"),
                    "Re-audita el código corregido ejecutando 'run_cargo_test'.",
                    "",
                    &mut events,
                )
                .await;

            dag.complete_task("3_audit_test", audit_retry_output);
        }

        let test_ok = Command::new("cargo")
            .arg("test")
            .current_dir(sandbox_dir)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        let auditor_passed = dag
            .get_task("3_audit_test")
            .and_then(|t| t.output.as_ref())
            .map(|out| out.contains("APROBADO") || out.contains("EXITO"))
            .unwrap_or(false);

        (test_ok && auditor_passed, events, dag)
    }

    /// Ejecuta una tarea individual llamando a Nemotron con el rol y herramientas específicas
    async fn execute_agent_task(
        &self,
        sandbox_dir: &Path,
        role: AgentRole,
        task_id: &str,
        task_description: &str,
        prior_context: &str,
        events: &mut Vec<OrchestrationEvent>,
    ) -> String {
        if let Some(key) = &self.api_key {
            let tools_schema = ToolExecutor::get_tools_schema_for_role(role);
            let user_content = format!(
                "Tarea asignada: {}\n{}\nPor favor ejecuta las acciones necesarias usando tus herramientas autorizadas.",
                task_description, prior_context
            );

            let mut messages = vec![
                serde_json::json!({"role": "system", "content": role.system_prompt()}),
                serde_json::json!({"role": "user", "content": user_content}),
            ];

            let mut final_output = String::new();
            let mut step = 0;
            let max_agent_steps = 3;

            while step < max_agent_steps {
                step += 1;
                let client = reqwest::Client::new();
                let body = serde_json::json!({
                    "model": self.model,
                    "messages": messages,
                    "tools": tools_schema,
                    "tool_choice": "auto",
                    "temperature": 0.2
                });

                let res = client
                    .post("https://integrate.api.nvidia.com/v1/chat/completions")
                    .header("Authorization", format!("Bearer {key}"))
                    .header("Content-Type", "application/json")
                    .json(&body)
                    .send()
                    .await;

                if let Ok(resp) = res {
                    if resp.status().is_success() {
                        if let Ok(json) = resp.json::<Value>().await {
                            let choice = &json["choices"][0]["message"];
                            let content = choice["content"].as_str().unwrap_or("").to_string();

                            if let Some(tool_calls_val) = choice.get("tool_calls") {
                                if let Ok(calls) =
                                    serde_json::from_value::<Vec<ToolCall>>(tool_calls_val.clone())
                                {
                                    if let Some(call) = calls.first() {
                                        let tool_name = &call.function.name;
                                        let args = &call.function.arguments;

                                        let observation = ToolExecutor::execute_tool(
                                            sandbox_dir,
                                            role,
                                            tool_name,
                                            args,
                                        );

                                        events.push(OrchestrationEvent {
                                            task_id: task_id.to_string(),
                                            role,
                                            action: format!("{tool_name}({args})"),
                                            details: observation.clone(),
                                        });

                                        messages.push(choice.clone());
                                        messages.push(serde_json::json!({
                                            "role": "tool",
                                            "tool_call_id": call.id,
                                            "content": observation.clone()
                                        }));

                                        final_output = format!(
                                            "{content}\n[Observación de {tool_name}]: {observation}"
                                        );

                                        // Si el rol ya ejecutó su acción terminal
                                        if role == AgentRole::Coder
                                            && tool_name == "write_file"
                                            && !observation.contains("Error")
                                        {
                                            return final_output;
                                        }
                                        if role == AgentRole::Auditor
                                            && tool_name == "run_cargo_test"
                                        {
                                            return final_output;
                                        }

                                        continue;
                                    }
                                }
                            }

                            events.push(OrchestrationEvent {
                                task_id: task_id.to_string(),
                                role,
                                action: "Respuesta Generada".to_string(),
                                details: content.clone(),
                            });

                            return content;
                        }
                    }
                }
            }

            if !final_output.is_empty() {
                return final_output;
            }
        }

        // Simulación determinista offline para cada rol
        match role {
            AgentRole::Analyzer => {
                let obs_check =
                    ToolExecutor::execute_tool(sandbox_dir, role, "run_cargo_check", "{}");
                let report = format!(
                    "Diagnóstico de AnalyzerAgent:\n\
                    - Fallo: error[E0308]: mismatched types (expected String, found f64)\n\
                    - Ubicación: 'src/main.rs' en calculate_tax\n\
                    - Causa: Retorna 'amount * 0.18' directamente en lugar de formatear a String.\n\
                    - Prescripción para CoderAgent: Usar 'format!(\"{{:.2}}\", amount * 0.18)'."
                );

                events.push(OrchestrationEvent {
                    task_id: task_id.to_string(),
                    role,
                    action: "run_cargo_check()".to_string(),
                    details: obs_check,
                });

                report
            }
            AgentRole::Coder => {
                let fixed_code = r#"//! Microservicio con error intencional para evaluar grados de agenticidad.

pub fn calculate_tax(amount: f64) -> String {
    // Corregido por CoderAgent: Retornar valor formateado con dos decimales
    format!("{:.2}", amount * 0.18)
}

fn main() {
    let tax = calculate_tax(100.0);
    println!("Impuesto: {}", tax);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_tax() {
        assert_eq!(calculate_tax(100.0), "18.00");
    }
}
"#;
                let args = serde_json::json!({
                    "relative_path": "src/main.rs",
                    "content": fixed_code
                })
                .to_string();

                let obs_write = ToolExecutor::execute_tool(sandbox_dir, role, "write_file", &args);

                events.push(OrchestrationEvent {
                    task_id: task_id.to_string(),
                    role,
                    action: "write_file('src/main.rs')".to_string(),
                    details: obs_write,
                });

                "Parche de código aplicado exitosamente en 'src/main.rs'.".to_string()
            }
            AgentRole::Auditor => {
                let obs_test =
                    ToolExecutor::execute_tool(sandbox_dir, role, "run_cargo_test", "{}");

                events.push(OrchestrationEvent {
                    task_id: task_id.to_string(),
                    role,
                    action: "run_cargo_test()".to_string(),
                    details: obs_test.clone(),
                });

                if obs_test.contains("EXITO") {
                    "APROBADO: Todas las pruebas unitarias pasaron y el código cumple los estándares.".to_string()
                } else {
                    "RECHAZADO: Las pruebas fallaron.".to_string()
                }
            }
            AgentRole::Supervisor => "Planificación completada".to_string(),
        }
    }
}
