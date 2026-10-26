//! Grafo Acíclico Dirigido (DAG) para descomposición y ejecución concurrente de tareas.

use crate::agent::AgentRole;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Estado del ciclo de vida de una subtarea dentro del DAG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
}

/// Nodo individual en el grafo de tareas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskNode {
    pub id: String,
    pub role: AgentRole,
    pub description: String,
    pub dependencies: Vec<String>,
    pub status: TaskStatus,
    pub output: Option<String>,
}

impl TaskNode {
    pub fn new(
        id: impl Into<String>,
        role: AgentRole,
        description: impl Into<String>,
        dependencies: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            role,
            description: description.into(),
            dependencies,
            status: TaskStatus::Pending,
            output: None,
        }
    }
}

/// Motor de grafo de dependencias de tareas (DAG).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskGraph {
    pub tasks: HashMap<String, TaskNode>,
}

impl TaskGraph {
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
        }
    }

    /// Añade una tarea al grafo.
    pub fn add_task(&mut self, task: TaskNode) {
        self.tasks.insert(task.id.clone(), task);
    }

    /// Retorna las tareas que están listas para ejecutarse (estado Pending y todas sus dependencias en Completed).
    pub fn get_ready_tasks(&self) -> Vec<TaskNode> {
        self.tasks
            .values()
            .filter(|task| {
                if task.status != TaskStatus::Pending {
                    return false;
                }
                task.dependencies.iter().all(|dep_id| {
                    self.tasks
                        .get(dep_id)
                        .map(|dep| dep.status == TaskStatus::Completed)
                        .unwrap_or(false)
                })
            })
            .cloned()
            .collect()
    }

    /// Marca una tarea como en ejecución.
    pub fn mark_running(&mut self, task_id: &str) {
        if let Some(task) = self.tasks.get_mut(task_id) {
            task.status = TaskStatus::Running;
        }
    }

    /// Marca una tarea como completada y guarda su resultado textual.
    pub fn mark_completed(&mut self, task_id: &str, output: String) {
        if let Some(task) = self.tasks.get_mut(task_id) {
            task.status = TaskStatus::Completed;
            task.output = Some(output);
        }
    }

    /// Marca una tarea como fallida.
    pub fn mark_failed(&mut self, task_id: &str, error: String) {
        if let Some(task) = self.tasks.get_mut(task_id) {
            task.status = TaskStatus::Failed(error);
        }
    }

    /// Recopila los resultados de todas las tareas previas de las que depende una tarea dada.
    pub fn get_dependency_outputs(&self, task_id: &str) -> Option<String> {
        let task = self.tasks.get(task_id)?;
        if task.dependencies.is_empty() {
            return None;
        }

        let mut context = String::new();
        for dep_id in &task.dependencies {
            if let Some(dep_task) = self.tasks.get(dep_id) {
                if let Some(out) = &dep_task.output {
                    context.push_str(&format!(
                        "--- Entregable de tarea '{}' (Rol: {}) ---\n{}\n\n",
                        dep_task.id, dep_task.role, out
                    ));
                }
            }
        }

        Some(context)
    }

    /// Verifica si todas las tareas del grafo han sido completadas con éxito.
    pub fn is_all_completed(&self) -> bool {
        !self.tasks.is_empty()
            && self
                .tasks
                .values()
                .all(|t| t.status == TaskStatus::Completed)
    }

    /// Verifica si alguna tarea falló.
    pub fn has_failures(&self) -> bool {
        self.tasks
            .values()
            .any(|t| matches!(t.status, TaskStatus::Failed(_)))
    }

    /// Valida si el grafo contiene ciclos mediante el algoritmo de ordenamiento topológico de Kahn.
    pub fn has_cycle(&self) -> bool {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();

        for (id, task) in &self.tasks {
            in_degree.entry(id.as_str()).or_insert(0);
            for dep in &task.dependencies {
                adj.entry(dep.as_str()).or_default().push(id.as_str());
                *in_degree.entry(id.as_str()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut visited_count = 0;

        while let Some(u) = queue.pop_front() {
            visited_count += 1;
            if let Some(neighbors) = adj.get(u) {
                for &v in neighbors {
                    if let Some(deg) = in_degree.get_mut(v) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(v);
                        }
                    }
                }
            }
        }

        visited_count != self.tasks.len()
    }
}
