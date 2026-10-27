//! Motor de Grafo Acíclico Dirigido (DAG) para coordinación multi-agente.
//! Gestiona la descomposición de tareas, dependencias y ejecución topológica.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use crate::agents::AgentRole;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskNode {
    pub id: String,
    pub title: String,
    pub role: AgentRole,
    pub description: String,
    pub dependencies: Vec<String>,
    pub status: TaskStatus,
    pub output: Option<String>,
}

impl TaskNode {
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        role: AgentRole,
        description: impl Into<String>,
        dependencies: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            role,
            description: description.into(),
            dependencies,
            status: TaskStatus::Pending,
            output: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskDag {
    pub nodes: HashMap<String, TaskNode>,
}

impl TaskDag {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    pub fn add_task(&mut self, node: TaskNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    pub fn get_task(&self, id: &str) -> Option<&TaskNode> {
        self.nodes.get(id)
    }

    pub fn get_task_mut(&mut self, id: &str) -> Option<&mut TaskNode> {
        self.nodes.get_mut(id)
    }

    /// Resuelve el orden de ejecución en lotes (batches) topológicos mediante el algoritmo de Kahn.
    /// Las tareas en un mismo lote no dependen entre sí y pueden ejecutarse en paralelo.
    pub fn topological_batches(&self) -> Result<Vec<Vec<String>>, String> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();

        for id in self.nodes.keys() {
            in_degree.insert(id.clone(), 0);
            dependents.insert(id.clone(), Vec::new());
        }

        // Construir el grafo de dependencias
        for (id, node) in &self.nodes {
            for dep in &node.dependencies {
                if !self.nodes.contains_key(dep) {
                    return Err(format!("Dependencia inexistente '{}' referenciada por '{}'", dep, id));
                }
                dependents.get_mut(dep).unwrap().push(id.clone());
                *in_degree.get_mut(id).unwrap() += 1;
            }
        }

        // Cola con nodos de grado de entrada 0 (sin dependencias pendientes)
        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| id.clone())
            .collect();

        let mut batches = Vec::new();
        let mut visited = HashSet::new();

        while !queue.is_empty() {
            let mut current_batch = Vec::new();
            let batch_size = queue.len();

            for _ in 0..batch_size {
                let node_id = queue.pop_front().unwrap();
                visited.insert(node_id.clone());
                current_batch.push(node_id.clone());

                if let Some(children) = dependents.get(&node_id) {
                    for child in children {
                        let deg = in_degree.get_mut(child).unwrap();
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(child.clone());
                        }
                    }
                }
            }

            batches.push(current_batch);
        }

        if visited.len() != self.nodes.len() {
            return Err("Se detectó un ciclo en el grafo de dependencias (DAG inválido)".to_string());
        }

        Ok(batches)
    }

    /// Marca una tarea como completada y guarda su resultado
    pub fn complete_task(&mut self, id: &str, output: String) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.status = TaskStatus::Completed;
            node.output = Some(output);
        }
    }

    /// Marca una tarea como fallida
    pub fn fail_task(&mut self, id: &str, reason: String) {
        if let Some(node) = self.nodes.get_mut(id) {
            node.status = TaskStatus::Failed(reason);
        }
    }
}
