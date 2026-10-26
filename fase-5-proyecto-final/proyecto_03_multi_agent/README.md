# Proyecto 03: Autonomous Agent & Multi-Agent Orchestration Framework

**Fase 5 — Ecosistema Production-Ready en Rust**

Un framework de orquestación de agentes autónomos y sistemas multi-agente desarrollado en **Rust puro**, implementando el **Modelo de Actores**, el patrón **ReAct** (Reasoning + Acting), grafos de tareas acíclicos dirigidos (**DAG**) y streaming reactivo de eventos mediante **Server-Sent Events (SSE)**.

---

## 🚀 Arquitectura y Primeros Principios

A diferencia de los frameworks tradicionales en Python (CrewAI, LangGraph, AutoGen) que sufren de bloqueos del GIL, contención en hilos y falta de tipos estrictos en tiempo de compilación, este framework aprovecha las garantías de Rust:

1. **Modelo de Actores Concurrente (`orchestrator/actor.rs`):**
   * Cada agente (`Supervisor`, `Researcher`, `Coder`, `Reviewer`) es un Actor asíncrono independiente en Tokio.
   * La comunicación es 100% mediante paso de mensajes en canales `mpsc` y `oneshot`, garantizando **cero condiciones de carrera** y **cero bloqueos globales con Mutex**.
2. **Ciclo ReAct Estricto (`agent/react.rs`):**
   * Autómata finito que alterna entre **Thought** (razonamiento), **Action** (invocación de herramientas tipadas), **Observation** (procesamiento de retorno) y **Reflection** (ajuste de contexto).
   * Límite estricto de iteraciones para prevenir bucles infinitos.
3. **Tool Calling Tipado y Sandbox (`tools/`):**
   * Trait asíncrono `Tool` con validación de esquemas JSON Schema.
   * Aislamiento por timeouts para evitar que llamadas bloqueantes cuelguen el runtime.
   * Exportación automática al formato estándar de OpenAI Tools (`type: "function"`).
4. **Grafo de Tareas Acíclico Dirigido (DAG) (`orchestrator/dag.rs`):**
   * Descomposición de metas complejas en tareas interdependientes.
   * Algoritmo de ordenamiento topológico de Kahn para detectar tareas listas para ejecutar en paralelo y validar ausencia de dependencias circulares.
5. **Bus Reactivo y Streaming SSE (`events/` y `api/`):**
   * Canal `broadcast` de alta concurrencia para emisión de eventos en tiempo real.
   * Endpoint Axum `/v1/orchestrator/events` para consumo instantáneo por interfaces de usuario o telemetría.

---

## 🛠️ Ejecución y Pruebas

### 1. Ejecutar Suite de Tests
```bash
cargo test -p proyecto_03_multi_agent
```

### 2. Ejecutar Demostración Interactiva por Terminal (CLI)
```bash
cargo run -p proyecto_03_multi_agent --bin cli_agent
```

Salida de la terminal:
```text
================================================================================
🤖 Autonomous Multi-Agent Orchestration Framework (ReAct + Actor Model)
================================================================================

🚀 [WORKFLOW INICIADO] ID: wf-demo-001
🎯 [META]: Diseñar e implementar una estructura de caché concurrente segura en Rust...

📌 [DAG TASK] task_1_research (Researcher) -> InProgress
💭 [Researcher] Razonamiento (Thought):
   Debo investigar las especificaciones arquitectónicas en la base de conocimiento RAG...
🛠️  [Researcher] Invocando Tool: `rag_search`
📋 [Researcher] Tool Completada: `rag_search` en 0ms (Éxito: true)

📌 [DAG TASK] task_2_code (Coder) -> InProgress
💭 [Coder] Razonamiento (Thought):
   He diseñado la estructura inicial en Rust puro...
🛠️  [Coder] Invocando Tool: `rust_code_validator`
📋 [Coder] Tool Completada: `rust_code_validator` en 0ms (Éxito: true)

📌 [DAG TASK] task_3_review (Reviewer) -> InProgress
💭 [Reviewer] Razonamiento (Thought):
   Audité el entregable: documentation=true, unsafe=false. Puntuación asignada: 0.98
⚖️  [CONSENSO & AUDITORÍA]
   Aprobado: true | Calidad: 0.98 | Veredicto: Código aprobado: Sintaxis limpia, cero unsafe blocks...

✅ [WORKFLOW COMPLETADO] Tiempo total: 52ms
```

### 3. Iniciar el Servidor REST y Streaming SSE
```bash
PORT=8082 cargo run -p proyecto_03_multi_agent
```

Endpoints disponibles:
* `GET http://localhost:8082/health`
* `POST http://localhost:8082/v1/orchestrator/run`
* `GET http://localhost:8082/v1/orchestrator/events` (Server-Sent Events)
