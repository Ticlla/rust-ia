# Nivel 4: Coordinación Multi-Agente con DAGs de Tareas (`nivel_4_multi_agent`)

> **Módulo:** Fase 5 — `proyecto_05_agenticidad`  
> **Grado de Autonomía:** **85% (Nivel 4 / L4 — Sistema Multi-Agente con Separación de Roles y RBAC)**  
> **Modelo Evaluado:** `nvidia/nemotron-3.5-bolt-e4b`  
> **Benchmark:** Microservicio roto con `error[E0308]: mismatched types` en [`sandbox/broken_service`](../sandbox/broken_service/)

---

## 1. ¿Qué es el Nivel 4?

El **Nivel 4** introduce **Especialización de Roles**, **Separación Estricta de Deberes (Principle of Least Privilege / RBAC)** y **Orquestación mediante un Grafo Acíclico Dirigido (Task DAG)**.

A diferencia del Nivel 3 (donde un solo agente monolítico lee, codifica, compila y aprueba sus propias pruebas), en el Nivel 4:
* **`SupervisorOrchestrator`**: Descompone la meta en un grafo topológico causal de tareas.
* **`AnalyzerAgent` (Solo Lectura)**: Diagnostica el error con `cargo check` y emite un informe estructurado. Tiene prohibido escribir en disco.
* **`CoderAgent` (Desarrollo)**: Escribe el parche de código usando `write_file`. Tiene prohibido ejecutar pruebas para evitar que falsee aserciones.
* **`AuditorQAAgent` (Calidad Independiente)**: Ejecuta `cargo test` y valida que el código cumpla los estándares sin hacer trampa. Tiene prohibido escribir código.

---

## 2. Comandos de Ejecución

### A. Ejecución en Vivo con la API de NVIDIA Nemotron

```bash
NVIDIA_API_KEY="tu_api_key_aqui" cargo run --manifest-path fase-5-proyecto-final/Cargo.toml --release -p nivel_4_multi_agent
```

### B. Modo Simulación Determinista (Offline)

```bash
cargo run --manifest-path fase-5-proyecto-final/Cargo.toml --release -p nivel_4_multi_agent
```

### C. Ejecutar las Pruebas de Integración y Seguridad RBAC

```bash
cargo test --manifest-path fase-5-proyecto-final/Cargo.toml -p nivel_4_multi_agent
```

---

## 3. Matriz de Control de Acceso (RBAC)

| Agente | Rol | Herramientas Permitidas | Restricción de Seguridad |
| :--- | :--- | :--- | :--- |
| `SupervisorOrchestrator` | Orquestador | Planificación DAG | No toca el disco |
| `AnalyzerAgent` | Diagnóstico | `read_file`, `run_cargo_check` | ⛔ Sin permiso de escritura (`write_file`) |
| `CoderAgent` | Desarrollo | `read_file`, `write_file` | ⛔ Sin permiso de pruebas (`run_cargo_test`) |
| `AuditorQAAgent` | Auditoría | `read_file`, `run_cargo_test` | ⛔ Sin permiso de escritura (`write_file`) |

---

## 4. Traza Real de Ejecución Multi-Agente

```text
================================================================================
  🔬 EVALUACIÓN DE AGENTICIDAD: NIVEL 4 — MULTI-AGENTE & TASK DAGs
================================================================================
  • Desafío: Reparar y auditar "sandbox/broken_service"
  • Modelo de Coordinación:                Grafo Acíclico Dirigido (Task DAG)
  • Principio de Seguridad:                Mínimo Privilegio (RBAC estricto)
  • Roles Especializados:                  Supervisor, Analyzer, Coder, Auditor
--------------------------------------------------------------------------------
  [Estado Inicial] ¿El microservicio compila?: NO (Roto con E0308)

  🛡️  VERIFICACIÓN DE SEGURIDAD (RBAC):
  [Analyzer intentando escribir]: ⛔ VIOLACIÓN DE SEGURIDAD (RBAC): El rol Analyzer no tiene permiso para invocar la herramienta 'write_file'.
  [Coder intentando auto-aprobarse]: ⛔ VIOLACIÓN DE SEGURIDAD (RBAC): El rol Coder no tiene permiso para invocar la herramienta 'run_cargo_test'.

  🔄 Iniciando Orquestación Multi-Agente mediante Task DAG (Nivel 4)...

  ┌─ [1_diagnose] 👔 SupervisorOrchestrator ── Delegando a "AnalyzerAgent"
  ┌─ [1_diagnose] 🔍 AnalyzerAgent ───────── read_file & run_cargo_check (Detecta E0308)
  ┌─ [2_code_patch] 👔 SupervisorOrchestrator ── Delegando a "CoderAgent"
  ┌─ [2_code_patch] 💻 CoderAgent ────────── write_file (Aplica formato con 2 decimales)
  ┌─ [3_audit_test] 👔 SupervisorOrchestrator ── Delegando a "AuditorQAAgent"
  ┌─ [3_audit_test] 🛡️ AuditorQAAgent ────── run_cargo_test (Pruebas unitarias pasan 100%)

  🧪 VERIFICACIÓN DEL ENTORNO FÍSICO TRAS LA AUDITORÍA:
  ✅ ¡MISIÓN CUMPLIDA AL 100% POR EL EQUIPO MULTI-AGENTE!
```

---

## 5. Estructura de Código

* [`src/dag.rs`](src/dag.rs): Algoritmo de orden topológico de Kahn y detección de ciclos.
* [`src/agents.rs`](src/agents.rs): Definición de roles y prompts ultra-especializados.
* [`src/tools.rs`](src/tools.rs): Ejecutor de herramientas con RBAC en tiempo de ejecución.
* [`src/lib.rs`](src/lib.rs): Orquestador [`MultiAgentOrchestrator`](src/lib.rs) con gestión de mensajes y re-auditoría.
* [`src/main.rs`](src/main.rs): Punto de entrada CLI.
* [`tests/integration_test.rs`](tests/integration_test.rs): Pruebas automatizadas de topología, RBAC y resolución.
