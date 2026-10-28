# Nivel 5: Demonio Autónomo 24/7 y Auto-Sanación Proactiva (`nivel_5_autonomous_daemon`)

> **Módulo:** Fase 5 — `proyecto_05_agenticidad`  
> **Grado de Autonomía:** **99% (Nivel 5 / L5 — Always-On Background Daemon / Zero-Human-in-the-Loop)**  
> **Modelo Evaluado:** `nvidia/nemotron-3.5-bolt-e4b`  
> **Benchmark:** Microservicio roto con `error[E0308]: mismatched types` en [`sandbox/broken_service`](../sandbox/broken_service/)

---

## 1. ¿Qué es el Nivel 5?

El **Nivel 5** representa la **cúspide de los sistemas agénticos de IA**. 

En todos los niveles previos (L0 a L4), el sistema depende de la presencia de un operador humano que envíe un prompt o ejecute una orden. En el **Nivel 5**, el agente opera como un servicio en segundo plano continuo que **convive con el sistema en producción**:

* **Monitoreo Proactivo:** Evalúa la integridad del código y compilación mediante un `ServiceWatcher` concurrente con Tokio.
* **Auto-Despertar por Eventos:** Al detectar cualquier degradación, error de tipos o regresión en pruebas, genera un incidente interno de forma inmediata.
* **Auto-Sanación inadvertida:** Genera el parche de código con Nemotron, certifica con `cargo test` y registra el informe con el **Tiempo Medio de Recuperación (*MTTR*)**, todo en cuestión de segundos y sin intervención humana.

---

## 2. Comandos de Ejecución

### A. Ejecución en Vivo con la API de NVIDIA Nemotron

```bash
NVIDIA_API_KEY="tu_api_key_aqui" cargo run --manifest-path fase-5-proyecto-final/Cargo.toml --release -p nivel_5_autonomous_daemon
```

### B. Modo Simulación Determinista (Offline)

```bash
cargo run --manifest-path fase-5-proyecto-final/Cargo.toml --release -p nivel_5_autonomous_daemon
```

### C. Pruebas de Integración Automatizadas

```bash
cargo test --manifest-path fase-5-proyecto-final/Cargo.toml -p nivel_5_autonomous_daemon
```

---

## 3. Telemetría Real de Incidente

```text
================================================================================
  🔬 EVALUACIÓN DE AGENTICIDAD: NIVEL 5 — DEMONIO AUTÓNOMO 24/7 & AUTO-SANACIÓN
================================================================================
  • Microservicio Monitorizado: "sandbox/broken_service"
  • Modelo de Operación:        Always-On Background Daemon (tokio::select!)
  • Nivel de Intervención:      ZERO HUMAN IN THE LOOP (Auto-activación proactiva)
  • Latencia de Detección:      Heartbeats cada 200 ms
--------------------------------------------------------------------------------
  ⚡ [EVENTO EXTERNO]: Se ha inyectado un fallo 'E0308' en 'src/main.rs'.
  👀 El humano NO enviará ninguna orden. Observa cómo el Demonio lo resuelve solo.

  🔄 Iniciando Demonio Autónomo en Segundo Plano (Nivel 5)...

  🚨 [ALERTA PROACTIVA] Se detectó degradación en el servicio (INC-2026-001).
     Causa: error[E0308]: mismatched types
     ⚡ Despertando pipeline de auto-sanación autónoma...
     ✅ [AUTO-SANADO] Incidente INC-2026-001 resuelto en 5262 ms (MTTR).
     🧪 Verificación de Pruebas: PASADO (100%)

  🏆 DIAGNÓSTICO DE AGENTICIDAD: NIVEL 5
  • ¿Requirió prompt de un humano?:   NO (Auto-activación por evento)
  • Tiempo de intervención humana:    0.0 segundos
  • Grado de Autonomía Real:          99.0% (Zero-Human-in-the-Loop)
```

---

## 4. Estructura de Código

* [`src/watcher.rs`](src/watcher.rs): Monitor de salud periódico con `cargo check`.
* [`src/healer.rs`](src/healer.rs): Pipeline de auto-sanación y métricas MTTR.
* [`src/daemon.rs`](src/daemon.rs): Bucle con `tokio::select!` y apagado seguro (*graceful shutdown*).
* [`src/main.rs`](src/main.rs): CLI demostrativo con inyección de caos y auto-sanación.
* [`tests/integration_test.rs`](tests/integration_test.rs): Pruebas automatizadas de detección y sanación.
* [`ARCHITECTURE.md`](ARCHITECTURE.md): Documento formal de arquitectura y telemetría.
