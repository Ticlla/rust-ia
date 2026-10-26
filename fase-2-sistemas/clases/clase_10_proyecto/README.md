# 🏆 Clase 10: Proyecto Final Integrador - Motor de Inferencia en Producción

¡Felicidades por llegar a la décima y última clase de la **Fase 2: Sistemas y Concurrencia**!

En este proyecto final hemos integrado de punta a punta los conocimientos adquiridos a lo largo de las 9 clases anteriores, construyendo un **servidor de inferencia de IA de grado industrial**.

---

## 📑 Arquitectura del Sistema

```
                            MOTOR DE INFERENCIA EN RUST
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ 1. ENTRADA HTTP ASÍNCRONA (Axum + Tokio)                                    │
 │    • GET  /health      -> Estado, versión y métricas.                       │
 │    • POST /predict     -> Inferencia sobre vectores de características.     │
 │    • POST /hot_reload  -> Actualización de pesos en caliente.               │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 2. DESACOPLO Y OFF-LOADING (tokio::task::spawn_blocking)                    │
 │    • Despacha el cálculo numérico al Thread Pool de CPU.                    │
 │    • Mantiene el Reactor de red libre para atender a miles de usuarios.     │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 3. CÓMPUTO TENSORIAL PARALELO (Rayon par_iter)                              │
 │    • Multiplica los vectores y pesos usando todos los núcleos del CPU.      │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 4. MEMORIA COMPARTIDA Y BAJO NIVEL (Arc<RwLock<TensorBuffer>> + RAII Drop)  │
 │    • Memoria física contigua asignada con alloc/dealloc.                    │
 │    • Lecturas concurrentes masivas sin colisiones de memoria.               │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 5. PIPELINE DATALOADER EN SEGUNDO PLANO (tokio::sync::mpsc)                 │
 │    • Ingesta asíncrona continua de lotes y control de flujo.                │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Ejecutables del Proyecto

```bash
# 1. Ejecutar la demostración automatizada completa:
cargo run --bin c10_proyecto

# 2. Iniciar el SERVIDOR EN VIVO permanente en http://127.0.0.1:3000:
cargo run --bin c10_servidor

# 3. Demostración interactiva de tráfico continuo + Hot-Reload en vivo:
cargo run --bin c10_demo_trafico

# 4. Generar tráfico continuo por script Bash (en otra terminal):
./generar_trafico.sh

# 5. Explorar el laboratorio de errores y antipatrones de arquitectura:
cargo run --bin c10_errores

# 6. Validar el banco completo de pruebas unitarias y de integración:
cargo test --bin c10_ejercicios
```
