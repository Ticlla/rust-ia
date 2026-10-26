# 🚀 Clase 08: Servidores de Inferencia de IA con `Axum`

Bienvenido a la octava clase de la **Fase 2**. En esta unidad construiremos un microservicio HTTP REST de producción para servir modelos de Inteligencia Artificial utilizando **Axum**, el framework web moderno desarrollado por el equipo de Tokio.

---

## 📑 Contenido de la Clase

1. [¿Qué es Axum y por qué supera a FastAPI?](#1-qué-es-axum-y-por-qué-supera-a-fastapi)
2. [Arquitectura de Axum: Router, Handlers y Extractores](#2-arquitectura-de-axum-router-handlers-y-extractores)
3. [El Patrón de Estado Compartido: `State(Arc<AppState>)`](#3-el-patrón-de-estado-compartido-statearcappstate)
4. [Serialización y Deserialización con `serde`](#4-serialización-y-deserialización-con-serde)
5. [Ejecutables de la Clase](#5-ejecutables-de-la-clase)

---

## 1. ¿Qué es Axum y por qué supera a FastAPI?

En Python, el estándar para servir modelos de IA es **FastAPI** (montado sobre Uvicorn/Starlette).
Aunque FastAPI tiene buena ergonomía:
* Está limitado por el **GIL** de Python.
* Para usar múltiples núcleos requiere duplicar procesos en memoria.
* Su rendimiento ronda las 5,000 a 10,000 peticiones por segundo.

**Axum** es el equivalente en Rust:
* Construido directamente sobre **Tokio** y el motor HTTP **Hyper**.
* Capaz de procesar **más de 100,000 peticiones por segundo** en una sola máquina.
* Tipado 100% seguro en tiempo de compilación (si un endpoint no coincide con el tipo de datos, el código no compila).

---

## 2. Arquitectura de Axum: Extractores con Type-Safety

En lugar de decoradores mágicos en tiempo de ejecución, Axum usa el sistema de tipos de Rust:

```rust
async fn predict_handler(
    State(state): State<Arc<AppState>>,   // 1. Extrae el estado global compartido (RwLock del modelo)
    Json(payload): Json<PredictRequest>, // 2. Deserializa el JSON del cuerpo HTTP
) -> Json<PredictResponse> {
    // Inferencia ultrarrápida
}
```

---

## 3. Ejecutables de la Clase

```bash
# 1. Probar el servidor HTTP con auto-test rápido:
cargo run --bin c08_teoria

# 2. Iniciar el SERVIDOR EN VIVO permanente en http://127.0.0.1:3000:
cargo run --bin c08_servidor

# 3. Explorar errores comunes (orden de extractores, desajuste de tipos):
cargo run --bin c08_errores

# 4. Reto práctico: Microservicio de Embeddings con actualización en caliente:
cargo run --bin c08_ejercicios

# 5. Validar las pruebas unitarias:
cargo test --bin c08_ejercicios
```
