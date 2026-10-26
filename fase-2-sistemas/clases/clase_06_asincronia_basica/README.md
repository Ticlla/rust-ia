# ⏳ Clase 06: Asincronía Básica (`async` / `await` y `Future`) en IA

Bienvenido a la sexta clase de la **Fase 2**. En esta unidad daremos el salto de la computación pesada en CPU (Rayon) al manejo eficiente de **red, disco y APIs de modelos de lenguaje (I/O-Bound)** mediante el modelo de asincronía de Rust.

---

## 📑 Contenido de la Clase

1. [I/O-Bound vs CPU-Bound](#1-io-bound-vs-cpu-bound)
2. [¿Qué es una `Future` en Rust? (Máquinas de Estados y Cero-Costo)](#2-qué-es-una-future-en-rust-máquinas-de-estados-y-cero-costo)
3. [La Naturaleza Perezosa (Lazy) de Rust vs Python/JS](#3-la-naturaleza-perezosa-lazy-de-rust-vs-pythonjs)
4. [Concurrencia Cooperativa con `futures::join!`](#4-concurrencia-cooperativa-con-futuresjoin)
5. [Ejecutables de la Clase](#5-ejecutables-de-la-clase)

---

## 1. I/O-Bound vs CPU-Bound

* **CPU-Bound (Usa Rayon):**  
  El procesador está al 100% calculando matemáticas matriciales, multiplicando tensores o aplicando GELU.
* **I/O-Bound (Usa Async):**  
  Tu programa envía una petición HTTP a una API de OpenAI, descarga un dataset de AWS S3 o espera una consulta a una base de datos vectorial. El CPU no tiene nada que calcular: **está esperando que lleguen los bytes por el cable de red**.

Si usas un hilo del sistema operativo para cada espera de red, desperdicias 2 MB de memoria por hilo. Con **Async**, un solo hilo puede gestionar **100,000 conexiones concurrentes** porque cada tarea pesa solo unos cientos de bytes.

---

## 2. ¿Qué es una `Future` en Rust?

En Rust, una `Future` es un trait que representa un valor que aún no está listo:

```rust
pub trait Future {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}
```

* **`Poll::Pending`**: Los bytes de la red aún no llegan. El runtime asíncrono pone esta tarea en pausa y aprovecha el hilo para avanzar otra tarea de otro usuario.
* **`Poll::Ready(valor)`**: La respuesta ya llegó. Se reanuda la función.

---

## 3. La Naturaleza Perezosa (Lazy) de Rust vs Python/JS

* **En JavaScript o Python (`asyncio`):**  
  Cuando llamas a `hacer_peticion()`, la promesa arranca a correr en segundo plano inmediatamente.
* **En Rust:**  
  Llamar a `hacer_peticion()` **NO hace absolutamente nada**. Solo crea un struct en memoria (la máquina de estados).  
  El código no avanza hasta que explícitamente le pones **`.await`** o lo ejecutas con un runtime (`block_on` o `tokio::spawn`).

---

## 4. Ejecutables de la Clase

```bash
# 1. Probar la teoría de ejecución Lazy y consultas concurrentes con join!:
cargo run --bin c06_teoria

# 2. Explorar errores comunes (olvidar el .await, thread::sleep vs async):
cargo run --bin c06_errores

# 3. Reto práctico: Enrutador asíncrono multi-proveedor de LLMs:
cargo run --bin c06_ejercicios

# 4. Validar las pruebas unitarias:
cargo test --bin c06_ejercicios
```
