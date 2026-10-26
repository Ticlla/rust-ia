# ⚡ Clase 07: Runtime Asíncrono con `Tokio` en IA

Bienvenido a la séptima clase de la **Fase 2**. En esta unidad nos adentraremos en el motor industrial que alimenta los servidores de producción en Rust: **Tokio**, aprendiendo a separar limpiamente el tráfico masivo de red (I/O) de la computación matricial intensiva en tensores (CPU).

---

## 📑 Contenido de la Clase

1. [¿Qué es un Runtime Asíncrono?](#1-qué-es-un-runtime-asíncrono)
2. [La Macro `#[tokio::main]`](#2-la-macro-tokiomain)
3. [Tareas Ligeras con `tokio::spawn`](#3-tareas-ligeras-con-tokiospawn)
4. [La Regla de Oro en IA: `tokio::task::spawn_blocking`](#4-la-regla-de-oro-en-ia-tokiotaskspawn_blocking)
5. [Canales Asíncronos `tokio::sync::mpsc`](#5-canales-asíncronos-tokiosyncmpsc)
6. [Ejecutables de la Clase](#6-ejecutables-de-la-clase)

---

## 1. ¿Qué es un Runtime Asíncrono?

Rust como lenguaje **no incluye un runtime asíncrono** en su biblioteca estándar. Te da la sintaxis (`async`, `await`, el trait `Future`), pero deja que la comunidad elija el motor.

**Tokio** es el runtime estándar por excelencia:
* Contiene un **Reactor I/O de alta velocidad** conectado directamente a `epoll` (Linux) o `kqueue` (macOS).
* Contiene un **Thread Pool multi-hilo con Work Stealing** para balancear tareas asíncronas entre los núcleos del CPU.
* Provee temporizadores asíncronos (`tokio::time::sleep`) y sincronización concurrente (`tokio::sync`).

---

## 2. La Macro `#[tokio::main]`

Reemplaza la necesidad de llamar manualmente a `block_on`:
```rust
#[tokio::main]
async fn main() {
    // ¡Tu punto de entrada ahora es 100% asíncrono!
    let res = mi_tarea().await;
}
```

---

## 3. La Regla de Oro en IA: `tokio::task::spawn_blocking`

En un servidor de IA:
* **Tareas I/O (Red/HTTP):** Son cooperativas, esperan paquetes con `.await`.
* **Cómputo Tensorial (Matriz $W \cdot x$, GELU, Tokenización):** Son CPU-Bound y no tienen `.await`.

Si ejecutas un bucle pesado de tensores dentro de un hilo de Tokio, **congelas las peticiones de los otros 10,000 usuarios asignados a ese hilo**.

```
                Arquitectura Híbrida en Servidores de IA
                ────────────────────────────────────────
                   [ Peticiones HTTP de Usuarios ]
                                  │
                                  ▼
                     [ Hilos Tokio (Async I/O) ]
                                  │
                  ¿Es cálculo pesado de tensores?
                                  │
               ┌──────────────────┴──────────────────┐
               │ SÍ                                  │ NO
               ▼                                     ▼
   [ tokio::task::spawn_blocking ]            Continúa en Tokio
  (Pool dedicado para CPU / Rayon)          (Rápido, no bloqueante)
```

---

## 4. Ejecutables de la Clase

```bash
# 1. Probar la teoría (I/O asíncrono + spawn_blocking + streaming de tokens):
cargo run --bin c07_teoria

# 2. Explorar errores comunes (bloqueo del reactor, Mutex a través de await):
cargo run --bin c07_errores

# 3. Reto práctico: Servidor de Inferencia Asíncrono con Colas:
cargo run --bin c07_ejercicios

# 4. Validar las pruebas unitarias asíncronas:
cargo test --bin c07_ejercicios
```
