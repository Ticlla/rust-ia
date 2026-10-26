# 📖 Clase 04: Lecturas Concurrentes con `RwLock<T>` en Inferencia de IA

Bienvenido a la cuarta clase de la **Fase 2**. En esta unidad abordaremos la arquitectura óptima para **servidores de inferencia y bases de datos vectoriales**: permitir que cientos o miles de peticiones lean el modelo en paralelo sin bloquearse mutuamente, garantizando exclusividad únicamente cuando se actualizan los pesos.

---

## 📑 Contenido de la Clase

1. [El Cuello de Botella del Mutex en Inferencia](#1-el-cuello-de-botella-del-mutex-en-inferencia)
2. [¿Qué es `RwLock<T>`? (Read-Write Lock)](#2-qué-es-rwlockt-read-write-lock)
3. [Comparación: Mutex vs RwLock](#3-comparación-mutex-vs-rwlock)
4. [El Patrón en IA: Actualización en Caliente (Hot-Reload)](#4-el-patrón-en-ia-actualización-en-caliente-hot-reload)
5. [Ejecutables de la Clase](#5-ejecutables-de-la-clase)

---

## 1. El Cuello de Botella del Mutex en Inferencia

En un servidor de producción (como los que sirven modelos LLM o embeddings):
* El **99.9% del tráfico son lecturas** (hacer inferencia: calcular $y = W \cdot x$).
* Solo el **0.1% son escrituras** (recargar nuevos pesos tras un fine-tuning o actualizar embeddings).

Si usas `Mutex<T>`:
```
Petición 1 [Lectura] ──> [ 🔒 Mutex ] (Bloquea a todos)
Petición 2 [Lectura] ─────────────> Espera en cola... ⏳
Petición 3 [Lectura] ───────────────────────────> Espera en cola... ⏳
```
Aunque las peticiones 1, 2 y 3 solo quieren **leer los mismos pesos**, `Mutex` las obliga a procesarse una por una en fila india, desperdiciando los núcleos de tu CPU o GPU.

---

## 2. ¿Qué es `RwLock<T>`? (Read-Write Lock)

`RwLock` implementa la semántica: **Muchos Lectores O Un Solo Escritor**:

* **Múltiples Lectores (`lock.read()`)**: 
  * Cientos de hilos pueden adquirir el cerrojo de lectura simultáneamente.
  * Ningún lector bloquea a otro lector.
* **Un Escritor Exclusivo (`lock.write()`)**:
  * Solo un hilo puede escribir.
  * Mientras haya un escritor, todos los lectores y demás escritores se bloquean.

```
Petición 1 [Lectura] ──┐
Petición 2 [Lectura] ──┼──> [ 🟢 RwLock::read ] (¡Todos ejecutan a la vez!)
Petición 3 [Lectura] ──┘
```

---

## 3. Comparación: Mutex vs RwLock

| Característica | `Mutex<T>` | `RwLock<T>` |
|---|---|---|
| **Lectores simultáneos** | 1 (exclusivo) | $\infty$ (concurrente) |
| **Escritor** | 1 (exclusivo) | 1 (exclusivo) |
| **Costo de CPU por Lock** | Mínimo | Ligeramente mayor |
| **Ideal para** | Entrenamiento continuo (muchos gradientes escribiendo) | Servidores de Inferencia, Caché de Embeddings, Feature Stores |

---

## 4. El Patrón en IA: Actualización en Caliente (Hot-Reload)

```rust
use std::sync::{Arc, RwLock};

// 1. Instanciamos el modelo protegido:
let modelo = Arc::new(RwLock::new(ModeloIA::cargar_pesos("v1.bin")));

// 2. Clientes de inferencia (concurrencia total):
let modelo_lector = Arc::clone(&modelo);
thread::spawn(move || {
    let m = modelo_lector.read().unwrap();
    m.predecir(&entrada); // No bloquea a otros clientes!
});

// 3. Worker de fondo (recarga de modelo sin reiniciar el servidor):
let modelo_escritor = Arc::clone(&modelo);
thread::spawn(move || {
    let mut m = modelo_escritor.write().unwrap();
    m.recargar_pesos("v2_fine_tuned.bin"); // Pausa temporal solo durante la escritura
});
```

---

## 5. Ejecutables de la Clase

```bash
# 1. Probar la teoría (clientes de inferencia en paralelo + actualización en caliente):
cargo run --bin c04_teoria

# 2. Explorar los errores comunes (Deadlock de upgrading, starving):
cargo run --bin c04_errores

# 3. Correr el reto de la Caché de Embeddings Concurrente:
cargo run --bin c04_ejercicios

# 4. Validar las pruebas unitarias:
cargo test --bin c04_ejercicios
```
