# ⚡ Clase 01: Hilos Nativos (Threads), `join()` y `move` en IA

Bienvenido a la primera clase de la **Fase 2: Sistemas y Concurrencia**. En esta unidad aprenderás a romper las limitaciones de Python para ejecutar cómputo de IA en paralelo real a través de los múltiples núcleos de tu CPU.

---

## 📑 Contenido de la Clase

1. [El Mito de los Hilos en Python vs Rust](#1-el-mito-de-los-hilos-en-python-vs-rust)
2. [Lanzar un Hilo con `thread::spawn`](#2-lanzar-un-hilo-con-threadspawn)
3. [Esperar y Recibir Resultados con `handle.join()`](#3-esperar-y-recibir-resultados-con-handlejoin)
4. [La Obligación del Closure `move`](#4-la-obligación-del-closure-move)
5. [Ejecutables de la Clase](#5-ejecutables-de-la-clase)

---

## 1. El Mito de los Hilos en Python vs Rust

En Python, el módulo `threading` sufre del **GIL (Global Interpreter Lock)**: solo un hilo puede ejecutar código Python a la vez. Si lanzas 8 hilos en una máquina de 8 núcleos, todos se turnan en un solo núcleo.

En **Rust**:
* `std::thread::spawn` invoca directamente a la API de hilos del Sistema Operativo (`pthread` en Linux / Mac).
* **Paralelismo 100% Real**: Si tienes 8 núcleos, tus 8 hilos corren simultáneamente al 100% de potencia del CPU.
* **Seguridad de Memoria (*Fearless Concurrency*)**: El compilador de Rust no te permite compilar código que tenga *data races* o acceso concurrente inseguro.

---

## 2. Lanzar un Hilo con `thread::spawn`

```rust
use std::thread;

let handle = thread::spawn(|| {
    println!("Hola desde otro núcleo!");
    100 // Retorna este número al terminar
});
```

---

## 3. Esperar y Recibir Resultados con `handle.join()`

A diferencia de C++ donde `join()` no devuelve el resultado del hilo, en Rust el método `.join()` devuelve un `Result<T, Box<dyn Any>>`:

```rust
let resultado: i32 = handle.join().expect("El hilo falló");
```

Si el hilo secundario sufrió un *panic*, `join()` te entrega un `Err`, permitiendo que el hilo principal sobreviva y maneje el fallo.

---

## 4. La Obligación del Closure `move`

Para usar variables de la función padre dentro del hilo, es obligatorio usar `move`:

```rust
let lote = vec![1.0, 2.0, 3.0];

thread::spawn(move || {
    // 'lote' ahora le pertenece exclusivamente a este hilo
    procesar(lote);
});
```

---

## 5. Ejecutables de la Clase

```bash
# 1. Probar la teoría y pipeline paralelo:
cargo run --bin c01_teoria

# 2. Explorar el laboratorio de errores comunes:
cargo run --bin c01_errores

# 3. Correr el reto de producto punto paralelo:
cargo run --bin c01_ejercicios

# 4. Validar las pruebas unitarias:
cargo test --bin c01_ejercicios
```
