# 🔐 Clase 03: Memoria Compartida en Paralelo con `Arc<T>` y `Mutex<T>`

Bienvenido a la tercera clase de la **Fase 2**. En esta unidad aprenderás a romper la barrera del Ownership único cuando múltiples hilos necesitan **poseer y modificar la misma estructura de datos en la memoria RAM** (el patrón fundamental detrás de los Parameter Servers y el entrenamiento con Paralelismo de Datos).

---

## 📑 Contenido de la Clase

1. [El Dilema de la Memoria Compartida](#1-el-dilema-de-la-memoria-compartida)
2. [¿Qué es `Arc<T>`? (Contador Atómico)](#2-qué-es-arct-contador-atómico)
3. [¿Qué es `Mutex<T>`? (La Llave de Exclusión Mutua)](#3-qué-es-mutext-la-llave-de-exclusión-mutua)
4. [El Dúo Dinámico: `Arc<Mutex<T>>`](#4-el-dúo-dinámico-arcmutext)
5. [Backpropagation en Paralelo: Cómo los hilos aplican gradientes](#5-backpropagation-en-paralelo-cómo-los-hilos-aplican-gradientes)
6. [Ejecutables de la Clase](#6-ejecutables-de-la-clase)

---

## 1. El Dilema de la Memoria Compartida

En la Clase 02 vimos que los canales `mpsc` transfieren el Ownership de un hilo a otro.  
Pero en Deep Learning, si tu modelo tiene **10 GB de parámetros**, **no puedes pasar 10 GB por un canal cada microsegundo**.  
Necesitas que todos los hilos lean y actualicen **la misma dirección física de memoria RAM**.

---

## 2. ¿Qué es `Arc<T>`? (Contador Atómico)

`Arc` significa **Atomic Reference Counting**:
* Envuelve un dato en el Heap.
* Cada vez que haces `Arc::clone(&modelo)`, no copias el modelo: **solo sumas 1 a un contador atómico**.
* Varios hilos pueden ser co-dueños del mismo modelo.
* Cuando todos los hilos mueren y el contador llega a 0, Rust destruye el modelo automáticamente.

---

## 3. ¿Qué es `Mutex<T>`? (La Llave de Exclusión Mutua)

`Arc` solo permite lecturas inmutables. Para permitir escrituras concurrentes seguras, usamos `Mutex`:
* `let mut guard = mutex.lock().unwrap();`
* Solo **un hilo a la vez** puede tener el cerrojo.
* **Liberación RAII**: Cuando `guard` sale de ámbito `{ }`, la llave se suelta sola automáticamente. Nunca se te olvidará desbloquearla como pasa en C++.

---

## 4. El Dúo Dinámico: `Arc<Mutex<T>>`

```rust
use std::sync::{Arc, Mutex};

// 1. Creamos el modelo protegido:
let modelo = Arc::new(Mutex::new(Pesos { w: 0.0 }));

// 2. Por cada hilo, clonamos el Arc:
let modelo_hilo = Arc::clone(&modelo);

thread::spawn(move || {
    let mut pesos = modelo_hilo.lock().unwrap();
    pesos.w -= lr * gradiente; // Modificación 100% segura
});
```

---

## 5. Ejecutables de la Clase (En Orden Secuencial)

```bash
# --- RUTA DIDÁCTICA: De los Errores a la Solución ---

# Caso 1: Intentar mover la misma variable a 2 hilos (E0382) -> Solución: Arc<T>
cargo run --bin c03_caso1

# Caso 2: Intentar mutar dos veces en paralelo (E0499) -> Solución: Mutex<T>
cargo run --bin c03_caso2

# Caso 3: Intentar usar Rc<RefCell<T>> entre hilos (E0277) -> Solución: Arc<Mutex<T>>
cargo run --bin c03_caso3

# --- TEORÍA COMPLETA Y RETO ---

# 4. Probar la teoría integrada con 4 hilos entrenando en paralelo:
cargo run --bin c03_teoria

# 5. Explorar el laboratorio de errores compilados:
cargo run --bin c03_errores

# 6. Correr el reto del Parameter Server concurrente:
cargo run --bin c03_ejercicios

# 7. Validar las pruebas unitarias:
cargo test --bin c03_ejercicios
```
