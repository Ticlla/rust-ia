# ⚡ Clase 05: Paralelismo de Datos con `Rayon` en IA

Bienvenido a la quinta clase de la **Fase 2**. En esta unidad aprenderás a aplicar **Data Parallelism** (Paralelismo de Datos masivo) sobre grandes volúmenes de datos, matrices y tensores, convirtiendo bucles secuenciales en código multihilo ultra optimizado que exprime todos los núcleos de tu CPU sin lidiar manualmente con hilos ni bloqueos.

---

## 📑 Contenido de la Clase

1. [Paralelismo de Tareas vs Paralelismo de Datos](#1-paralelismo-de-tareas-vs-paralelismo-de-datos)
2. [El Algoritmo de "Work Stealing" de Rayon](#2-el-algoritmo-de-work-stealing-de-rayon)
3. [De `iter()` a `par_iter()`: La Transición Mágica](#3-de-iter-a-par_iter-la-transición-mágica)
4. [Map-Reduce en Tensores: Norma L2, Dot Product y LayerNorm](#4-map-reduce-en-tensores-norma-l2-dot-product-y-layernorm)
5. [Ejecutables de la Clase](#5-ejecutables-de-la-clase)

---

## 1. Paralelismo de Tareas vs Paralelismo de Datos

* **Paralelismo de Tareas (`thread::spawn`):**
  * Cada hilo hace un trabajo diferente (ej: Hilo 1 carga datos de disco, Hilo 2 recibe peticiones HTTP, Hilo 3 guarda checkpoints).
  * Es ideal para arquitectura de sistemas.
* **Paralelismo de Datos (`Rayon`):**
  * Tienes **un solo algoritmo** (ej: calcular la función de activación $y = \text{GELU}(x)$ o normalizar un vector).
  * Tienes **millones de datos** (un tensor de 10 millones de números).
  * Rayon divide los 10 millones de números entre los núcleos de tu CPU para que todos trabajen en paralelo.

---

## 2. El Algoritmo de "Work Stealing" de Rayon

En lugar de crear y destruir hilos del Sistema Operativo continuamente (lo cual es muy costoso en memoria y CPU), Rayon:
1. Crea un **Thread Pool** al inicio igual al número de núcleos lógicos de tu procesador.
2. Cada núcleo tiene su propia cola de tareas locales.
3. **Robo de Trabajo (*Work Stealing*):** Si el Núcleo 1 termina su lote de tensores antes que el Núcleo 2, **le roba trabajo a la cola del Núcleo 2**.
4. Ningún núcleo del procesador se queda ocioso mientras haya datos por procesar.

---

## 3. De `iter()` a `par_iter()`: La Transición Mágica

En Rust estándar (1 solo núcleo):
```rust
let cuadrados: Vec<f32> = tensor.iter().map(|&x| x * x).collect();
```

Con Rayon (todos los núcleos en paralelo):
```rust
use rayon::prelude::*;

let cuadrados: Vec<f32> = tensor.par_iter().map(|&x| x * x).collect();
```
Solo cambias `.iter()` por `.par_iter()`. Rayon se encarga de dividir, balancear la carga y juntar los resultados en orden.

---

## 4. Ejecutables de la Clase

```bash
# 1. Benchmark real de 5 millones de números con activación GELU (Secuencial vs Rayon):
cargo run --bin c05_teoria

# 2. Laboratorio de errores y antipatrones (Overhead en datos chicos, Mutex en par_iter):
cargo run --bin c05_errores

# 3. Reto práctico: LayerNorm y Multiplicación Matriz-Vector Paralela:
cargo run --bin c05_ejercicios

# 4. Validar las pruebas unitarias:
cargo test --bin c05_ejercicios
```
