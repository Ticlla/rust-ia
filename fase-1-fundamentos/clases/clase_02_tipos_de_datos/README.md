# 📚 Clase 02: Tipos de Datos y Conversión Estricta en Rust para IA

En esta clase profundizaremos en el sistema de tipos estático y estricto de Rust, y por qué es una de las mayores ventajas para el desarrollo de Inteligencia Artificial y computación numérica de alto rendimiento.

---

## 📑 Contenido
1. [Tipos Escalares Clave en IA](#1-tipos-escalares-clave-en-ia)
2. [Casting Estricto (No hay conversión implícita)](#2-casting-estricto-no-hay-conversión-implícita)
3. [Tuplas (Datos heterogéneos)](#3-tuplas-datos-heterogéneos)
4. [Arrays Fijos `[T; N]` en el Stack](#4-arrays-fijos-t-n-en-el-stack)
5. [Comandos para Ejecutar](#5-comandos-para-ejecutar)

---

## 1. Tipos Escalares Clave en IA

* **`u8` (0 a 255):** Píxeles de imágenes RGB, canales de audio PCM 8-bit, datos en bruto de sensores.
* **`f32` (Float 32 bits):** Estándar en Deep Learning (pesos, gradientes, tensores en GPU).
* **`f64` (Float 64 bits):** Precisión doble para simulaciones y cálculo numérico riguroso.
* **`usize`:** Entero positivo nativo de la CPU. Utilizado **obligatoriamente** para indexar arrays y medir longitudes.
* **`bool`:** Máscaras booleanas (ej. Attention Masks en Transformers).

---

## 2. Casting Estricto (No hay conversión implícita)

En C++ o Python, sumar un entero con un flotante convierte los tipos tras bambalinas:
```python
x = 10 + 2.5 # Python convierte 10 a float sin avisar
```
En Rust, **esto es un error de compilación**. Debes hacer la conversión explícita usando la palabra clave `as`:
```rust
let loss_total: f32 = 45.8;
let batches: usize = 10;
let loss_promedio = loss_total / (batches as f32); // ✅ Casting explícito
```

---

## 3. Tuplas y Arrays

* **Tupla `(A, B, C)`:** Permite agrupar valores de **distintos tipos**. Ideal para devolver múltiples métricas desde una función de evaluación (ej. `(nombre, accuracy, loss)`).
* **Array `[T; N]`:** Secuencia de tamaño fijo en el **Stack**. Todos los elementos son del **mismo tipo**. Ideal para shapes de tensores o dimensiones de capas (`[32, 224, 224]`).

---

## 5. Comandos para Ejecutar

```bash
cargo run --bin c02_teoria      # Teoría guiada
cargo run --bin c02_errores     # Laboratorio de errores (E0308, bounds, overflow)
cargo run --bin c02_ejercicios   # Retos prácticos en IA
```
