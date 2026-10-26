# 📚 Clase 03: Funciones, Sentencias y Expresiones en Rust para IA

En esta clase aprenderemos a diseñar funciones idiomáticas en Rust, distinguiendo entre expresiones y sentencias, y cómo implementar funciones de activación fundamentales en Deep Learning (como ReLU y Leaky ReLU).

---

## 📑 Contenido
1. [Anatomía de una Función y Tipado Estricto](#1-anatomía-de-una-función)
2. [Sentencias vs Expresiones](#2-sentencias-vs-expresiones)
3. [El Retorno Implícito](#3-el-retorno-implícito)
4. [Funciones de Activación en IA](#4-funciones-de-activación-en-ia)
5. [Comandos para Ejecutar](#5-comandos-para-ejecutar)

---

## 1. Anatomía de una Función
Rust exige que los tipos de los argumentos y el tipo de retorno sean siempre explícitos:
```rust
fn calcular_neurona(x: f32, w: f32, b: f32) -> f32 {
    x * w + b // Sin ';' al final -> es la expresión de retorno
}
```

---

## 2. Sentencias vs Expresiones
* **Sentencia (*Statement*):** Ejecuta una acción pero no produce un valor (termina en `;`). Ej: `let x = 5;`.
* **Expresión (*Expression*):** Evalúa a un valor (NO lleva `;`). Todo bloque `{ }` y todo `if/else` en Rust es una expresión.

---

## 5. Comandos para Ejecutar
```bash
cargo run --bin c03_teoria
cargo run --bin c03_errores
cargo run --bin c03_ejercicios
```
