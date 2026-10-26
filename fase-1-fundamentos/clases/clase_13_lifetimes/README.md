# 📚 Clase 13: Lifetimes Básicos (`'a`) en Rust para IA

En esta clase desmitificaremos los **Lifetimes**. Aprenderás cómo la sintaxis del apóstrofe (`'a`) actúa como una etiqueta de seguridad para conectar préstamos de salida con préstamos de entrada, y cómo construir vistas de tensores (*Tensor Views*) de costo cero en memoria asegurando que nunca sobrevivan a su buffer original.

---

## 📑 Contenido
1. [¿Qué es un Lifetime `'a`?](#1-qué-es-un-lifetime)
2. [Funciones que Devuelven Referencias](#2-funciones-y-referencias)
3. [Structs con Referencias (`TensorView<'a>`)](#3-structs-con-referencias)
4. [Prevención de Punteros Colgantes (*Dangling Pointers*)](#4-prevención-de-punteros-colgantes)
5. [Comandos para Ejecutar](#5-comandos-para-ejecutar)

---

## 5. Comandos para Ejecutar
```bash
cargo run --bin c13_teoria
cargo run --bin c13_errores
cargo run --bin c13_ejercicios
```
