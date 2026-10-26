# 📚 Clase 10: Manejo Robusto de Errores (`Option` y `Result`) en IA

En esta clase aprenderás la filosofía de Rust frente a fallos y valores ausentes: **eliminar el error del millón de dólares (*Null Pointer Exception*)**. Usaremos `Option<T>` para datos que pueden o no existir y `Result<T, E>` para operaciones fallibles (como carga de datasets), junto con el operador `?` para propagación limpia.

---

## 📑 Contenido
1. [`Option<T>` (`Some` vs `None`)](#1-option)
2. [`Result<T, E>` (`Ok` vs `Err`)](#2-result)
3. [El Operador `?`](#3-el-operador-)
4. [Comandos para Ejecutar](#4-comandos-para-ejecutar)

---

## 4. Comandos para Ejecutar
```bash
cargo run --bin c10_teoria
cargo run --bin c10_errores
cargo run --bin c10_ejercicios
```
