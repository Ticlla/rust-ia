# 📚 Clase 05: Ownership (Propiedad), Stack vs Heap y Semántica Move

En esta clase aprenderás el pilar central de Rust: el **Ownership**. Entenderás cómo Rust garantiza seguridad de memoria sin un recolector de basura (*Garbage Collector*), por qué los datos dinámicos como vectores de tensores se mueven (*Move*) en lugar de clonarse, y la diferencia con los tipos que viven en el Stack (*Copy*).

---

## 📑 Contenido
1. [Las 3 Reglas de Oro del Ownership](#1-las-3-reglas-de-oro)
2. [Stack vs Heap en IA](#2-stack-vs-heap)
3. [Semántica Move vs Semántica Copy](#3-move-vs-copy)
4. [Transferencia a Funciones y `.clone()`](#4-funciones-y-clone)
5. [Comandos para Ejecutar](#5-comandos-para-ejecutar)

---

## 5. Comandos para Ejecutar
```bash
cargo run --bin c05_teoria
cargo run --bin c05_errores
cargo run --bin c05_ejercicios
```
