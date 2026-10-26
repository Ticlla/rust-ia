# ⚡ Clase 09: Punteros Crudos y `unsafe` Rust (C / CUDA) en IA

Bienvenido a la novena clase de la **Fase 2**. En esta unidad nos adentraremos en el nivel más bajo del hardware y la memoria en Rust: cómo interactuar con punteros crudos (`raw pointers`) y cómo crear envoltorios seguros (*Safe Wrappers*) para librerías nativas escritas en **C / C++** o kernels de **NVIDIA CUDA**.

---

## 📑 Contenido de la Clase

1. [¿Qué es y qué NO es `unsafe`?](#1-qué-es-y-qué-no-es-unsafe)
2. [Punteros Crudos: `*const T` y `*mut T`](#2-punteros-crudos-const-t-y-mut-t)
3. [El Patrón "Safe Wrapper" para Tensores de IA](#3-el-patrón-safe-wrapper-para-tensores-de-ia)
4. [FFI (Foreign Function Interface) con C y CUDA](#4-ffi-foreign-function-interface-con-c-y-cuda)
5. [Ejecutables de la Clase](#5-ejecutables-de-la-clase)

---

## 1. ¿Qué es y qué NO es `unsafe`?

`unsafe` en Rust **NO significa "código malo o peligroso"**, ni desactiva el verificador de tipos de Rust.  
Solo desbloquea **5 superpoderes específicos** que el compilador no puede verificar matemáticamente por sí solo:

1. **Desreferenciar punteros crudos** (`*ptr`).
2. **Llamar a funciones externas** (como funciones en C o kernels de CUDA).
3. **Implementar traits `unsafe`** (como `Send` o `Sync` para tipos con punteros).
4. **Modificar variables estáticas mutables**.
5. **Acceder a campos de una `union`**.

El 99% de tu código sigue siendo **Safe Rust**. Usas `unsafe` en un bloque diminuto de 5 líneas para hablar con la GPU o C, y lo sellas dentro de una estructura segura.

---

## 2. Punteros Crudos vs Referencias de Rust

| Característica | Referencia Rust (`&T` / `&mut T`) | Puntero Crudo (`*const T` / `*mut T`) |
|---|:---:|:---:|
| **¿Puede ser nulo?** | ❌ Jamás (No existe `null` en Safe Rust) | ✅ Sí (`ptr.is_null()`) |
| **¿Garantiza memoria válida?** | ✅ Sí (Garantizado por el Borrow Checker) | ❌ No (Puede apuntar a memoria liberada) |
| **¿Reglas de Aliasing?** | ✅ Estrictas (1 mutable O muchos inmutables) | ❌ Ninguna (Múltiples punteros a la misma RAM) |
| **¿Para qué se usa en IA?** | Lógica de negocio, servidores web, pipelines | FFI con CUDA, cuBLAS, TensorRT, llama.cpp |

---

## 3. Ejecutables de la Clase

```bash
# 1. Probar la teoría (Punteros crudos, asignación manual de memoria y kernel CUDA mock):
cargo run --bin c09_teoria

# 2. Explorar errores comunes (Puntero nulo, Memory leaks, Aliasing):
cargo run --bin c09_errores

# 3. Reto práctico: Tensor Continuo y Producto Punto sin Bounds Checking:
cargo run --bin c09_ejercicios

# 4. Validar las pruebas unitarias:
cargo test --bin c09_ejercicios
```
