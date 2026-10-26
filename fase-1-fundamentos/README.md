# 🦀 Fase 1: Fundamentos de Rust para Ingenieros en IA

Bienvenido a la primera fase. Aquí aprenderemos el lenguaje paso a paso, clase por clase, construyendo las bases necesarias para trabajar con datos, tensores y modelos de IA en Rust.

---

## 🗂️ Estructura Modular de las Clases

Cada clase cuenta con su propia carpeta dentro de `clases/` que incluye:
* 📖 **`README.md`:** Guía teórica exhaustiva, tablas, diagramas y notas de IA.
* 💻 **`01_teoria.rs`:** Código funcional guiado (`cargo run --bin cXX_teoria`).
* ⚠️ **`02_errores_comunes.rs`:** Laboratorio de errores del compilador (`cargo run --bin cXX_errores`).
* 🎯 **`03_ejercicios.rs`:** Retos prácticos de IA con autocomprobación (`cargo run --bin cXX_ejercicios`).

---

| Clase | Tema | Carpeta de Materiales | Comandos Rápidos |
|---|---|---|---|
| **Clase 01** | Variables, Mutabilidad, Constantes y Shadowing | [📂 `clase_01_variables/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_01_variables/README.md) | `cargo run --bin c01_teoria`<br>`cargo run --bin c01_errores`<br>`cargo run --bin c01_ejercicios` |
| **Clase 02** | Tipos de Datos (Escalares y Compuestos) y Casting | [📂 `clase_02_tipos_de_datos/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_02_tipos_de_datos/README.md) | `cargo run --bin c02_teoria`<br>`cargo run --bin c02_errores`<br>`cargo run --bin c02_ejercicios` |
| **Clase 03** | Funciones, Expresiones vs Sentencias | [📂 `clase_03_funciones/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_03_funciones/README.md) | `cargo run --bin c03_teoria`<br>`cargo run --bin c03_errores`<br>`cargo run --bin c03_ejercicios` |
| **Clase 04** | Control de Flujo (`if`, `loop`, `while`, `for`) en IA | [📂 `clase_04_control_de_flujo/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_04_control_de_flujo/README.md) | `cargo run --bin c04_teoria`<br>`cargo run --bin c04_errores`<br>`cargo run --bin c04_ejercicios` |
| **Clase 05** | Ownership: Stack vs Heap y Semántica Move | [📂 `clase_05_ownership/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_05_ownership/README.md) | `cargo run --bin c05_teoria`<br>`cargo run --bin c05_errores`<br>`cargo run --bin c05_ejercicios` |
| **Clase 06** | Borrowing y Referencias (`&` vs `&mut`) | [📂 `clase_06_borrowing/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_06_borrowing/README.md) | `cargo run --bin c06_teoria`<br>`cargo run --bin c06_errores`<br>`cargo run --bin c06_ejercicios` |
| **Clase 07** | Slices y Buffers de Datos | [📂 `clase_07_slices/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_07_slices/README.md) | `cargo run --bin c07_teoria`<br>`cargo run --bin c07_errores`<br>`cargo run --bin c07_ejercicios` |
| **Clase 08** | Structs y Métodos (`impl`) | [📂 `clase_08_structs/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_08_structs/README.md) | `cargo run --bin c08_teoria`<br>`cargo run --bin c08_errores`<br>`cargo run --bin c08_ejercicios` |
| **Clase 09** | Enums y Pattern Matching (`match`) | [📂 `clase_09_enums/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_09_enums/README.md) | `cargo run --bin c09_teoria`<br>`cargo run --bin c09_errores`<br>`cargo run --bin c09_ejercicios` |
| **Clase 10** | Manejo Robusto de Errores (`Option` y `Result`) | [📂 `clase_10_errores/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_10_errores/README.md) | `cargo run --bin c10_teoria`<br>`cargo run --bin c10_errores`<br>`cargo run --bin c10_ejercicios` |
| **Clase 11** | Iteradores y Closures (Funciones Lambda) | [📂 `clase_11_iteradores/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_11_iteradores/README.md) | `cargo run --bin c11_teoria`<br>`cargo run --bin c11_errores`<br>`cargo run --bin c11_ejercicios` |
| **Clase 12** | Traits (Interfaces y Abstracciones en IA) | [📂 `clase_12_traits/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_12_traits/README.md) | `cargo run --bin c12_teoria`<br>`cargo run --bin c12_errores`<br>`cargo run --bin c12_ejercicios` |
| **Clase 13** | Lifetimes Básicos (`'a`) y Zero-Copy | [📂 `clase_13_lifetimes/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_13_lifetimes/README.md) | `cargo run --bin c13_teoria`<br>`cargo run --bin c13_errores`<br>`cargo run --bin c13_ejercicios` |
| **Clase 14** | Checkpoint Fase 1: Regresión Lineal desde cero | [📂 `clase_14_proyecto_regresion/`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_14_proyecto_regresion/README.md) | `cargo run --bin c14_interactivo` (🎮 en vivo)<br>`cargo run --bin c14_proyecto`<br>`cargo run --bin c14_errores`<br>`cargo run --bin c14_ejercicios` |

---

## 🚀 Comandos Rápidos

```bash
# Ver el catálogo interactivo en consola:
cargo run

# Probar la teoría de cualquier clase (ej. Clase 05):
cargo run --bin c05_teoria

# Probar el laboratorio de errores de cualquier clase (ej. Clase 06):
cargo run --bin c06_errores

# Probar y validar los retos de cualquier clase (ej. Clase 11):
cargo run --bin c11_ejercicios

# Verificar sintaxis y tipos sin generar binarios:
cargo check
```
