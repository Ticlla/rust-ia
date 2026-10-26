# 📚 Clase 01: Variables, Mutabilidad e Inmutabilidad en Rust para IA

Bienvenido a la primera clase de fundamentos. En esta unidad aprenderás cómo Rust maneja las variables en memoria, por qué la inmutabilidad es su valor por defecto y cómo esta decisión previene fallos críticos en sistemas y modelos de Inteligencia Artificial.

---

## 📑 Contenido de la Clase

1. [Inmutabilidad por Defecto](#1-inmutabilidad-por-defecto)
2. [Mutabilidad Explícita con `let mut`](#2-mutabilidad-explícita-con-let-mut)
3. [Constantes (`const`) vs Variables (`let`)](#3-constantes-const-vs-variables-let)
4. [Shadowing (Sombreado de Variables)](#4-shadowing-sombreado-de-variables)
5. [Tipos Escalares Clave en IA](#5-tipos-escalares-clave-en-ia)
6. [Materiales y Ejecutables](#6-materiales-y-ejecutables)

---

## 1. Inmutabilidad por Defecto

En la mayoría de los lenguajes de programación comunes (como Python o JavaScript), todas las variables son mutables por defecto:
```python
# En Python todo puede cambiar en cualquier momento:
lr = 0.01
lr = 0.005 # Válido sin avisar a nadie
```

En Rust, cuando declaras una variable con `let`, **es inmutable por defecto**:
```rust
let learning_rate: f32 = 0.01;
// learning_rate = 0.005; // ❌ ERROR: cannot assign twice to immutable variable `learning_rate`
```

### ¿Por qué esto es vital en IA?
En pipelines de entrenamiento distribuido o concurrente (multi-GPU / multi-hilo), si varios hilos leen la configuración del modelo o la arquitectura, **saber que nadie puede modificarla en secreto evita condiciones de carrera (*data races*) catastróficas**.

---

## 2. Mutabilidad Explícita con `let mut`

Cuando un dato realmente necesita cambiar a lo largo del tiempo (como el contador de épocas de entrenamiento o el acumulador de pérdida), debes declarar tu intención de forma explícita usando la palabra clave `mut`:

```rust
let mut epoca: usize = 0;
epoca += 1; // ✅ Totalmente válido
```

> [!TIP]
> En Rust, la mutabilidad no es una propiedad del dato, sino del **enlace (*binding*)** que creas con su nombre.

---

## 3. Constantes (`const`) vs Variables (`let`)

| Característica | Variable Inmutable (`let`) | Constante (`const`) |
| :--- | :--- | :--- |
| **Cuándo se evalúa** | En tiempo de ejecución (*Runtime*) | En tiempo de compilación (*Compile time*) |
| **Anotación de tipo** | Opcional (Rust la infiere) | **Obligatoria siempre** |
| **Convención de nombre** | `snake_case` | `SCREAMING_SNAKE_CASE` |
| **Ámbito (*Scope*)** | Local al bloque donde se define | Puede ser global a todo el módulo |

```rust
const MAX_EPOCAS: usize = 100;
const FACTOR_DESCUENTO: f32 = 0.99; // Gamma en Reinforcement Learning
```

---

## 4. Shadowing (Sombreado de Variables)

El **Shadowing** consiste en declarar una nueva variable con el **mismo nombre** que una variable anterior usando `let`:

```rust
let batch_size = "64";              // 1. &str (texto)
let batch_size: usize = 64;          // 2. usize (número entero)
let batch_size = batch_size * 2;     // 3. Escalado a 128
```

### ¿Por qué es una joya para Pipelines de IA?
En Python a menudo terminamos con nombres feos y propensos a confusión:
`batch_size_str`, `batch_size_int`, `batch_size_final`.
Con Shadowing en Rust, transformas el dato en cada etapa del pipeline reutilizando el nombre semántico ideal sin mutar ni perder seguridad de tipos.

---

## 5. Tipos Escalares Clave en IA

* **`f32`:** Precisión simple (32 bits). El estándar por excelencia en Deep Learning (tensores, activaciones, pesos).
* **`f64`:** Doble precisión (64 bits). Usado en simulaciones científicas o cálculos numéricos de alta precisión.
* **`usize`:** Entero sin signo dependiente de la arquitectura (64 bits en CPUs modernas). Usado para índices de arrays, tamaños de datasets y conteo de muestras.
* **`bool`:** `true` o `false` (flags de validación, estados de convergencia).

---

## 6. Materiales y Ejecutables de esta Clase

Esta clase cuenta con 3 herramientas interactivas independientes:

| Archivo | Propósito | Comando para ejecutar |
| :--- | :--- | :--- |
| **[`01_teoria.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_01_variables/01_teoria.rs)** | Código guiado paso a paso con todos los conceptos | `cargo run --bin c01_teoria` |
| **[`02_errores_comunes.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_01_variables/02_errores_comunes.rs)** | Laboratorio interactivo de errores del compilador (`E0384`, `E0282`, `E0308`) | `cargo run --bin c01_errores` |
| **[`03_ejercicios.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-1-fundamentos/clases/clase_01_variables/03_ejercicios.rs)** | Retos prácticos aplicados a IA con autocomprobación | `cargo run --bin c01_ejercicios` |
