# Clase 01: Álgebra Lineal y Tensores con `ndarray` 🧠📐

En los modelos de Inteligencia Artificial (desde redes densas hasta Transformers como GPT o Llama), casi todo el tiempo de CPU/GPU se consume en operaciones de tensores multidimensionales: multiplicación de matrices (**GEMM**), cálculo de similitud entre embeddings y transformaciones de activaciones.

En el ecosistema Python, la herramienta estándar es **NumPy** (`numpy.ndarray`). En el ecosistema nativo de Rust, la librería equivalente y más optimizada es **`ndarray`**.

---

## 🎯 Objetivos de la Clase
1. Comprender la **anatomía física** de un tensor en la memoria RAM:
   - Dimensiones (`shape`).
   - Saltos de memoria (`strides`).
   - Contigüidad en formato C (*Row-Major*) y cómo maximizar los aciertos en la caché L1/L2.
2. Dominar las **vistas cero-copia** (`ArrayView`) y slicing multidimensional con la macro `s![filas, columnas]`.
3. Implementar operaciones de **álgebra lineal**:
   - Multiplicación matricial (*Matmul*) con `.dot()`.
   - Multiplicación elemento a elemento (*Hadamard*) con `*`.
   - Normalización de vectores de embedding ($L_2$ norm) y similitud coseno.
4. Aplicar **Broadcasting** para sumar vectores de sesgo (*bias*) a lotes de predicciones.

---

## 🏗️ Analogía: NumPy vs Rust `ndarray`

| Operación | Python (NumPy) | Rust (`ndarray`) |
|---|---|---|
| Crear Matriz 2D | `np.array([[1.0, 2.0], [3.0, 4.0]])` | `array![[1.0, 2.0], [3.0, 4.0]]` |
| Slicing Cero-Copia | `sub = matriz[0:10, :]` | `let sub = matriz.slice(s![0..10, ..]);` |
| Matmul | `A @ B` o `np.dot(A, B)` | `A.dot(&B)` |
| Element-wise | `A * B` | `&A * &B` |
| Transpuesta (0 copias) | `A.T` | `A.t()` |
| Forma del Tensor | `A.shape` | `A.shape()` |

---

## 🚀 Comandos de Ejecución

```bash
# 1. Demostración teórica de tensores y broadcasting
cargo run --bin c01_teoria

# 2. Laboratorio de errores comunes y caché L1/L2
cargo run --bin c01_errores

# 3. Ejecutar suite de pruebas de los ejercicios prácticos
cargo test --bin c01_ejercicios
```
