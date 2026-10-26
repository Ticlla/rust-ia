# Fase 3: Datos y Numérica para IA en Rust 📊⚡

Bienvenido a la **Fase 3** del curso de Rust para Ingenieros de Inteligencia Artificial.
Aquí dominaremos el procesamiento numérico de alto rendimiento, manipulación masiva de datos en memoria sin copias, serialización binaria de pesos de redes neuronales y aceleración vectorial por hardware (SIMD).

---

## 🗺️ Estructura de Clases

| Clase | Tema | Conceptos Clave | Estado |
|---|---|---|---|
| **[Clase 01](clases/clase_01_ndarray/)** | Tensores y Álgebra Lineal con `ndarray` | Formas (`shape`), saltos (`strides`), vistas cero-copia (`s![]`), matmul (`dot`), broadcasting. | 🟢 Completada |
| **[Clase 02](clases/clase_02_polars/)** | Procesamiento Masivo con `Polars` | Arrow columnar en RAM, LazyFrame vs DataFrame, optimizador de consultas, pushdown de predicados. | 🟢 Completada |
| **[Clase 03](clases/clase_03_serializacion/)** | Serialización y Formatos Binarios | Por qué JSON/Pickle son lentos en IA, serialización binaria compacta con `bincode` y `safetensors`. | 🟢 Completada |
| **[Clase 04](clases/clase_04_mmap/)** | Memoria Mapeada (`memmap2`) | Carga cero-copias de modelos masivos (LLMs) sin saturar la memoria RAM física mediante el kernel de Linux. | 🟢 Completada |
| **[Clase 05](clases/clase_05_simd/)** | Vectorización y SIMD Explícito | Instrucciones AVX2/AVX-512, paralelismo a nivel de registros de CPU, procesamiento de 8 floats por ciclo. | 🟢 Completada |
| **[Clase 06](clases/clase_06_motor_vectorial/)** | Checkpoint Integrador: Motor de Búsqueda Vectorial | Búsqueda por similitud coseno acelerada por hardware, indexación y ranking de embeddings de IA. | 🟢 Completada |

---

## 🛠️ Ejecución de Pruebas y Binarios

```bash
# Ejecutar teoría de la Clase 01
cargo run --bin c01_teoria

# Ejecutar laboratorio de errores comunes
cargo run --bin c01_errores

# Ejecutar ejercicios y tests automatizados
cargo test --bin c01_ejercicios
```
