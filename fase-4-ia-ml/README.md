# Fase 4: IA y Machine Learning en Rust 🤖⚡

Bienvenido a la **Fase 4** del programa de formación en Rust para Ingenieros de Inteligencia Artificial.
Aquí construimos, entrenamos, optimizamos y servimos modelos de Machine Learning y Deep Learning nativos en Rust.

---

## 🗺️ Mapa de Clases de la Fase 4

| Clase | Tema | Conceptos Clave | Estado |
|---|---|---|---|
| **[Clase 01](clases/clase_01_linfa/)** | ML Clásico con `linfa` | Estructura `Dataset`, Regresión Lineal, K-Means clustering, Métricas de evaluación. | 🟢 Completada |
| **[Clase 02](clases/clase_02_onnx/)** | Inferencia de Modelos con ONNX (`ort`) | Inferencia de modelos de PyTorch en Rust, Dynamic Batching, Sesiones de inferencia. | 🟢 Completada |
| **[Clase 03](clases/clase_03_tokenizers/)** | Tokenización Masiva con `tokenizers` | Algoritmos BPE / WordPiece de Hugging Face, codificación en paralelo con Rayon. | 🟢 Completada |
| **[Clase 04](clases/clase_04_candle/)** | Deep Learning con `Candle` | Tensores de Hugging Face en Rust, capas neuronales, activaciones (GELU, SwiGLU). | 🟡 En Curso |
| **Clase 05** | Transformers y KV-Cache | Multi-Head Attention, RoPE (Rotary Embeddings), Key-Value Cache autoregresivo. | ⚪ Pendiente |
| **Clase 06** | Checkpoint: Motor de Inferencia LLM | Mini-vLLM / Mini-Ollama en Rust con Token Streaming en tiempo real. | ⚪ Pendiente |

---

## 🛠️ Ejecución de Pruebas y Binarios

```bash
# Ejecutar teoría de la Clase 01
cargo run --bin c01_teoria

# Ejecutar errores comunes
cargo run --bin c01_errores

# Ejecutar tests automatizados
cargo test --bin c01_ejercicios
```
