# Clase 04: Framework de Deep Learning `Candle` (Hugging Face) 🕯️⚡

En esta clase aprenderemos a usar **`Candle`**, el framework de Deep Learning creado por **Hugging Face** en Rust para sustituir la sobrecarga pesada de PyTorch en entornos de producción y servicio de modelos.

---

## 🎯 Objetivos de la Clase

1. **La Filosofía de Candle:** Un PyTorch minimalista, tipado y sin dependencias pesadas de C++, capaz de correr en CPU (AVX2/NEON), WGPU y CUDA.
2. **El Dispositivo (`Device`) y el `Tensor`:**
   * Abstracción de hardware: `Device::Cpu` vs `Device::new_cuda(0)`.
   * Tipos de datos en Deep Learning: `F32`, `F16`, `BF16`.
   * Formas (`shape`), broadcasting y operaciones tensoriales básicas.
3. **Capas Neuronales y Activaciones:**
   * Capas lineales densas (`Linear`).
   * Activaciones modernas de LLMs: **GELU** (BERT/GPT) y **SwiGLU** (LLaMA/Qwen).
   * **RMSNorm:** La normalización que reemplazó a LayerNorm en los LLMs modernos.
4. **Carga de Pesos desde `.safetensors` con `VarBuilder`:**
   * Cómo Candle lee directamente los archivos `.safetensors` de Hugging Face y los asocia a estructuras de Rust en microsegundos.

---

## 📂 Archivos de la Clase

* [`01_teoria.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-4-ia-ml/clases/clase_04_candle/01_teoria.rs): Creación de tensores, operaciones tensoriales, capas densas con activación SwiGLU y carga de pesos.
* [`02_errores_comunes.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-4-ia-ml/clases/clase_04_candle/02_errores_comunes.rs): Operaciones entre tensores en diferentes `Device`, incompatibilidad de `DType` (F32 vs F16) y desalineación en multiplicación matricial.
* [`03_ejercicios.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-4-ia-ml/clases/clase_04_candle/03_ejercicios.rs): Tests automatizados para verificar capas lineales, activación SwiGLU y cálculo de RMSNorm.
