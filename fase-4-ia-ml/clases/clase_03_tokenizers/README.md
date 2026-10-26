# Clase 03: Tokenización de Alto Rendimiento con `tokenizers` de Hugging Face 🔤⚡

En esta clase aprenderemos a usar la librería oficial **`tokenizers`** de Hugging Face en su lenguaje de origen: **Rust**.
La tokenización es la puerta de entrada obligatoria a cualquier modelo de lenguaje moderno (LLMs como LLaMA, GPT-4, Qwen, Mistral o BERT).

---

## 🎯 Objetivos de la Clase

1. **¿Qué es un Token?:** Por qué los caracteres son demasiado largos ($O(N^2)$ en atención) y las palabras completas sufren de explosión de vocabulario y palabras desconocidas (`<UNK>`).
2. **Algoritmo BPE (Byte-Pair Encoding):** Cómo las subpalabras permiten representar cualquier término técnico nuevo descomponiéndolo en morfemas frecuentes o bytes crudos.
3. **Las 4 Fases de un Tokenizer:**
   * **Normalizer:** Limpieza de texto (Unicode, minúsculas).
   * **Pre-Tokenizer:** Segmentación en palabras o bytes (`ByteLevel`, `Whitespace`).
   * **Model:** Vocabulario y reglas de unión (`BPE`, `WordPiece`).
   * **Post-Processor:** Tokens especiales (`[CLS]`, `[SEP]`, `<|endoftext|>`).
4. **Anatomía de un `Encoding`:**
   * `ids`: Vector de enteros `u32` que entran a la matriz de embeddings del LLM.
   * `tokens`: Representación textual legible de cada token.
   * `attention_mask`: Filtro binario para ignorar tokens de relleno (padding) en un batch.
   * `offsets`: Posiciones de caracteres exactas en el texto original (crucial para RAG y resaltado de citas).
5. **Codificación Masiva en Paralelo:** Uso de `encode_batch` para procesar miles de textos por segundo sin consumir el hilo principal.

---

## 📂 Archivos de la Clase

* [`01_teoria.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-4-ia-ml/clases/clase_03_tokenizers/01_teoria.rs): Construcción de un tokenizador BPE desde cero, codificación, decodificación e inspección de offsets.
* [`02_errores_comunes.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-4-ia-ml/clases/clase_03_tokenizers/02_errores_comunes.rs): Truncamiento accidental de texto, desalineación de padding en lotes y pérdida de espacios por pre-tokenización incorrecta.
* [`03_ejercicios.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-4-ia-ml/clases/clase_03_tokenizers/03_ejercicios.rs): Tests automatizados para validar flujos de tokenización y reconstrucción de texto (`roundtrip`).
