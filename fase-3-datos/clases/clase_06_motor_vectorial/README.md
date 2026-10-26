# Clase 06: Checkpoint Final - Motor de Búsqueda Vectorial (Vector DB / RAG) 🚀

En este proyecto integrador unificamos **las 5 clases de la Fase 3** para construir un motor de búsqueda vectorial de alto rendimiento desde cero en Rust, equivalente al núcleo de almacenamiento y recuperación de bases de datos como **Qdrant** o **LanceDB**.

---

## 🏗️ Arquitectura del Motor

```text
 1. Consultas y Metadatos (Polars)
    Filtrado columnar Lazy: category == "IA" && fecha >= 2024
                       │
                       ▼ Índices válidos
 2. Archivo .safetensors en Disco (Clase 03)
    Contiene la matriz de embeddings normalizados: [N, D]
                       │
                       ▼ Mapeo virtual cero-copia
 3. Memoria Mapeada (memmap2 - Clase 04)
    Punteros directos a los bytes de embeddings sin inflar la RAM
                       │
                       ▼ Slices &[f32]
 4. Scoring por Hardware (SIMD AVX2 + FMA - Clase 05)
    Producto punto a 256 bits (8 floats por ciclo) contra el embedding de la query
                       │
                       ▼ Scores [0.0 .. 1.0]
 5. Ranking Top-K (ndarray / Sort - Clase 01)
    Ordenamiento y retorno de los K documentos más relevantes para el LLM
```

---

## 📂 Archivos de la Clase

* [`01_teoria.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-3-datos/clases/clase_06_motor_vectorial/01_teoria.rs): Arquitectura completa del motor vectorial paso a paso y pipeline de búsqueda en vivo.
* [`02_errores_comunes.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-3-datos/clases/clase_06_motor_vectorial/02_errores_comunes.rs): Cuellos de botella en Vector DBs: O(N) secuencial sin prefiltrado, copias innecesarias y desnormalización de vectores.
* [`03_ejercicios.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-3-datos/clases/clase_06_motor_vectorial/03_ejercicios.rs): Tests automatizados para verificar:
  1. Indexación y persistencia en `.safetensors`.
  2. Búsqueda Híbrida (Pre-filtrado de metadatos con `Polars` + ranking SIMD).
  3. Búsqueda Top-K con similitud coseno.
* [`04_interactivo.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-3-datos/clases/clase_06_motor_vectorial/04_interactivo.rs): Consola interactiva con corpus extendido de 24 documentos técnicos y benchmark de estrés sobre 10,000 vectores en SSD.
