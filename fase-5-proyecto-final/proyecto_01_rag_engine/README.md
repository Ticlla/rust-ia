# Proyecto 01: Enterprise Hybrid RAG Engine

Motor de Búsqueda Semántica Híbrida de Nivel Empresarial construido en **Rust** con **Axum**, **Candle**, **SIMD AVX2**, y el algoritmo de fusión de rankings **Reciprocal Rank Fusion (RRF)**.

---

## 1. Arquitectura de Dos Etapas (Two-Stage Retrieval)

El motor implementa el patrón industrial estándar de búsqueda en dos etapas utilizado por sistemas de alto rendimiento (como Vespa, Qdrant y Elastic):

```
                       [ Consulta del Usuario (Query) ]
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
              ▼                                               ▼
   [ Búsqueda Densa (Semántica) ]                 [ Búsqueda Léxica (BM25) ]
   - Embeddings L2-Normalizados                   - Frecuencia de términos TF
   - Aceleración SIMD AVX2                        - Penalización de longitud avgdl
   - Similaridad Coseno                           - IDF Robertson-Spärck Jones
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                                      ▼
                      [ Fusión Híbrida: RRF ]
                      Score = w_d / (60 + r_d) + w_s / (60 + r_s)
                                      │
                                      ▼  (Top 15 Candidatos)
                     [ Reranker Neuronal Cross-Encoder ]
                     - Interacción cruzada Query-Documento
                     - Activación no lineal SiLU + Sigmoid
                                      │
                                      ▼  (Top 5 Definitivo)
                          [ Resultados Relevantes ]
```

---

## 2. Componentes Técnicos

1. **Segmentación Inteligente ([`chunker.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-5-proyecto-final/proyecto_01_rag_engine/src/ingestion/chunker.rs)):**
   * Divide documentos respetando límites de oraciones y párrafos.
   * Ventana deslizante con solapamiento (*overlap*) configurable para preservar el contexto entre fragmentos.

2. **Índice Denso SIMD ([`dense_simd.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-5-proyecto-final/proyecto_01_rag_engine/src/search/dense_simd.rs)):**
   * Vectores densos de dimensión $D$ generados por [`candle_embedder.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-5-proyecto-final/proyecto_01_rag_engine/src/embeddings/candle_embedder.rs).
   * Producto punto optimizado con desenrollado de bucle (auto-vectorizado en registros AVX2) para comparar cientos de miles de vectores por milisegundo.

3. **Índice Léxico Invertido BM25 ([`sparse_bm25.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-5-proyecto-final/proyecto_01_rag_engine/src/search/sparse_bm25.rs)):**
   * Tokenización alfanumérica y postings en memoria.
   * Puntuación BM25 canónica:
     $$\text{BM25}(D, Q) = \sum_{q \in Q} \ln\left(1 + \frac{N - n(q) + 0.5}{n(q) + 0.5}\right) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

4. **Reciprocal Rank Fusion ([`hybrid_fusion.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-5-proyecto-final/proyecto_01_rag_engine/src/search/hybrid_fusion.rs)):**
   * Algoritmo de agregación que combina posiciones ordinales en vez de scores crudos, eliminando la necesidad de calibración heurística de escalas.

5. **Cross-Encoder Reranker ([`cross_encoder.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-5-proyecto-final/proyecto_01_rag_engine/src/reranking/cross_encoder.rs)):**
   * Evalúa la interacción profunda entre la consulta y el texto candidato mediante una red neuronal profunda en Candle.

---

## 3. Especificación de la API REST (Axum)

### A. Healthcheck y Estadísticas
* **Ruta:** `GET /health`
* **Respuesta:**
```json
{
  "status": "healthy",
  "uptime_secs": 12,
  "stats": {
    "total_documents": 4,
    "total_chunks": 4,
    "embedding_dim": 64,
    "dense_vectors_indexed": 4,
    "bm25_terms_indexed": 96
  }
}
```

### B. Ingesta Dinámica de Documentos
* **Ruta:** `POST /api/v1/documents`
* **Body:**
```json
{
  "documents": [
    {
      "id": "doc_arch_1",
      "title": "Arquitectura de Inferencia",
      "content": "El uso de Rust en pipelines de serving reduce la latencia P99 por debajo de 5 milisegundos.",
      "metadata": {
        "author": "AI Team",
        "env": "production"
      }
    }
  ]
}
```

### C. Búsqueda Híbrida Multimodal
* **Ruta:** `POST /api/v1/search`
* **Modos:** `"dense"`, `"sparse"`, `"hybrid"`, `"hybridwithrerank"`
* **Body:**
```json
{
  "query": "latencia P99 en Rust",
  "top_k": 3,
  "mode": "hybridwithrerank"
}
```
* **Respuesta:**
```json
{
  "query": "latencia P99 en Rust",
  "total_hits": 1,
  "mode": "hybridwithrerank",
  "duration_ms": 1.42,
  "hits": [
    {
      "chunk_id": "doc_arch_1_chunk_0",
      "doc_id": "doc_arch_1",
      "doc_title": "Arquitectura de Inferencia",
      "text": "El uso de Rust en pipelines de serving reduce la latencia P99 por debajo de 5 milisegundos.",
      "final_score": 0.892,
      "metadata": {
        "env": "production"
      }
    }
  ]
}
```

---

## 4. Ejecución y Validación

```bash
# Entrar al workspace de la Fase 5
cd /home/alcidesticlla/Documents/rust-ia/fase-5-proyecto-final

# Ejecutar el suite completo de pruebas de integración (5/5 tests pasando)
cargo test

# Iniciar el servidor RAG en vivo en el puerto 3000
cargo run -p proyecto_01_rag_engine
```
