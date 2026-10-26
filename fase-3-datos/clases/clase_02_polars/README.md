# Clase 02: Procesamiento Masivo con `Polars` y Apache Arrow 📊⚡

En Machine Learning e Inteligencia Artificial, la mayor parte del tiempo de ingeniería no se gasta entrenando el modelo, sino **limpiando, transformando y preparando los datasets tabulares** (Feature Engineering).

En el ecosistema Python, la herramienta tradicional ha sido **Pandas**. Sin embargo, Pandas es monohilo, ineficiente en memoria y suele fallar por Out-Of-Memory (*OOM*) con datasets de varios gigabytes.

La solución de última generación es **`Polars`**, un motor de DataFrames ultrarrápido desarrollado **100% en Rust** sobre el estándar **Apache Arrow**.

---

## 🎯 Objetivos de la Clase
1. Comprender la **memoria columnar de Apache Arrow**: por qué guardar datos por columnas es $10\times$ a $50\times$ más veloz que Pandas para operaciones analíticas.
2. Dominar la **API Perezosa (`LazyFrame`)**: cómo el optimizador de consultas reordena filtros y proyecciones (*Predicate & Projection Pushdown*) para minimizar el uso de RAM.
3. Aplicar **Feature Engineering para IA**:
   - Imputación de nulos (`.fill_null(...)`).
   - Normalización Min-Max de columnas continuas.
   - Creación de características calculadas vectorialmente con `col(...)`.
4. Construir el **puente entre Polars y `ndarray`**: extraer matrices numéricas limpias para alimentar capas densas de redes neuronales.

---

## 🏗️ Comparativa: Pandas vs Polars en Rust

| Característica | Python (Pandas) | Rust (`Polars`) |
|---|---|---|
| **Formato en RAM** | Fila por fila con objetos de Python | Columnar continuo con **Apache Arrow** |
| **Paralelismo** | Monohilo (limitado por el GIL) | Multihilo nativo automático con **Rayon** |
| **Optimización de Consultas** | Ninguna (ejecución paso a paso inmediata) | **Query Optimizer** (Predicate & Projection Pushdown) |
| **Consumo de Memoria** | $5\times$ a $10\times$ el tamaño del archivo | Casi idéntico al tamaño en disco |
| **Valores Faltantes** | `NaN` flotante (corrompe enteros) | Bitmaps de validez Arrow (mantiene tipos) |

---

## 🚀 Comandos de Ejecución

```bash
# 1. Demostración teórica de DataFrames, LazyFrames y el puente a ndarray:
cargo run --bin c02_teoria

# 2. Laboratorio de errores comunes (antipatrón iterrows, eager vs lazy, nulos):
cargo run --bin c02_errores

# 3. Ejecutar la suite de pruebas unitarias:
cargo test --bin c02_ejercicios
```
