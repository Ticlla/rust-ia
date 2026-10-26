# Clase 01: Machine Learning Clásico con `linfa` 🤖📊

En esta clase aprenderemos a construir, entrenar y evaluar algoritmos de Machine Learning tradicionales en Rust utilizando el ecosistema **`linfa`** (el equivalente a Scikit-Learn en Rust).

---

## 🎯 Objetivos de la Clase

1. **La Estructura `Dataset`:** Comprender cómo `linfa` desacopla y relaciona las características (features en `Array2`) con las etiquetas objetivo (`targets`).
2. **El Patrón `Fit` y `Predict`:** El estándar tipado en Rust para entrenar modelos (`Fit`) y realizar inferencias (`Predict`).
3. **Regresión Lineal con `linfa-linear`:** Ajuste de coeficientes, cálculo de pendientes, sesgos e inferencia continua.
4. **Clustering con K-Means (`linfa-clustering`):** Agrupamiento no supervisado de datos y su aplicación directa en la indexación vectorial de IA (**IVFFlat**).
5. **Métricas de Evaluación:** Medición de calidad con Mean Squared Error (MSE).

---

## 📂 Archivos de la Clase

* [`01_teoria.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-4-ia-ml/clases/clase_01_linfa/01_teoria.rs): Fundamentos de `Dataset`, Regresión Lineal y K-Means en vivo.
* [`02_errores_comunes.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-4-ia-ml/clases/clase_01_linfa/02_errores_comunes.rs): Desalineación de formas matriciales, tipos flotantes incompatibles y trampa de centroides vacíos.
* [`03_ejercicios.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-4-ia-ml/clases/clase_01_linfa/03_ejercicios.rs): Tests automatizados para validar el flujo completo de entrenamiento y predicción.
