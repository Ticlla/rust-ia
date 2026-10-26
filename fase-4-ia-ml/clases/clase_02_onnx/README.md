# Clase 02: Inferencia de Modelos con ONNX Runtime (`ort`) ⚡🧠

En esta clase aprenderemos a salvar la brecha entre el entrenamiento en Python (PyTorch/TensorFlow) y el despliegue de alta eficiencia en Rust usando el estándar abierto **ONNX (Open Neural Network Exchange)** y el motor **ONNX Runtime (`ort`)**.

---

## 🎯 Objetivos de la Clase

1. **El Estándar ONNX:** Por qué exportar grafos de PyTorch a `.onnx` elimina el GIL de Python y el consumo excesivo de RAM en servidores de producción.
2. **Ciclo de Inferencia con `ort`:**
   * Inicialización del entorno (`ort::init`).
   * Configuración de la sesión (`Session::builder`).
   * Opciones de optimización de grafo (fusión de operadores).
3. **Mapeo de Tensores Rust ➔ ONNX:** Creación de `ort::value::Tensor` a partir de datos nativos sin copias lentas.
4. **Dynamic Batching:** Ejecutar inferencias para lotes de tamaño variable (batch 1 vs batch 32) usando dimensiones dinámicas `[-1, D]`.
5. **Benchmarks de Latencia:** Medición de tiempo de ejecución por inferencia en microsegundos.

---

## 📂 Archivos de la Clase

* [`01_teoria.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-4-ia-ml/clases/clase_02_onnx/01_teoria.rs): Creación de sesión, inspección de entradas/salidas y ejecución de inferencias con lotes dinámicos.
* [`02_errores_comunes.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-4-ia-ml/clases/clase_02_onnx/02_errores_comunes.rs): Nombres de entradas no coincidentes, discrepancias de tipo de dato (f32 vs f64) y dimensiones fijas vs dinámicas.
* [`03_ejercicios.rs`](file:///home/alcidesticlla/Documents/rust-ia/fase-4-ia-ml/clases/clase_02_onnx/03_ejercicios.rs): Tests automatizados para validar carga de modelos, Dynamic Batching y extracción de predicciones.
