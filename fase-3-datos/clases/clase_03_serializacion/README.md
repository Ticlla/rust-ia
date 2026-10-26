# Clase 03: Serialización Binaria y Formatos de Modelos de IA (`safetensors` y `bincode`) 💾📦

Cuando terminas de entrenar un modelo, de calcular millones de embeddings o de ajustar hiperparámetros, necesitas **guardar los pesos en el disco duro** y poder cargarlos en producción en cuestión de milisegundos.

Tradicionalmente, en Python se usaba `pickle` (`model.pkl`) o texto en `JSON`. Ambos formatos son una pesadilla en la IA moderna:
* **JSON:** Infla el tamaño en disco hasta un $300\%$, pierde precisión en los números flotantes y es lentísimo de parsear.
* **Pickle:** Es un agujero de seguridad crítico (permite ejecutar código remoto malicioso - *RCE*).

La solución estándar de la industria hoy fue creada por **Hugging Face en Rust**: el formato **`safetensors`**.

---

## 🎯 Objetivos de la Clase
1. Comprender la diferencia entre **formatos de texto (JSON)** vs **formatos binarios exactos (Bincode / SafeTensors)**.
2. Usar **`bincode`** para guardar y restaurar estructuras completas de Rust (checkpoints, épocas, configuraciones).
3. Dominar el formato **`safetensors`** de Hugging Face:
   - Encabezado JSON con metadatos + bytes contiguos de tensores en memoria.
   - Seguridad absoluta (cero ejecución de código malicioso).
   - Lectura Cero-Copias (*Zero-Copy*).
4. Conectar SafeTensors con **`ndarray`**: leer pesos directamente hacia matrices `Array2<f32>` listas para inferir.

---

## 🏗️ Comparativa de Formatos para Pesos de IA

| Característica | JSON | Python Pickle (`.pkl`) | SafeTensors (Rust / Hugging Face) |
|---|---|---|---|
| **Velocidad de Carga** | Lenta (parseo de texto) | Media | **Instantánea (Cero-Copias)** |
| **Seguridad** | Segura | ❌ **Peligrosa (Ejecuta código)** | 🛡️ **100% Segura (Solo datos)** |
| **Precisión Flotante** | Pérdida por redondeo | Exacta | **Exacta bit a bit (IEEE-754)** |
| **Tamaño en Disco** | Muy grande (strings) | Binario | **Binario compacto** |
| **Estándar en Hugging Face**| No | Obsoleto | **El estándar oficial global** |

---

## 🚀 Comandos de Ejecución

```bash
# 1. Demostración teórica de Bincode y SafeTensors:
cargo run --bin c03_teoria

# 2. Laboratorio de errores (JSON drift, ataques de Pickle y cero-copia):
cargo run --bin c03_errores

# 3. Ejecutar las pruebas unitarias automatizadas:
cargo test --bin c03_ejercicios
```
