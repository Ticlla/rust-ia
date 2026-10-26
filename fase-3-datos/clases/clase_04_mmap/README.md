# Clase 04: Memoria Mapeada (`mmap`) para Modelos Gigantes ⚡🧠

En Inteligencia Artificial moderna, los modelos como Llama 3 (70B), Qwen (72B) o DeepSeek pesan entre **40 GB y más de 140 GB**.

Si un servidor intentara leer un modelo de 70 GB con `std::fs::read()`, la computadora necesitaría reservar 70 GB continuos en la memoria RAM de golpe, provocando un fallo por **Out-Of-Memory (OOM)** y tardando decenas de segundos en arrancar.

La solución de grado industrial que usan frameworks como **`vLLM`**, **`llama.cpp`** y **`Candle`** para arrancar modelos masivos en **0.001 segundos** es la **Memoria Mapeada (`mmap`)**.

---

## 🎯 Objetivos de la Clase
1. Comprender la llamada al kernel de Linux **`mmap`** y la **Paginación Virtual**:
   - Cómo el sistema operativo crea un mapa virtual entre el proceso y el archivo en el disco SSD NVMe.
   - Paginación bajo demanda (*Page Faults*): la CPU solo carga a la RAM las páginas de 4 KB en el instante exacto en que las necesita.
2. Integrar **`mmap` con `safetensors`**:
   - Abrir archivos gigantescos en tiempo récord.
   - Cero copias: los punteros a las matrices apuntan directamente a la memoria virtual del archivo en disco.
3. Ejecutar **Inferencia Directa** sin precargar el modelo en la RAM:
   - Calcular productos matriciales ($Y = X \cdot W$) consumiendo los pesos directamente desde el archivo mapeado.

---

## 🏗️ Comparativa: Carga Tradicional vs Memoria Mapeada (`mmap`)

| Métrica | Carga Tradicional (`std::fs::read`) | Memoria Mapeada (`mmap`) |
|---|---|---|
| **Tiempo de Arranque (14 GB)** | 15 a 30 segundos (copiando bytes) | **0.001 segundos (Instantáneo)** |
| **Consumo Inicial de RAM** | 14 GB reservados en el Heap | **~0 MB (Bajo demanda)** |
| **Límite de Tamaño de Modelo** | Limitado por tu RAM física (ej. 16 GB)| **Limitado solo por el tamaño del SSD** |
| **Mecanismo del Kernel** | Copia al espacio de usuario | Paginación por hardware (MMU / Page Cache) |

---

## 🚀 Comandos de Ejecución

```bash
# 1. Demostración teórica de mmap + safetensors:
cargo run --bin c04_teoria

# 2. Laboratorio de errores (OOM, riesgos de SIGBUS y gestión de ciclos de vida):
cargo run --bin c04_errores

# 3. Ejecutar suite de pruebas de inferencia directa sobre disco:
cargo test --bin c04_ejercicios
```
