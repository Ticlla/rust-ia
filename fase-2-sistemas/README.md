# ⚙️ Fase 2: Sistemas y Concurrencia para Ingenieros en IA

Bienvenido a la **Fase 2**. En esta fase aprenderás a exprimir al 100% la capacidad de tu hardware: procesar datos en paralelo en múltiples núcleos del CPU, sincronizar memoria compartida sin *data races*, y construir servidores asíncronos de inferencia de ultra alta velocidad.

---

## 📅 Hoja de Ruta de Clases (Fase 2)

| Clase | Tema | Carpeta | Comandos de Ejecución |
|---|---|---|---|
| **Clase 01** | Hilos Nativos (`std::thread`), `join()` y `move` | [📂 `clase_01_hilos/`](file:///home/alcidesticlla/Documents/rust-ia/fase-2-sistemas/clases/clase_01_hilos/README.md) | `cargo run --bin c01_teoria`<br>`cargo run --bin c01_errores`<br>`cargo run --bin c01_ejercicios` |
| **Clase 02** | Canales (`mpsc`): Pipelines y DataLoaders | [📂 `clase_02_canales/`](file:///home/alcidesticlla/Documents/rust-ia/fase-2-sistemas/clases/clase_02_canales/README.md) | `cargo run --bin c02_teoria`<br>`cargo run --bin c02_errores`<br>`cargo run --bin c02_ejercicios` |
| **Clase 03** | Memoria Compartida: `Arc<T>` y `Mutex<T>` | [📂 `clase_03_memoria_compartida/`](file:///home/alcidesticlla/Documents/rust-ia/fase-2-sistemas/clases/clase_03_memoria_compartida/README.md) | `cargo run --bin c03_teoria`<br>`cargo run --bin c03_errores`<br>`cargo run --bin c03_ejercicios` |
| **Clase 04** | Lecturas Concurrentes con `RwLock<T>` | [📂 `clase_04_rwlock/`](file:///home/alcidesticlla/Documents/rust-ia/fase-2-sistemas/clases/clase_04_rwlock/README.md) | `cargo run --bin c04_teoria`<br>`cargo run --bin c04_errores`<br>`cargo run --bin c04_ejercicios` |
| **Clase 05** | Paralelismo de Datos con `Rayon` | [📂 `clase_05_rayon/`](file:///home/alcidesticlla/Documents/rust-ia/fase-2-sistemas/clases/clase_05_rayon/README.md) | `cargo run --bin c05_teoria`<br>`cargo run --bin c05_errores`<br>`cargo run --bin c05_ejercicios` |
| **Clase 06** | Asincronía Básica (`async` / `await`) | [📂 `clase_06_asincronia_basica/`](file:///home/alcidesticlla/Documents/rust-ia/fase-2-sistemas/clases/clase_06_asincronia_basica/README.md) | `cargo run --bin c06_teoria`<br>`cargo run --bin c06_errores`<br>`cargo run --bin c06_ejercicios` |
| **Clase 07** | Runtime Asíncrono con `Tokio` | [📂 `clase_07_tokio/`](file:///home/alcidesticlla/Documents/rust-ia/fase-2-sistemas/clases/clase_07_tokio/README.md) | `cargo run --bin c07_teoria`<br>`cargo run --bin c07_errores`<br>`cargo run --bin c07_ejercicios` |
| **Clase 08** | Servidores de Inferencia con `Axum` | [📂 `clase_08_axum/`](file:///home/alcidesticlla/Documents/rust-ia/fase-2-sistemas/clases/clase_08_axum/README.md) | `cargo run --bin c08_teoria`<br>`cargo run --bin c08_errores`<br>`cargo run --bin c08_ejercicios` |
| **Clase 09** | Punteros Crudos y `unsafe` Rust (C / CUDA) | [📂 `clase_09_unsafe_cuda/`](file:///home/alcidesticlla/Documents/rust-ia/fase-2-sistemas/clases/clase_09_unsafe_cuda/README.md) | `cargo run --bin c09_teoria`<br>`cargo run --bin c09_errores`<br>`cargo run --bin c09_ejercicios` |
| **Clase 10** | Checkpoint Fase 2: Pipeline DataLoader + API IA | [📂 `clase_10_proyecto/`](file:///home/alcidesticlla/Documents/rust-ia/fase-2-sistemas/clases/clase_10_proyecto/README.md) | `cargo run --bin c10_proyecto`<br>`cargo run --bin c10_servidor`<br>`cargo run --bin c10_errores`<br>`cargo test --bin c10_ejercicios` |

---

## 🚀 Comandos Rápidos

```bash
# Ver el catálogo en la terminal:
cargo run

# Probar la teoría de la Clase 01:
cargo run --bin c01_teoria

# Ejecutar las pruebas unitarias:
cargo test
```
