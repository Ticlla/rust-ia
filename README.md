# 🦀 Plan de Aprendizaje: Rust para Ingenieros en IA

> **Objetivo**: Pasar de cero conocimiento de Rust a poder construir componentes de producción para sistemas de IA — en aproximadamente 6 meses a ritmo moderado (1–2 horas/día).

---

## 🗺️ Vista General del Camino

```
FASE 1          FASE 2           FASE 3            FASE 4           FASE 5
Fundamentos  →  Sistemas     →   Datos & Numerics → IA & ML      →  Proyecto Final
(4 semanas)     (4 semanas)      (4 semanas)        (6 semanas)      (6 semanas)
```

---

## 📦 FASE 1 — Fundamentos de Rust (Semanas 1–4)

### Objetivos
Entender el modelo mental único de Rust antes de escribir cualquier cosa seria.

### Conceptos clave

| Concepto | Por qué importa en IA |
|---|---|
| **Ownership & Borrowing** | Gestión de buffers de datos sin GC |
| **Lifetimes** | Datos que viven en GPU vs CPU |
| **Traits** | Abstracción similar a protocolos de ML |
| **Enums + Pattern Matching** | Manejo de resultados de modelos |
| **Error handling** (`Result`, `Option`) | Pipelines robustos de inferencia |
| **Iteradores y closures** | Procesamiento funcional de datasets |

### Recursos

- 📖 **[The Rust Book](https://doc.rust-lang.org/book/)** — gratuito, oficial, el mejor punto de partida
- 🎮 **[Rustlings](https://github.com/rust-lang/rustlings)** — ejercicios interactivos en la terminal
- 🎥 **[Let's Get Rusty](https://www.youtube.com/@letsgetrusty)** — canal de YouTube muy claro

### Ejercicios de la fase

```rust
// Semana 1: Hola Mundo y tipos básicos
fn main() {
    let nombre = "Alcides";
    let version: u32 = 1;
    println!("Bienvenido a Rust, {nombre} v{version}!");
}

// Semana 2: Ownership — la base de todo
fn calcular_media(datos: Vec<f64>) -> f64 {
    let suma: f64 = datos.iter().sum();
    suma / datos.len() as f64
    // datos es devuelto al final, sin copia
}

// Semana 3: Structs y Traits
trait Modelo {
    fn predecir(&self, entrada: &[f64]) -> f64;
    fn nombre(&self) -> &str;
}

struct RegresionLineal {
    peso: f64,
    sesgo: f64,
}

impl Modelo for RegresionLineal {
    fn predecir(&self, entrada: &[f64]) -> f64 {
        entrada[0] * self.peso + self.sesgo
    }
    fn nombre(&self) -> &str { "Regresion Lineal" }
}

// Semana 4: Manejo de errores
use std::num::ParseFloatError;

fn parsear_datos(raw: &str) -> Result<Vec<f64>, ParseFloatError> {
    raw.split(',')
       .map(|s| s.trim().parse::<f64>())
       .collect()
}
```

### ✅ Checkpoint Fase 1
- [ ] Completar Rustlings (94 ejercicios)
- [ ] Leer capítulos 1–10 del Rust Book
- [x] Implementar una regresión lineal desde cero (completado en Fase 1)

---

## ⚙️ FASE 2 — Sistemas y Concurrencia (Semanas 5–8)

### Objetivos
Entender cómo Rust maneja threads, async, y estructuras de datos de bajo nivel — crítico para pipelines de datos paralelos.

### Conceptos clave

| Concepto | Aplicación en IA |
|---|---|
| **Threads y `Arc<Mutex<T>>`** | Entrenamiento paralelo, data loading |
| **`async`/`await` + Tokio** | Servidores de inferencia no bloqueantes |
| **Canales (`mpsc`)** | Pipelines productor-consumidor de datos |
| **`unsafe` Rust** | Bindings a CUDA, operaciones de bajo nivel |
| **Memoria: Stack vs Heap** | Optimización de buffers de tensores |

### Recursos

- 📖 **[Rust Async Book](https://rust-lang.github.io/async-book/)** — gratuito
- 🎥 **[Jon Gjengset - Crust of Rust](https://www.youtube.com/@jonhoo)** — profundo y técnico
- 🛠️ **[Tokio Tutorial](https://tokio.rs/tokio/tutorial)** — el runtime async más usado

### Proyecto de la fase

```rust
// Mini data loader paralelo con canales
use std::sync::mpsc;
use std::thread;

fn cargar_batch(archivo: &str, tx: mpsc::Sender<Vec<f64>>) {
    // Simular carga de datos
    let datos = vec![1.0, 2.0, 3.0, 4.0];
    tx.send(datos).unwrap();
}

fn main() {
    let (tx, rx) = mpsc::channel();
    let archivos = vec!["batch_1.csv", "batch_2.csv", "batch_3.csv"];

    for archivo in archivos {
        let tx_clone = tx.clone();
        let archivo = archivo.to_string();
        thread::spawn(move || cargar_batch(&archivo, tx_clone));
    }
    drop(tx); // cerrar canal original

    for batch in rx {
        println!("Batch recibido con {} elementos", batch.len());
        // procesar batch...
    }
}
```

### ✅ Checkpoint Fase 2
- [ ] Construir un servidor HTTP simple con Tokio + Axum
- [ ] Implementar un pipeline paralelo de carga de datos
- [ ] Entender cuándo usar `Arc<Mutex<T>>` vs canales

---

## 📊 FASE 3 — Datos y Numérica (Semanas 9–12)

### Objetivos
Trabajar con datos reales: DataFrames, álgebra lineal, serialización — el pan de cada día en IA.

### Librerías clave

| Librería | Equivalente Python | Uso |
|---|---|---|
| **Polars** | Pandas | DataFrames de alto rendimiento |
| **ndarray** | NumPy | Arrays N-dimensionales |
| **nalgebra** | — | Álgebra lineal |
| **serde** | json/pickle | Serialización de datos y modelos |
| **csv** | csv | Lectura de datasets |
| **rayon** | multiprocessing | Paralelismo de datos |

### Recursos

- 📖 **[Polars Docs](https://docs.pola.rs/)** — muy bien escrita
- 📖 **[ndarray docs](https://docs.rs/ndarray)** — referencia esencial
- 🛠️ **[Are We Learning Yet?](https://www.arewelearningyet.com/)** — ecosistema ML en Rust

### Ejercicios

```rust
// Polars — análisis de dataset de IA
use polars::prelude::*;

fn analizar_dataset(ruta: &str) -> PolarsResult<()> {
    let df = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(ruta.into()))?
        .finish()?;

    // Estadísticas descriptivas
    println!("{}", df.describe(None)?);

    // Filtrar y transformar — como pandas pero más rápido
    let filtrado = df
        .lazy()
        .filter(col("accuracy").gt(0.9))
        .select([col("model_name"), col("accuracy"), col("latency_ms")])
        .sort(["accuracy"], SortMultipleOptions::default())
        .collect()?;

    println!("{}", filtrado);
    Ok(())
}

// ndarray — operaciones matriciales
use ndarray::Array2;

fn multiplicar_matrices() {
    let a = Array2::<f64>::ones((128, 256));  // batch_size x features
    let b = Array2::<f64>::ones((256, 10));   // features x num_clases
    let resultado = a.dot(&b);               // 128 x 10
    println!("Forma del resultado: {:?}", resultado.shape());
}
```

### ✅ Checkpoint Fase 3
- [x] Reemplazar procesamiento con Polars y medir speedup (Clase 02)
- [x] Implementar multiplicación matricial y SIMD AVX2 de bajo nivel (Clases 01 y 05)
- [x] Serializar y deserializar pesos con `serde`, `bincode` y `safetensors` (Clase 03)
- [x] Carga cero-copia de modelos gigantes con `memmap2` (Clase 04)
- [x] Motor de Búsqueda Vectorial Híbrido RAG de alto rendimiento (Clase 06)

---

## 🤖 FASE 4 — IA y Machine Learning en Rust (Semanas 13–18)

### Objetivos
Construir y desplegar modelos reales usando el ecosistema de IA de Rust.

### Ecosistema IA en Rust

```
┌─────────────────────────────────────────────────────┐
│                  CAPA DE APLICACIÓN                  │
│         Axum (API REST)  +  Tokio (async)            │
├─────────────────────────────────────────────────────┤
│                  CAPA DE MODELOS                     │
│   Candle (HuggingFace)  │  ort (ONNX Runtime)        │
│   linfa (ML clásico)    │  burn (framework general)  │
├─────────────────────────────────────────────────────┤
│                  CAPA DE DATOS                       │
│         Polars  │  ndarray  │  arrow2                │
├─────────────────────────────────────────────────────┤
│                  CAPA DE SISTEMA                     │
│         rayon  │  tokio  │  serde  │  unsafe/CUDA    │
└─────────────────────────────────────────────────────┘
```

### Sub-fases

#### 4A — ML Clásico con `linfa` (Semana 13–14)

```rust
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::array;

fn regresion_con_linfa() {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

    let dataset = Dataset::new(x, y);
    let modelo = LinearRegression::default().fit(&dataset).unwrap();

    let prediccion = modelo.predict(array![[6.0]]);
    println!("Predicción para x=6: {:?}", prediccion);
}
```

#### 4B — Inferencia con ONNX (`ort`) (Semana 15–16)

```rust
// Cargar un modelo entrenado en PyTorch/TF y correrlo en Rust
use ort::{Environment, SessionBuilder, Value};

fn inferencia_onnx() -> ort::Result<()> {
    let env = Environment::builder()
        .with_name("mi_modelo_ia")
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&env)?
        .with_model_from_file("modelo.onnx")?;

    // Crear tensor de entrada
    let input = ndarray::array![[1.0f32, 2.0, 3.0, 4.0]];
    let outputs = session.run(vec![Value::from_array(input)?])?;

    println!("Resultado: {:?}", outputs[0]);
    Ok(())
}
```

#### 4C — LLMs con `candle` (HuggingFace) (Semana 17–18)

```rust
// Inferencia de un modelo de lenguaje con Candle
use candle_core::{Device, Tensor};
use candle_transformers::models::bert::BertModel;

fn embeddings_con_candle() -> candle_core::Result<()> {
    let device = Device::Cpu; // o Device::Cuda(0) si tienes GPU

    // Cargar modelo BERT
    // Los tokens vendrían del tokenizador de HuggingFace
    let input_ids = Tensor::new(&[[101u32, 7592, 1010, 2088, 102]], &device)?;

    println!("Tensor de entrada: {:?}", input_ids.shape());
    Ok(())
}
```

### Recursos

- 📦 **[Candle](https://github.com/huggingface/candle)** — HuggingFace ML en Rust
- 📦 **[linfa](https://github.com/rust-ml/linfa)** — ML clásico (como scikit-learn)
- 📦 **[burn](https://github.com/tracel-ai/burn)** — framework de deep learning
- 📦 **[ort](https://github.com/pykeio/ort)** — ONNX Runtime bindings

### ✅ Checkpoint Fase 4 (100% Completada)
- [x] Machine Learning Clásico con `linfa` (Clase 01)
- [x] Inferencia de Modelos Exportados con ONNX Runtime `ort` (Clase 02)
- [x] Tokenización de Alto Rendimiento con `tokenizers` de Hugging Face (Clase 03)
- [x] Deep Learning Nativo con `Candle` (Clase 04)
- [x] Arquitectura Transformer y Memoria KV-Cache (Clase 05)
- [x] Checkpoint Integrador: Servidor de Inferencia LLM con Token Streaming en Vivo (Clase 06)

---

## 🚀 FASE 5 — Set de Proyectos Production-Ready (Sistemas Industriales de IA)

### Objetivo
Construir una suite de proyectos de nivel empresarial en Rust, diseñados para producción real, con arquitecturas asíncronas no bloqueantes, tipado estricto, gestión eficiente de memoria y rendimiento superior a implementaciones tradicionales en Python.

### Proyectos de la Fase 5

#### 1. 🔍 Enterprise Hybrid RAG Engine
* **Motor:** Búsqueda Híbrida Vectorial (Densa con `Candle` Embeddings + Léxica BM25).
* **Stack:** `Axum`, `Tokio`, `Polars`, `Candle`, SIMD AVX2.
* **Características:** Ingesta asíncrona concurrente, chunking inteligente, reranking neuronal, y cache de vectores en memoria.

#### 2. ⚡ High-Throughput LLM Serving Gateway (vLLM en Rust)
* **Motor:** Gateway de inferencia compatible con la especificación de OpenAI (`/v1/chat/completions`).
* **Stack:** `Axum`, `Candle`, `tokenizers`, `eventsource-stream`.
* **Características:** Continuous/Dynamic Batching, Paged KV-Cache para evitar fragmentación de VRAM/RAM, y Server-Sent Events (SSE) para Token Streaming en vivo.

#### 3. 🤖 Autonomous Agent & Multi-Agent Orchestration Framework
* **Motor:** Sistema de orquestación de agentes autónomos y multi-agente basado en el **Modelo de Actores**.
* **Stack:** `Tokio` (canales MPSC/Broadcast), `serde_json`, `schemars`, `async-trait`.
* **Características:**
  * **Agente Individual (ReAct Loop):** Bucle de Razonamiento -> Acción (Tool Call) -> Observación -> Reflexión.
  * **Tool Calling con Seguridad de Tipos:** Registro modular de herramientas con validación estricta de esquemas JSON y control de timeouts.
  * **Sistema Multi-Agente:** Enrutador Supervisor-Trabajadores, consenso distribuido y grafos de estado (State Graph / DAG) para resolución colaborativa de tareas complejas.
  * **Streaming de Eventos:** Emisión en tiempo real de pensamientos, llamadas a herramientas y decisiones finales.

#### 4. 👁️ Microservicio de Inferencia de Visión y Audio en Tiempo Real
* **Motor:** Pipeline de clasificación/detección de baja latencia.
* **Stack:** `ort` (ONNX Runtime), `image`, `ndarray`, `Axum`.
* **Características:** Session Pooling concurrente, buffers cero-copia y colas de inferencia con contrapresión (backpressure).

### ✅ Checkpoint Fase 5 (100% Completada)
- [x] Proyecto 1: Enterprise Hybrid RAG Engine implementado y testeado
- [x] Proyecto 2: Servidor LLM OpenAI-Compatible con SSE Streaming y Batching
- [x] Proyecto 3: Sistema Autónomo Multi-Agente con Tool Calling y Modelo de Actores
- [x] Proyecto 4: Microservicio de Visión/Audio de Ultra Baja Latencia
- [x] Suite de pruebas de integración completa para todos los proyectos (21/21 tests pasando)

---

## 🌐 FASE 6 — Deployment, Observabilidad y Publicación de Productos de Prueba

### Objetivo
Llevar los proyectos desarrollados en la Fase 5 al mundo real, implementando infraestructura de nivel industrial, monitoreo de métricas críticas y desplegando productos de prueba funcionales y accesibles públicamente en la nube.

### Módulos de la Fase 6

#### 1. 🐳 Empaquetado y Optimización de Contenedores
* **Docker Multi-Stage con `cargo-chef`:** Cacheo de dependencias para builds reproducibles en segundos.
* **Binarios Estáticos Ultraligeros:** Compilación con target `x86_64-unknown-linux-musl` en imágenes `scratch` o `distroless` (peso final < 30 MB vs 4 GB de Python/PyTorch).

#### 2. 📊 Observabilidad y Métricas de Serving
* **Prometheus & Grafana:** Métricas en vivo de TTFT (Time To First Token), TPOT (Time Per Output Token), uso de KV-Cache y latencias P50/P95/P99.
* **OpenTelemetry & Tracing:** Rastreo distribuido extremo a extremo de peticiones HTTP, inferencia y llamadas de herramientas en agentes.

#### 3. 🛡️ Resiliencia y Pruebas de Carga
* **Stress Testing:** Pruebas de estrés masivo con `k6` / `wrk` evaluando cientos de conexiones concurrentes y saturación de hardware.
* **Políticas de Resiliencia:** Rate limiting con `governor`, Circuit Breakers y Graceful Shutdown en Rust para preservar streams activos durante reinicios.

#### 4. 🚀 CI/CD y Publicación de Productos de Prueba en Vivo
* **GitHub Actions:** Pipeline automatizado de linting (`clippy`), tests, auditoría de vulnerabilidades (`cargo audit`) y compilación de imágenes Docker.
* **Despliegue Público en la Nube:** Despliegue de al menos uno de los productos de prueba en infraestructura cloud (Fly.io, Hetzner, AWS o VPS) con proxy Caddy/Nginx, SSL automático y un frontend/API interactivo público.

### ✅ Checkpoint Fase 6
- [x] Dockerfiles multi-stage optimizados con `cargo-chef` (< 40 MB por servicio)
- [x] Endpoint `/metrics` con métricas Prometheus, dashboards en Grafana y logs en ELK
- [x] Resiliencia con Token Bucket Rate Limiting (HTTP 429) y Graceful Shutdown
- [x] Pipeline de CI/CD automatizado en GitHub Actions (`.github/workflows/ci.yml`)
- [x] Orquestación cloud-native en Kubernetes con HPA elástico y Probes de salud
- [ ] Reporte formal de pruebas de carga con `k6` demostrando estabilidad P99
- [ ] Producto de prueba publicado en la nube pública y accesible vía Internet con SSL

---

## 📅 Resumen del Plan Completo

| Fase | Enfoque | Entregable Principal | Estado |
|---|---|---|:---:|
| **1 — Fundamentos** | Ownership, Lifetimes, Traits, Tipos | Regresión lineal y evaluador de expresiones | ✅ Completado |
| **2 — Sistemas y Concurrencia** | Tokio, Canales, Arc/Mutex, Async | Microservicio de inferencia concurrente | ✅ Completado |
| **3 — Datos y Numérica** | Polars, ndarray, safetensors, SIMD AVX2, mmap | Pipeline de datos y motor RAG local SIMD | ✅ Completado |
| **4 — IA y Machine Learning** | Linfa, ONNX (`ort`), Tokenizers, Candle, KV-Cache | Servidor LLM con Token Streaming en Vivo | ✅ Completado |
| **5 — Proyectos Production-Ready** | RAG Híbrido, vLLM Gateway, Multi-Agente, Visión/Audio Edge | Suite de 4 proyectos empresariales completos (21 tests) | ✅ Completado |
| **6 — Deployment y Operaciones** | Docker chef, Observabilidad, CI/CD, Kubernetes HPA | **Infraestructura Cloud y Despliegue en K8s** | 🟡 En curso (~85%) |

---

## 🛠️ Setup inicial (hazlo hoy)

```bash
# 1. Instalar Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Instalar Rustlings (ejercicios interactivos)
cargo install rustlings
rustlings init
cd rustlings
rustlings watch   # ← empieza aquí

# 3. Tu primer proyecto de IA
cargo new mi-ia-rust
cd mi-ia-rust

# Agregar dependencias en Cargo.toml:
# [dependencies]
# polars = { version = "0.43", features = ["csv", "lazy"] }
# ndarray = "0.16"
# serde = { version = "1", features = ["derive"] }
# tokio = { version = "1", features = ["full"] }
```

---

## 📚 Recursos Adicionales

### Libros
- 📖 [The Rust Programming Language](https://doc.rust-lang.org/book/) — gratuito
- 📖 [Rust for Rustaceans](https://nostarch.com/rust-rustaceans) — nivel avanzado
- 📖 [Programming Rust (O'Reilly)](https://www.oreilly.com/library/view/programming-rust-2nd/9781492052586/)

### Comunidad
- 💬 [r/rust](https://www.reddit.com/r/rust/) — muy activa y amable
- 💬 [Rust Discord](https://discord.gg/rust-lang) — ayuda en tiempo real
- 🐦 Seguir a **@jonhoo**, **@fasterthanlime** en redes

### Canales YouTube
- 🎥 [Let's Get Rusty](https://www.youtube.com/@letsgetrusty)
- 🎥 [Jon Gjengset](https://www.youtube.com/@jonhoo) — nivel avanzado
- 🎥 [No Boilerplate](https://www.youtube.com/@NoBoilerplate) — conceptos rápidos

---

> **Pro tip**: No trates de aprender Rust "perfectamente" antes de usarlo en IA. Empieza a usar **Polars** desde la Fase 1 para tareas cotidianas — aprenderás el lenguaje mientras haces cosas útiles.
