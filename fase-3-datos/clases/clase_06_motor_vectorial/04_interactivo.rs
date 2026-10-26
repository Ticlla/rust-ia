//! ==============================================================================
//! CLASE 06: LABORATORIO INTERACTIVO - Motor de Búsqueda Vectorial Híbrido (RAG)
//! ==============================================================================
//!
//! Este laboratorio interactivo demuestra cómo opera un motor de búsqueda vectorial
//! real (similar a Qdrant, LanceDB o Milvus) con un corpus amplio de 24 documentos
//! técnicos y capacidad de benchmark contra 10,000 vectores en disco.
//!
//! ARQUITECTURA:
//! 1. Catálogo de Metadatos con Polars (ID, Título, Categoría, Contenido, Año, Tags).
//! 2. Embeddings normalizados de dimensión D = 32 guardados en `.safetensors`.
//! 3. Acceso Cero-Copia mediante `memmap2`.
//! 4. Pre-filtrado declarativo con `Polars` (LazyFrame).
//! 5. Scoring acelerado por hardware con SIMD AVX2 + FMA (`_mm256_fmadd_ps`).

use memmap2::Mmap;
use polars::prelude::*;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const DIMENSION: usize = 32; // 32 floats = exactamente 4 bloques AVX2 de 256 bits

/// Estructura de un Documento del Corpus
#[derive(Debug, Clone)]
pub struct Documento {
    pub id: i64,
    pub titulo: String,
    pub categoria: String,
    pub anio: i32,
    pub contenido: String,
    pub embedding: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("🔍 MOTOR DE BÚSQUEDA VECTORIAL HÍBRIDO (RAG) - CONSOLA INTERACTIVA");
    println!("================================================================================\n");

    // 1. Construir corpus amplio (24 documentos reales de Ingeniería de IA y Sistemas)
    let corpus = construir_corpus_tecnico();
    println!("📚 Corpus cargado: {} documentos técnicos estructurados.", corpus.len());

    // 2. Crear DataFrame de Polars para los metadatos
    let metadatos_df = crear_dataframe_metadatos(&corpus)?;
    println!("📊 Catálogo de metadatos registrado en memoria columnar (Apache Arrow).");

    // 3. Serializar matriz de embeddings a disco con safetensors
    let temp_dir = std::env::temp_dir();
    let safetensors_path = temp_dir.join("corpus_embeddings_interactivo.safetensors");
    guardar_matriz_safetensors(&safetensors_path, &corpus)?;
    println!("💾 Matriz de embeddings persistida en: {:?}\n", safetensors_path);

    // 4. Mapear a memoria virtual con memmap2 (cero copias de RAM)
    let file = File::open(&safetensors_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let safetensors = SafeTensors::deserialize(&mmap)?;
    let tensor = safetensors.tensor("embeddings")?;
    let bytes_crudos = tensor.data();
    let floats_mmap: &[f32] = unsafe {
        std::slice::from_raw_parts(
            bytes_crudos.as_ptr() as *const f32,
            bytes_crudos.len() / std::mem::size_of::<f32>(),
        )
    };
    println!("⚡ Archivo safetensors mapeado en memoria virtual con memmap2 (Cero Copia).");
    println!("   Dimensiones en disco: {:?}, Tipo: {:?}\n", tensor.shape(), tensor.dtype());

    // Bucle interactivo de consola
    ejecutar_menu_interactivo(&metadatos_df, floats_mmap)?;

    Ok(())
}

/// Menú interactivo para explorar el motor
fn ejecutar_menu_interactivo(
    metadatos: &DataFrame,
    embeddings_mmap: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        println!("--------------------------------------------------------------------------------");
        println!("SELECCIONA UNA OPERACIÓN DE BÚSQUEDA:");
        println!("  1. Búsqueda Semántica: 'Transformers y Mecanismo de Atención'");
        println!("  2. Búsqueda Semántica: 'Sistemas Rust, Zero-Copy y Seguridad de Memoria'");
        println!("  3. Búsqueda Semántica: 'Índices HNSW y Similitud Coseno en Vector DBs'");
        println!("  4. Búsqueda Semántica: 'Optimización de GPU, CUDA Kernels y FlashAttention'");
        println!("  5. Búsqueda Híbrida Filtrada: Solo categoría 'rust_sistemas' del año >= 2024");
        println!("  6. Benchmark de Estrés SIMD: 10,000 vectores sintéticos en memoria mapeada");
        println!("  7. Salir");
        print!("\n👉 Ingresa una opción (1-7) [Enter para opción 1]: ");
        io::stdout().flush()?;

        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input)?;

        // Si es fin de archivo (modo no interactivo / pipe), ejecutamos la opción 1 y el benchmark, y salimos
        if bytes_read == 0 {
            println!("1 (Modo demostración automática)");
            ejecutar_consulta_semantica(1, metadatos, embeddings_mmap, None)?;
            println!("\nEjecutando demostración híbrida filtrada (Opción 5):");
            ejecutar_consulta_semantica(2, metadatos, embeddings_mmap, Some("rust_sistemas"))?;
            println!("\nEjecutando benchmark de estrés SIMD (Opción 6):");
            ejecutar_benchmark_estres_simd()?;
            break;
        }

        let seleccion = input.trim();
        let opcion = if seleccion.is_empty() { 1 } else { seleccion.parse::<usize>().unwrap_or(1) };

        match opcion {
            1 => ejecutar_consulta_semantica(1, metadatos, embeddings_mmap, None)?,
            2 => ejecutar_consulta_semantica(2, metadatos, embeddings_mmap, None)?,
            3 => ejecutar_consulta_semantica(3, metadatos, embeddings_mmap, None)?,
            4 => ejecutar_consulta_semantica(4, metadatos, embeddings_mmap, None)?,
            5 => {
                println!("\n🔍 Aplicando pre-filtrado con Polars: Categoria = 'rust_sistemas'...");
                ejecutar_consulta_semantica(2, metadatos, embeddings_mmap, Some("rust_sistemas"))?;
            }
            6 => ejecutar_benchmark_estres_simd()?,
            7 => {
                println!("¡Hasta pronto! Has completado el laboratorio del Motor Vectorial.");
                break;
            }
            _ => println!("⚠️ Opción no válida. Por favor selecciona entre 1 y 7."),
        }
    }

    Ok(())
}

/// Ejecuta una consulta semántica y mide el tiempo exacto de resolución
fn ejecutar_consulta_semantica(
    query_id: usize,
    metadatos: &DataFrame,
    embeddings_mmap: &[f32],
    filtro_categoria: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (texto_query, mut query_vec) = match query_id {
        1 => (
            "¿Cómo funciona el mecanismo de auto-atención en los LLMs modernos?",
            generar_query_embedding_por_cluster(1), // Cluster LLM
        ),
        2 => (
            "Seguridad de memoria, ownership y rendimiento zero-cost en Rust",
            generar_query_embedding_por_cluster(0), // Cluster Rust
        ),
        3 => (
            "Estructura de grafos HNSW y búsqueda aproximada de vecinos más cercanos",
            generar_query_embedding_por_cluster(2), // Cluster Vector Search
        ),
        4 => (
            "Paralelismo masivo en GPUs con CUDA y optimización de memoria HBM",
            generar_query_embedding_por_cluster(3), // Cluster GPU
        ),
        _ => (
            "Consulta general de IA",
            generar_query_embedding_por_cluster(1),
        ),
    };

    normalizar_vector(&mut query_vec);

    println!("\n================================================================================");
    println!("🔎 CONSULTA: \"{}\"", texto_query);
    if let Some(cat) = filtro_categoria {
        println!("🏷️  PRE-FILTRO APLICADO (Polars): categoria == \"{}\"", cat);
    }
    println!("================================================================================");

    let t0 = Instant::now();

    // FASE 1: Filtrado de Metadatos con Polars (LazyFrame)
    let t_filtro = Instant::now();
    let mut lazy = metadatos.clone().lazy();
    if let Some(cat) = filtro_categoria {
        lazy = lazy.filter(col("categoria").eq(lit(cat)));
    }
    let candidatos_df = lazy
        .select([col("id"), col("titulo"), col("categoria"), col("anio"), col("contenido")])
        .collect()?;
    let duracion_filtro = t_filtro.elapsed();

    let total_candidatos = candidatos_df.height();
    let ids_ca = candidatos_df.column("id")?.i64()?;
    let titulos_ca = candidatos_df.column("titulo")?.str()?;
    let categorias_ca = candidatos_df.column("categoria")?.str()?;
    let anios_ca = candidatos_df.column("anio")?.i32()?;
    let contenidos_ca = candidatos_df.column("contenido")?.str()?;

    // FASE 2: Scoring con SIMD AVX2 sobre memoria mapeada
    let t_simd = Instant::now();
    struct MatchResultado<'a> {
        id: i64,
        titulo: &'a str,
        categoria: &'a str,
        anio: i32,
        contenido: &'a str,
        score: f32,
    }

    let mut resultados: Vec<MatchResultado> = Vec::with_capacity(total_candidatos);

    for i in 0..total_candidatos {
        if let Some(id) = ids_ca.get(i) {
            let offset = (id as usize) * DIMENSION;
            let doc_vec = &embeddings_mmap[offset..offset + DIMENSION];

            // Invocación del cálculo SIMD AVX2 acelerado por hardware:
            let score = producto_punto_simd_avx2(&query_vec, doc_vec);

            resultados.push(MatchResultado {
                id,
                titulo: titulos_ca.get(i).unwrap_or(""),
                categoria: categorias_ca.get(i).unwrap_or(""),
                anio: anios_ca.get(i).unwrap_or(0),
                contenido: contenidos_ca.get(i).unwrap_or(""),
                score,
            });
        }
    }

    // FASE 3: Ranking Top-K (ordenamiento por similitud descendente)
    resultados.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let duracion_simd = t_simd.elapsed();
    let duracion_total = t0.elapsed();

    // Mostrar Top 4 resultados
    let top_k = 4.min(resultados.len());
    println!("\n🏆 TOP {} DOCUMENTOS MÁS RELEVANTES ENCONTRADOS:", top_k);
    for (idx, doc) in resultados.iter().take(top_k).enumerate() {
        println!(
            "  [{}] Score: {:.4} | ID: {:2} | [{}] ({}) - \"{}\"",
            idx + 1,
            doc.score,
            doc.id,
            doc.categoria,
            doc.anio,
            doc.titulo
        );
        println!("      Snippet: \"{}\"\n", doc.contenido);
    }

    println!("⏱️  MÉTRICAS DE RENDIMIENTO:");
    println!("   • Candidatos evaluados:       {}/{}", total_candidatos, metadatos.height());
    println!("   • Tiempo pre-filtrado Polars: {:?}", duracion_filtro);
    println!("   • Tiempo scoring SIMD AVX2:   {:?}", duracion_simd);
    println!("   • Latencia total de búsqueda: {:?}", duracion_total);

    Ok(())
}

/// Benchmark masivo: Evalúa 10,000 vectores continuos en memoria mapeada con SIMD AVX2
fn ejecutar_benchmark_estres_simd() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n================================================================================");
    println!("⚡ BENCHMARK DE ESTRÉS: BÚSQUEDA EXHAUSTIVA SOBRE 10,000 VECTORES EN SSD");
    println!("================================================================================");

    let n_vectores = 10_000;
    println!("1. Generando matriz sintética de {} vectores x {} dimensiones...", n_vectores, DIMENSION);
    let mut matriz = vec![0.0f32; n_vectores * DIMENSION];
    for (i, val) in matriz.iter_mut().enumerate() {
        *val = ((i % 100) as f32) / 100.0;
    }
    // Normalizar cada vector
    for i in 0..n_vectores {
        normalizar_vector(&mut matriz[i * DIMENSION..(i + 1) * DIMENSION]);
    }

    let temp_file = std::env::temp_dir().join("stress_test_10k.safetensors");
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            matriz.as_ptr() as *const u8,
            matriz.len() * std::mem::size_of::<f32>(),
        )
    };
    let shape = vec![n_vectores, DIMENSION];
    let view = TensorView::new(Dtype::F32, shape, bytes)?;
    let mut map = HashMap::new();
    map.insert("stress_embeddings".to_string(), view);
    let serializado = safetensors::serialize(&map, None)?;
    std::fs::write(&temp_file, serializado)?;

    // Mapear con memmap2
    let file = File::open(&temp_file)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let t = st.tensor("stress_embeddings")?;
    let floats: &[f32] = unsafe {
        std::slice::from_raw_parts(
            t.data().as_ptr() as *const f32,
            t.data().len() / std::mem::size_of::<f32>(),
        )
    };

    let mut query = vec![0.5f32; DIMENSION];
    normalizar_vector(&mut query);

    println!("2. Ejecutando escaneo SIMD AVX2 sobre 10,000 vectores...");
    let t_start = Instant::now();
    let mut mejor_score = -1.0f32;
    let mut mejor_id = 0;

    for id in 0..n_vectores {
        let vec_doc = &floats[id * DIMENSION..(id + 1) * DIMENSION];
        let score = producto_punto_simd_avx2(&query, vec_doc);
        if score > mejor_score {
            mejor_score = score;
            mejor_id = id;
        }
    }
    let duracion = t_start.elapsed();

    let qps = 1.0 / duracion.as_secs_f64();
    println!("✓ Escaneo de 10,000 vectores completado:");
    println!("   • Tiempo total:       {:.3?}", duracion);
    println!("   • Mejor coincidencia: Vector #{} con score {:.4}", mejor_id, mejor_score);
    println!("   • Rendimiento:        {:.1} consultas completas por segundo (QPS)", qps);
    println!("   • Throughput:         {:.2} Millones de vectores comparados por segundo", (n_vectores as f64 / duracion.as_secs_f64()) / 1_000_000.0);

    let _ = std::fs::remove_file(temp_file);
    Ok(())
}

/// ----------------------------------------------------------------------------
/// NÚCLEO MATEMÁTICO Y DE HARDWARE (SIMD AVX2)
/// ----------------------------------------------------------------------------
pub fn producto_punto_simd_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { producto_punto_avx2_impl(a, b) };
        }
    }

    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
unsafe fn producto_punto_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let trozos = len / 8;
    let resto = len % 8;

    let mut acc = unsafe { _mm256_setzero_ps() };
    let mut ptr_a = a.as_ptr();
    let mut ptr_b = b.as_ptr();

    for _ in 0..trozos {
        unsafe {
            let va = _mm256_loadu_ps(ptr_a);
            let vb = _mm256_loadu_ps(ptr_b);
            acc = _mm256_fmadd_ps(va, vb, acc);

            ptr_a = ptr_a.add(8);
            ptr_b = ptr_b.add(8);
        }
    }

    let mut buf = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    }
    let mut suma: f32 = buf.iter().sum();

    let inicio_resto = trozos * 8;
    for i in 0..resto {
        suma += a[inicio_resto + i] * b[inicio_resto + i];
    }

    suma
}

pub fn normalizar_vector(v: &mut [f32]) {
    let mut suma_cuadrados = 0.0f32;
    for &x in v.iter() {
        suma_cuadrados += x * x;
    }
    let norma = suma_cuadrados.sqrt();
    if norma > 1e-12 {
        for x in v.iter_mut() {
            *x /= norma;
        }
    }
}

/// ----------------------------------------------------------------------------
/// CONSTRUCCIÓN DEL CORPUS DE 24 DOCUMENTOS Y MATRIZ SAFETENSORS
/// ----------------------------------------------------------------------------
fn guardar_matriz_safetensors(path: &Path, corpus: &[Documento]) -> Result<(), Box<dyn std::error::Error>> {
    let n = corpus.len();
    let mut flat_embeddings = Vec::with_capacity(n * DIMENSION);
    for doc in corpus {
        flat_embeddings.extend_from_slice(&doc.embedding);
    }

    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            flat_embeddings.as_ptr() as *const u8,
            flat_embeddings.len() * std::mem::size_of::<f32>(),
        )
    };

    let shape = vec![n, DIMENSION];
    let view = TensorView::new(Dtype::F32, shape, bytes)?;

    let mut tensors = HashMap::new();
    tensors.insert("embeddings".to_string(), view);

    let serialized = safetensors::serialize(&tensors, None)?;
    std::fs::write(path, serialized)?;
    Ok(())
}

fn crear_dataframe_metadatos(corpus: &[Documento]) -> Result<DataFrame, PolarsError> {
    let ids: Vec<i64> = corpus.iter().map(|d| d.id).collect();
    let titulos: Vec<&str> = corpus.iter().map(|d| d.titulo.as_str()).collect();
    let categorias: Vec<&str> = corpus.iter().map(|d| d.categoria.as_str()).collect();
    let anios: Vec<i32> = corpus.iter().map(|d| d.anio).collect();
    let contenidos: Vec<&str> = corpus.iter().map(|d| d.contenido.as_str()).collect();

    df!(
        "id" => ids,
        "titulo" => titulos,
        "categoria" => categorias,
        "anio" => anios,
        "contenido" => contenidos
    )
}

fn generar_query_embedding_por_cluster(cluster_id: usize) -> Vec<f32> {
    let mut vec = vec![0.02f32; DIMENSION];
    let offset = cluster_id * 8;
    for i in 0..8 {
        if offset + i < DIMENSION {
            vec[offset + i] = 0.85f32;
        }
    }
    vec
}

fn generar_embedding_cluster(cluster_id: usize, doc_idx: usize) -> Vec<f32> {
    let mut v = vec![0.03f32; DIMENSION];
    let offset = cluster_id * 8;
    for i in 0..8 {
        if offset + i < DIMENSION {
            // Variación semántica ligera para simular documentos reales del mismo tema
            let factor = 0.70 + ((doc_idx + i) % 5) as f32 * 0.06;
            v[offset + i] = factor;
        }
    }
    normalizar_vector(&mut v);
    v
}

fn construir_corpus_tecnico() -> Vec<Documento> {
    vec![
        // DOMINIO 0: RUST & SISTEMAS (Cluster 0: offsets 0..8)
        Documento {
            id: 0,
            titulo: "Seguridad de Memoria y Ownership en Rust".into(),
            categoria: "rust_sistemas".into(),
            anio: 2024,
            contenido: "El compilador de Rust garantiza ausencia de punteros nulos y data races en tiempo de compilación sin recolector de basura.".into(),
            embedding: generar_embedding_cluster(0, 0),
        },
        Documento {
            id: 1,
            titulo: "Tokio: Runtime Asíncrono de Ultra-Alto Rendimiento".into(),
            categoria: "rust_sistemas".into(),
            anio: 2024,
            contenido: "Tokio permite microservicios de inferencia con millones de conexiones concurrentes usando epoll y work-stealing thread pools.".into(),
            embedding: generar_embedding_cluster(0, 1),
        },
        Documento {
            id: 2,
            titulo: "Cero-Copia y Memoria Mapeada con memmap2".into(),
            categoria: "rust_sistemas".into(),
            anio: 2023,
            contenido: "Mapeo de archivos masivos a la memoria virtual del kernel de Linux, permitiendo leer tensores de 40 GB sin saturar la RAM.".into(),
            embedding: generar_embedding_cluster(0, 2),
        },
        Documento {
            id: 3,
            titulo: "Aceleración Vectorial con Intrínsecos SIMD AVX2".into(),
            categoria: "rust_sistemas".into(),
            anio: 2024,
            contenido: "Uso de registros YMM de 256 bits y FMA para multiplicar y sumar 8 números floats en un único ciclo de reloj de la CPU.".into(),
            embedding: generar_embedding_cluster(0, 3),
        },
        Documento {
            id: 4,
            titulo: "Gestión de Dependencias y Compilación Nativa con Cargo".into(),
            categoria: "rust_sistemas".into(),
            anio: 2022,
            contenido: "Configuración de RUSTFLAGS='-C target-cpu=native' para desbloquear todas las instrucciones de hardware del silicio local.".into(),
            embedding: generar_embedding_cluster(0, 4),
        },
        Documento {
            id: 5,
            titulo: "Unsafe Rust: Cuándo cruzar la frontera del compilador".into(),
            categoria: "rust_sistemas".into(),
            anio: 2023,
            contenido: "Desreferenciación de punteros crudos para comunicarse directamente con controladores de hardware y buffers de la GPU.".into(),
            embedding: generar_embedding_cluster(0, 5),
        },

        // DOMINIO 1: LLMS & ARQUITECTURA TRANSFORMER (Cluster 1: offsets 8..16)
        Documento {
            id: 6,
            titulo: "Mecanismo de Auto-Atención Escalar (Scaled Dot-Product Attention)".into(),
            categoria: "llm_ia".into(),
            anio: 2024,
            contenido: "Cálculo de softmax((Q * K^T) / sqrt(d_k)) * V para modelar dependencias contextuales de largo alcance en el lenguaje.".into(),
            embedding: generar_embedding_cluster(1, 0),
        },
        Documento {
            id: 7,
            titulo: "Modelos de Lenguaje de Parámetros Mixtos (MoE: Mixture of Experts)".into(),
            categoria: "llm_ia".into(),
            anio: 2024,
            contenido: "Enrutamiento dinámico de tokens a diferentes redes neuronales expertas, activando solo una fracción de parámetros por inferencia.".into(),
            embedding: generar_embedding_cluster(1, 1),
        },
        Documento {
            id: 8,
            titulo: "Cuantización Extrema de Pesos: GGUF, AWQ y GPTQ a 4 bits".into(),
            categoria: "llm_ia".into(),
            anio: 2023,
            contenido: "Reducción de pesos flotantes FP16 a enteros INT4 permitiendo ejecutar modelos de 70B en hardware comercial con pérdida mínima de perplexity.".into(),
            embedding: generar_embedding_cluster(1, 2),
        },
        Documento {
            id: 9,
            titulo: "Alineación y Aprendizaje por Refuerzo con Feedback Humano (RLHF)".into(),
            categoria: "llm_ia".into(),
            anio: 2023,
            contenido: "Optimización por DPO y PPO para garantizar respuestas útiles, honestas e inocuas en modelos conversacionales.".into(),
            embedding: generar_embedding_cluster(1, 3),
        },
        Documento {
            id: 10,
            titulo: "LoRA y QLoRA: Fine-Tuning Eficiente con Matrices de Bajo Rango".into(),
            categoria: "llm_ia".into(),
            anio: 2024,
            contenido: "Congelación del modelo base y entrenamiento exclusivo de adaptadores A y B (W = W0 + B*A) reduciendo los requisitos de VRAM en un 80%.".into(),
            embedding: generar_embedding_cluster(1, 4),
        },
        Documento {
            id: 11,
            titulo: "Rotary Positional Embeddings (RoPE) y Ventanas de Contexto Infinitas".into(),
            categoria: "llm_ia".into(),
            anio: 2024,
            contenido: "Incrustación de información posicional mediante matrices de rotación 2D en los espacios de Query y Key, permitiendo escalar a 128k tokens.".into(),
            embedding: generar_embedding_cluster(1, 5),
        },

        // DOMINIO 2: VECTOR DATABASES & RAG (Cluster 2: offsets 16..24)
        Documento {
            id: 12,
            titulo: "Índices HNSW: Grafos del Mundo Pequeño Navegables Jerárquicos".into(),
            categoria: "vector_search".into(),
            anio: 2024,
            contenido: "Búsqueda aproximada de vecinos más cercanos (ANN) con complejidad logarítmica O(log N) saltando entre capas de diferente densidad.".into(),
            embedding: generar_embedding_cluster(2, 0),
        },
        Documento {
            id: 13,
            titulo: "Búsqueda Híbrida: Fusión Recíproca de Rangos (RRF) entre BM25 y Vectores".into(),
            categoria: "vector_search".into(),
            anio: 2024,
            contenido: "Combinación de búsqueda léxica de palabras clave exactas con similitud semántica densa para máxima precisión en RAG.".into(),
            embedding: generar_embedding_cluster(2, 1),
        },
        Documento {
            id: 14,
            titulo: "Product Quantization (PQ) para Compresión de Índices Vectoriales".into(),
            categoria: "vector_search".into(),
            anio: 2023,
            contenido: "División de vectores de alta dimensión en sub-vectores cuantizados con k-means, logrando indexar 100M de vectores en 16 GB de RAM.".into(),
            embedding: generar_embedding_cluster(2, 2),
        },
        Documento {
            id: 15,
            titulo: "Similitud Coseno vs Distancia Euclidiana L2 en Espacios de Embeddings".into(),
            categoria: "vector_search".into(),
            anio: 2023,
            contenido: "Por qué normalizar vectores a norma 1.0 transforma la similitud coseno en un simple producto punto ultra-rápido en CPU y GPU.".into(),
            embedding: generar_embedding_cluster(2, 3),
        },
        Documento {
            id: 16,
            titulo: "Estrategias de Chunking Semántico y Solapamiento en Pipelines RAG".into(),
            categoria: "vector_search".into(),
            anio: 2024,
            contenido: "Fragmentación de documentos basada en párrafos, títulos y densidad de información para evitar pérdida de contexto en la recuperación.".into(),
            embedding: generar_embedding_cluster(2, 4),
        },
        Documento {
            id: 17,
            titulo: "Reranking Neuronal con Modelos Cross-Encoder".into(),
            categoria: "vector_search".into(),
            anio: 2024,
            contenido: "Re-ordenamiento de los 50 documentos candidatos devueltos por el índice vectorial usando un modelo de atención bidireccional densa.".into(),
            embedding: generar_embedding_cluster(2, 5),
        },

        // DOMINIO 3: COMPUTACIÓN EN GPU Y ACELERACIÓN DE HARDWARE (Cluster 3: offsets 24..32)
        Documento {
            id: 18,
            titulo: "FlashAttention: Atención Exacta Rápida y Consciente de la Caché SRAM".into(),
            categoria: "gpu_hpc".into(),
            anio: 2024,
            contenido: "Reordenamiento de bloques de computación para minimizar lecturas y escrituras hacia la memoria HBM de la GPU mediante tiling en SRAM.".into(),
            embedding: generar_embedding_cluster(3, 0),
        },
        Documento {
            id: 19,
            titulo: "Programación de Kernels CUDA y Jerarquía de Memoria de GPU".into(),
            categoria: "gpu_hpc".into(),
            anio: 2023,
            contenido: "Organización de hilos en bloques y warps, y utilización de Shared Memory para evitar colisiones de bancos de memoria.".into(),
            embedding: generar_embedding_cluster(3, 1),
        },
        Documento {
            id: 20,
            titulo: "Triton: Lenguaje y Compilador para Kernels de Deep Learning de Alto Rendimiento".into(),
            categoria: "gpu_hpc".into(),
            anio: 2024,
            contenido: "Generación de código GPU ultra-optimizado sin la complejidad del ensamblador PTX o C++ CUDA de bajo nivel.".into(),
            embedding: generar_embedding_cluster(3, 2),
        },
        Documento {
            id: 21,
            titulo: "Tensor Cores y Operaciones Matriciales Asíncronas en Arquitecturas Hopper y Blackwell".into(),
            categoria: "gpu_hpc".into(),
            anio: 2024,
            contenido: "Multiplicación de matrices en precisión reducida FP8 y FP4 con instrucciones hardware TMA (Tensor Memory Accelerator).".into(),
            embedding: generar_embedding_cluster(3, 3),
        },
        Documento {
            id: 22,
            titulo: "Ancho de Banda de Memoria HBM3e vs Capacidad de Cómputo (Roofline Model)".into(),
            categoria: "gpu_hpc".into(),
            anio: 2024,
            contenido: "Identificación de fases memory-bound vs compute-bound durante la generación de tokens autorregresivos en modelos masivos.".into(),
            embedding: generar_embedding_cluster(3, 4),
        },
        Documento {
            id: 23,
            titulo: "WGPU y Shaders de Cómputo WGSL: El Futuro de la IA Nativa en Rust".into(),
            categoria: "gpu_hpc".into(),
            anio: 2024,
            contenido: "Estándar multiplataforma WebGPU para ejecutar inferencia de redes neuronales tanto en navegadores como en servidores Linux con Vulkan y Metal.".into(),
            embedding: generar_embedding_cluster(3, 5),
        },
    ]
}
