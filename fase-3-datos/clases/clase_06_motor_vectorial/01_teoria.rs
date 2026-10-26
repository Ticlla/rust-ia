//! ==============================================================================
//! CLASE 06: Checkpoint Final de Fase 3 - Motor de Búsqueda Vectorial (RAG Index)
//! ==============================================================================
//!
//! Este proyecto integra las 5 herramientas fundamentales dominadas en la Fase 3:
//! 1. `ndarray`: Manejo matricial y manipulación de vectores.
//! 2. `polars`: Filtrado declarativo rápido de metadatos (pre-filtrado de documentos).
//! 3. `safetensors`: Almacenamiento seguro, compacto y de alta velocidad de embeddings.
//! 4. `memmap2`: Mapeo a memoria virtual desde el disco para acceso Cero-Copia.
//! 5. `SIMD (AVX2)`: Aceleración por hardware para el cálculo de similitud coseno.

use memmap2::Mmap;
use polars::prelude::*;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("🚀 CLASE 06: MOTOR DE BÚSQUEDA VECTORIAL DE ALTO RENDIMIENTO (RAG)");
    println!("============================================================\n");

    let temp_dir = std::env::temp_dir();
    let safetensors_path = temp_dir.join("embeddings_rag_index.safetensors");

    println!("--- 1. Construcción del Corpus y Embeddings Normalizados ---");
    // Metadatos de documentos para el sistema RAG:
    let metadatos = df!(
        "id" => [0i64, 1, 2, 3, 4],
        "titulo" => [
            "Introducción a Rust para IA",
            "Tensores y Álgebra con ndarray",
            "Optimización de Consultas en Polars",
            "Carga Cero-Copia con Memmap2",
            "Receta de Pizza Margarita Tradicional"
        ],
        "categoria" => [
            "rust_ia",
            "rust_ia",
            "big_data",
            "sistemas",
            "cocina"
        ],
        "contenido" => [
            "Rust ofrece seguridad de memoria y rendimiento nativo para LLMs.",
            "ndarray permite operaciones tensoriales eficientes sin copias.",
            "Polars aprovecha Apache Arrow para procesamiento masivo en memoria.",
            "memmap2 mapea archivos gigantes del SSD a la memoria virtual.",
            "Masa fermentada 24 horas con salsa de tomate y albahaca fresca."
        ]
    )?;

    println!("✓ Catálogo de metadatos creado con Polars (5 documentos):");
    println!("{:?}\n", metadatos);

    // Dimensión de los embeddings (D = 8 para simplificar la visualización)
    let dimension = 8;
    // Creamos 5 vectores pre-normalizados (norma L2 = 1.0)
    let mut embeddings_raw: Vec<f32> = vec![
        // Doc 0: rust_ia general
        0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0,
        // Doc 1: tensores y ndarray
        0.7, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
        // Doc 2: polars big data
        0.0, 0.2, 0.8, 0.5, 0.2, 0.0, 0.0, 0.0,
        // Doc 3: sistemas memmap
        0.4, 0.1, 0.2, 0.8, 0.3, 0.0, 0.0, 0.0,
        // Doc 4: cocina (completamente ortogonal a la IA)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.7,
    ];

    // Normalizamos cada fila a norma L2 = 1.0 para que dot_product == cosine_similarity:
    normalizar_matriz_en_sitio(&mut embeddings_raw, 5, dimension);

    println!("--- 2. Persistencia en Formato Safetensors (Clase 03) ---");
    guardar_indice_safetensors(&safetensors_path, &embeddings_raw, 5, dimension)?;
    println!("✓ Embeddings guardados en disco: {:?}\n", safetensors_path);

    println!("--- 3. Apertura Cero-Copia con Memmap2 (Clase 04) ---");
    let file = File::open(&safetensors_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;
    let tensor_embeddings = tensors.tensor("document_embeddings")?;
    println!("✓ Tensor mapeado desde SSD sin duplicar memoria RAM:");
    println!("   Forma: {:?}, Tipo de dato: {:?}", tensor_embeddings.shape(), tensor_embeddings.dtype());

    println!("\n--- 4. Consulta de Búsqueda Vectorial (Query) ---");
    // Consulta del usuario: "Quiero saber sobre sistemas y optimización de memoria en Rust"
    let mut query_embedding = vec![0.4f32, 0.2, 0.3, 0.8, 0.2, 0.0, 0.0, 0.0];
    normalizar_vector(&mut query_embedding);
    println!("✓ Vector de consulta normalizado (dimensión 8)");

    println!("\n--- 5. Pre-filtrado Híbrido con Polars (Clase 02) ---");
    // Filtramos para buscar únicamente documentos de tecnología/sistemas (excluimos 'cocina'):
    let candidatos_df = metadatos
        .clone()
        .lazy()
        .filter(col("categoria").eq(lit("rust_ia")).or(col("categoria").eq(lit("sistemas"))))
        .select([col("id"), col("titulo"), col("contenido")])
        .collect()?;

    println!("✓ Candidatos tras filtro de metadatos (pre-filtrado):");
    println!("{:?}\n", candidatos_df);

    println!("--- 6. Scoring SIMD AVX2 + Top-K Ranking (Clase 01 y 05) ---");
    let ids_ca = candidatos_df.column("id")?.i64()?;
    let titulos_ca = candidatos_df.column("titulo")?.str()?;
    let contenidos_ca = candidatos_df.column("contenido")?.str()?;

    // Convertimos los bytes crudos del tensor de safetensors a &[f32] con cero copia:
    let bytes_datos = tensor_embeddings.data();
    let floats_datos: &[f32] = unsafe {
        std::slice::from_raw_parts(
            bytes_datos.as_ptr() as *const f32,
            bytes_datos.len() / std::mem::size_of::<f32>(),
        )
    };

    struct DocumentoScore<'a> {
        id: i64,
        titulo: &'a str,
        contenido: &'a str,
        score: f32,
    }

    let mut resultados = Vec::new();

    for i in 0..candidatos_df.height() {
        if let Some(id) = ids_ca.get(i) {
            let titulo = titulos_ca.get(i).unwrap_or("");
            let contenido = contenidos_ca.get(i).unwrap_or("");

            let inicio = (id as usize) * dimension;
            let fin = inicio + dimension;
            let doc_embedding = &floats_datos[inicio..fin];

            // Cálculo acelerado por hardware con SIMD AVX2:
            let score = producto_punto_simd(&query_embedding, doc_embedding);

            resultados.push(DocumentoScore {
                id,
                titulo,
                contenido,
                score,
            });
        }
    }

    // Ordenamos de mayor a menor similitud (Top-K):
    resultados.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    println!("🏆 RANKING FINAL DE RELEVANCIA (Top Documentos):");
    for (ranking, doc) in resultados.iter().enumerate() {
        println!(
            "   Top {}: [Score: {:.4}] (ID: {}) - \"{}\"",
            ranking + 1,
            doc.score,
            doc.id,
            doc.titulo
        );
        println!("          Snippet: \"{}\"\n", doc.contenido);
    }

    println!("✅ Motor de Búsqueda Vectorial ejecutado con éxito al 100%.");
    Ok(())
}

/// Normaliza un vector en el sitio a norma L2 = 1.0
pub fn normalizar_vector(v: &mut [f32]) {
    let mut sum_sq = 0.0f32;
    for &x in v.iter() {
        sum_sq += x * x;
    }
    let norm = sum_sq.sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Normaliza una matriz plana de embeddings [N, D] fila por fila
pub fn normalizar_matriz_en_sitio(matriz: &mut [f32], filas: usize, cols: usize) {
    for i in 0..filas {
        let inicio = i * cols;
        let fin = inicio + cols;
        normalizar_vector(&mut matriz[inicio..fin]);
    }
}

/// Serializa los embeddings en formato safetensors
pub fn guardar_indice_safetensors(
    path: &Path,
    embeddings: &[f32],
    filas: usize,
    cols: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            embeddings.as_ptr() as *const u8,
            embeddings.len() * std::mem::size_of::<f32>(),
        )
    };

    let shape = vec![filas, cols];
    let view = TensorView::new(Dtype::F32, shape, bytes)?;

    let mut tensors = HashMap::new();
    tensors.insert("document_embeddings".to_string(), view);

    let serializado = safetensors::serialize(&tensors, None)?;
    std::fs::write(path, serializado)?;
    Ok(())
}

/// Producto punto con soporte para SIMD AVX2 + FMA y fallback escalar seguro
pub fn producto_punto_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { producto_punto_avx2(a, b) };
        }
    }

    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
unsafe fn producto_punto_avx2(a: &[f32], b: &[f32]) -> f32 {
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
