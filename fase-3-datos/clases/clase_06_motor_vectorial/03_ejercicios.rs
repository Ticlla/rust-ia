//! ==============================================================================
//! CLASE 06: Ejercicios del Motor de Búsqueda Vectorial (Checkpoint Fase 3)
//! ==============================================================================
//!
//! En este laboratorio implementarás y validarás los componentes de un motor RAG:
//! 1. `guardar_y_cargar_embeddings_mmap`: Guardar en `.safetensors` y mapear con `memmap2`.
//! 2. `filtrar_candidatos_polars`: Pre-filtrado declarativo de metadatos.
//! 3. `buscar_top_k_simd`: Ranking acelerado de candidatos.

use memmap2::Mmap;
use polars::prelude::*;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() {
    println!("============================================================");
    println!("🧪 EJERCICIOS CLASE 06: Checkpoint Final Fase 3");
    println!("============================================================\n");
    println!("Ejecuta 'cargo test --bin c06_ejercicios' para validar los tests.");
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 1: Persistencia y Mapeo Cero-Copia de Embeddings
/// ----------------------------------------------------------------------------
pub fn exportar_a_safetensors(
    path: &Path,
    matriz: &[f32],
    filas: usize,
    cols: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            matriz.as_ptr() as *const u8,
            matriz.len() * std::mem::size_of::<f32>(),
        )
    };

    let shape = vec![filas, cols];
    let view = TensorView::new(Dtype::F32, shape, bytes)?;

    let mut map = HashMap::new();
    map.insert("embeddings".to_string(), view);

    let data = safetensors::serialize(&map, None)?;
    std::fs::write(path, data)?;
    Ok(())
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 2: Pre-filtrado Declarativo con Polars
/// ----------------------------------------------------------------------------
/// Retorna los IDs de documentos cuya categoría coincida con `categoria_filtro`.
pub fn filtrar_ids_por_categoria(
    metadatos: &DataFrame,
    categoria_filtro: &str,
) -> Result<Vec<i64>, PolarsError> {
    let filtrado = metadatos
        .clone()
        .lazy()
        .filter(col("categoria").eq(lit(categoria_filtro)))
        .select([col("id")])
        .collect()?;

    let ids_series = filtrado.column("id")?;
    let ids_ca = ids_series.i64()?;

    let mut lista_ids = Vec::new();
    for i in 0..ids_ca.len() {
        if let Some(val) = ids_ca.get(i) {
            lista_ids.push(val);
        }
    }

    Ok(lista_ids)
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 3: Ranking Top-K con SIMD
/// ----------------------------------------------------------------------------
/// Dado un query vector y una matriz de embeddings en memoria mapeada,
/// evalúa únicamente los `candidatos_ids` y retorna los `k` mejores pares `(id, score)`.
pub fn ranking_top_k(
    query: &[f32],
    embeddings_flat: &[f32],
    dimension: usize,
    candidatos_ids: &[i64],
    k: usize,
) -> Vec<(i64, f32)> {
    let mut scores = Vec::new();

    for &id in candidatos_ids {
        let inicio = (id as usize) * dimension;
        let fin = inicio + dimension;
        let doc_vec = &embeddings_flat[inicio..fin];

        let score = producto_punto_simd(query, doc_vec);
        scores.push((id, score));
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exportar_y_mapear_safetensors_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_roundtrip.safetensors");

        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        exportar_a_safetensors(&path, &original, 1, 8)?;

        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;
        let view = tensors.tensor("embeddings")?;

        let bytes = view.data();
        let recuperado: &[f32] = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                bytes.len() / std::mem::size_of::<f32>(),
            )
        };

        assert_eq!(recuperado, &original[..]);
        Ok(())
    }

    #[test]
    fn test_prefiltrado_hibrido_polars() -> Result<(), Box<dyn std::error::Error>> {
        let df = df!(
            "id" => [0i64, 1, 2, 3],
            "categoria" => ["ia", "finanzas", "ia", "salud"]
        )?;

        let candidatos = filtrar_ids_por_categoria(&df, "ia")?;
        assert_eq!(candidatos, vec![0i64, 2i64]);
        Ok(())
    }

    #[test]
    fn test_ranking_top_k_simd() {
        // Dimensión 4, 3 documentos
        let query = vec![1.0f32, 0.0, 0.0, 0.0];

        // Doc 0: idéntico al query (score = 1.0)
        // Doc 1: ortogonal (score = 0.0)
        // Doc 2: parcialmente similar (score = 0.5)
        let embeddings = vec![
            1.0, 0.0, 0.0, 0.0, // Doc 0
            0.0, 1.0, 0.0, 0.0, // Doc 1
            0.5, 0.5, 0.0, 0.0, // Doc 2
        ];

        let candidatos = vec![0i64, 1, 2];
        let top2 = ranking_top_k(&query, &embeddings, 4, &candidatos, 2);

        assert_eq!(top2.len(), 2);
        // El primero debe ser Doc 0 con score 1.0
        assert_eq!(top2[0].0, 0);
        assert!((top2[0].1 - 1.0).abs() < 1e-5);
        // El segundo debe ser Doc 2 con score 0.5
        assert_eq!(top2[1].0, 2);
        assert!((top2[1].1 - 0.5).abs() < 1e-5);
    }
}
