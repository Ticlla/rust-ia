//! ============================================================================
//! 🎯 FASE 2 - CLASE 05: Retos Prácticos con Rayon en Computación para IA
//! ============================================================================
//!
//! Ejecutar el reto:
//!     cargo run --bin c05_ejercicios
//!
//! Ejecutar las pruebas unitarias:
//!     cargo test --bin c05_ejercicios

use rayon::prelude::*;

/// Normaliza un lote (batch) de vectores de características en paralelo (LayerNorm).
/// Cada vector se transforma a: x_norm = (x - media) / sqrt(varianza + eps)
pub fn layer_norm_batch_paralelo(lote: &mut [Vec<f32>], eps: f32) {
    // 💡 Procesamos cada fila/muestra del batch en un hilo diferente de forma paralela:
    lote.par_iter_mut().for_each(|muestra| {
        if muestra.is_empty() {
            return;
        }

        let n = muestra.len() as f32;
        let media: f32 = muestra.iter().sum::<f32>() / n;
        let varianza: f32 = muestra.iter().map(|&x| (x - media).powi(2)).sum::<f32>() / n;
        let desviacion_std = (varianza + eps).sqrt();

        for x in muestra.iter_mut() {
            *x = (*x - media) / desviacion_std;
        }
    });
}

/// Multiplicación Paralela de Matriz por Vector (Inferencia de Capa Lineal): y = W * x
/// Cada fila de la matriz W se proyecta contra el vector x en paralelo.
pub fn matriz_vector_mul_paralela(matriz: &[Vec<f32>], vector: &[f32]) -> Vec<f32> {
    if matriz.is_empty() {
        return vec![];
    }

    let dim = vector.len();

    // 💡 Calculamos cada elemento del vector de salida en paralelo:
    matriz
        .par_iter()
        .map(|fila| {
            assert_eq!(fila.len(), dim, "Incompatibilidad de dimensiones en la multiplicación");
            fila.iter().zip(vector.iter()).map(|(&w, &x)| w * x).sum()
        })
        .collect()
}

fn main() {
    println!("============================================================");
    println!("🎯 RETO PRÁCTICO: Operaciones Tensoriales Paralelas con Rayon");
    println!("============================================================\n");

    // 1. Simulación de Multiplicación Matriz x Vector:
    println!("1. Multiplicación Matriz (3x3) x Vector (3) en Paralelo:");
    let pesos = vec![
        vec![1.0, 2.0, 3.0],
        vec![0.0, 1.0, 4.0],
        vec![5.0, 6.0, 0.0],
    ];
    let entrada = vec![0.5, 1.0, 2.0];

    let salida = matriz_vector_mul_paralela(&pesos, &entrada);
    println!("   Pesos W : {:?}", pesos);
    println!("   Vector x: {:?}", entrada);
    println!("   Salida y: {:?}", salida);
    println!("   Esperado: [1*0.5 + 2*1 + 3*2 = 8.5, 0*0.5 + 1*1 + 4*2 = 9.0, 5*0.5 + 6*1 = 8.5]\n");

    // 2. Simulación de LayerNorm sobre un lote:
    println!("2. LayerNorm Paralelo sobre un Batch de 2 vectores:");
    let mut batch = vec![
        vec![10.0, 20.0, 30.0],
        vec![100.0, 200.0, 300.0],
    ];
    println!("   Antes de normalizar: {:?}", batch);

    layer_norm_batch_paralelo(&mut batch, 1e-5);
    println!("   Normalizado (Media ≈ 0, Varianza ≈ 1):");
    for (i, v) in batch.iter().enumerate() {
        println!("     Muestra #{i}: {:?}", v);
    }

    println!("\n✅ Reto ejecutado exitosamente.");
}

// =============================================================================
// PRUEBAS UNITARIAS AUTOMÁTICAS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matriz_vector_mul_correcta() {
        let matriz = vec![
            vec![2.0, 0.0],
            vec![1.0, 3.0],
        ];
        let vector = vec![4.0, 5.0];

        let res = matriz_vector_mul_paralela(&matriz, &vector);
        // fila 0: 2*4 + 0*5 = 8.0
        // fila 1: 1*4 + 3*5 = 19.0
        assert_eq!(res, vec![8.0, 19.0]);
    }

    #[test]
    fn test_layer_norm_media_cero_varianza_uno() {
        let mut batch = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
        ];

        layer_norm_batch_paralelo(&mut batch, 1e-5);

        for muestra in batch {
            let n = muestra.len() as f32;
            let media: f32 = muestra.iter().sum::<f32>() / n;
            let varianza: f32 = muestra.iter().map(|&x| (x - media).powi(2)).sum::<f32>() / n;

            // Media debe ser ~0.0 y varianza debe ser ~1.0:
            assert!(media.abs() < 1e-5, "Media esperada ~0, obtenida: {}", media);
            assert!((varianza - 1.0).abs() < 1e-3, "Varianza esperada ~1, obtenida: {}", varianza);
        }
    }

    #[test]
    fn test_batch_vacio() {
        let mut vacio: Vec<Vec<f32>> = vec![];
        layer_norm_batch_paralelo(&mut vacio, 1e-5);
        assert!(vacio.is_empty());

        let res_mul = matriz_vector_mul_paralela(&[], &[1.0, 2.0]);
        assert!(res_mul.is_empty());
    }
}
