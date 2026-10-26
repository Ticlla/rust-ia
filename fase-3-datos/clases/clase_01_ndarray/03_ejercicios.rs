//! ==============================================================================
//! CLASE 01: Ejercicios Prácticos con `ndarray`
//! ==============================================================================
//!
//! En este laboratorio pondrás en práctica los tres pilares del álgebra en IA:
//! 1. Normalización de Embeddings (Norma L2 y Similitud Coseno).
//! 2. Extracción de Mini-Batches mediante Vistas Cero-Copia (`ArrayView2`).
//! 3. Forward Pass de una Capa Densa (Linear + Bias + ReLU).

use ndarray::{array, s, Array1, Array2, ArrayView2};

fn main() {
    println!("============================================================");
    println!("🧪 EJERCICIOS CLASE 01: Álgebra Lineal en IA con `ndarray`");
    println!("============================================================\n");
    println!("Ejecuta 'cargo test --bin c01_ejercicios' para evaluar tus soluciones.");
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 1: Normalización L2 de Embeddings y Similitud Coseno
/// ----------------------------------------------------------------------------
/// En motores de búsqueda semántica (como vector databases), los embeddings
/// deben normalizarse dividiendo cada vector por su norma L2:
///   norma_l2 = sqrt(sum(x_i^2))
///   x_normalizado = x / norma_l2
///
/// Si la norma es 0.0 (o menor a 1e-12), se debe evitar división por cero.
pub fn normalizar_l2(matriz: &Array2<f32>) -> Array2<f32> {
    let mut resultado = matriz.clone();
    for mut fila in resultado.rows_mut() {
        let suma_cuadrados: f32 = fila.iter().map(|&x| x * x).sum();
        let norma = suma_cuadrados.sqrt();
        if norma > 1e-12 {
            fila.mapv_inplace(|x| x / norma);
        }
    }
    resultado
}

/// Similitud coseno entre todas las filas: S = X_norm · (X_norm)^T
pub fn matriz_similitud_coseno(matriz: &Array2<f32>) -> Array2<f32> {
    let normalizada = normalizar_l2(matriz);
    normalizada.dot(&normalizada.t())
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 2: Particionador de Mini-Batches Cero-Copia
/// ----------------------------------------------------------------------------
/// Dado un dataset de tamaño [N, D] y un `batch_size`, retorna un vector de
/// vistas (`ArrayView2<'a, f32>`) de cada lote sin copiar datos en RAM.
pub fn extraer_mini_batches<'a>(
    dataset: &'a Array2<f32>,
    batch_size: usize,
) -> Vec<ArrayView2<'a, f32>> {
    let mut batches = Vec::new();
    let total_filas = dataset.nrows();
    let mut inicio = 0;

    while inicio < total_filas {
        let fin = (inicio + batch_size).min(total_filas);
        let vista = dataset.slice(s![inicio..fin, ..]);
        batches.push(vista);
        inicio = fin;
    }

    batches
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 3: Capa Densa (Forward Pass: Y = ReLU(X · W + b))
/// ----------------------------------------------------------------------------
/// Implementa una pasada hacia adelante (forward pass) completa:
/// 1. Multiplica X [Batch, In_dim] por W [In_dim, Out_dim].
/// 2. Suma el vector de sesgo 'b' [Out_dim] mediante broadcasting.
/// 3. Aplica la función de activación ReLU: f(x) = max(0, x).
pub fn forward_capa_lineal_relu(
    x: &Array2<f32>,
    w: &Array2<f32>,
    bias: &Array1<f32>,
) -> Array2<f32> {
    // 1. Matmul
    let mut z = x.dot(w);

    // 2. Broadcasting del sesgo a cada fila
    z = z + bias;

    // 3. Activación ReLU in-place
    z.mapv_into(|val| val.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalizar_l2_y_similitud_coseno() {
        let embeddings: Array2<f32> = array![
            [3.0, 4.0], // Norma = 5.0 -> [0.6, 0.8]
            [1.0, 0.0], // Norma = 1.0 -> [1.0, 0.0]
            [0.0, 2.0]  // Norma = 2.0 -> [0.0, 1.0]
        ];

        let normalizados = normalizar_l2(&embeddings);

        // Comprobar que la norma de cada fila es 1.0
        for fila in normalizados.rows() {
            let norma: f32 = fila.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!((norma - 1.0).abs() < 1e-5);
        }

        // Matriz de similitud coseno
        let similitudes = matriz_similitud_coseno(&embeddings);

        // La diagonal siempre debe ser 1.0 (un vector consigo mismo)
        assert!((similitudes[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((similitudes[[1, 1]] - 1.0).abs() < 1e-5);
        assert!((similitudes[[2, 2]] - 1.0).abs() < 1e-5);

        // Similitud entre [1.0, 0.0] y [0.0, 1.0] es 0.0 (ortogonales)
        assert!((similitudes[[1, 2]] - 0.0).abs() < 1e-5);

        // Similitud entre [0.6, 0.8] y [1.0, 0.0] es 0.6
        assert!((similitudes[[0, 1]] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_extraer_mini_batches_cero_copia() {
        let dataset: Array2<f32> = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0]
        ];

        let batches = extraer_mini_batches(&dataset, 2);
        assert_eq!(batches.len(), 3); // Lotes de [2, 2, 1]

        assert_eq!(batches[0].shape(), &[2, 2]);
        assert_eq!(batches[1].shape(), &[2, 2]);
        assert_eq!(batches[2].shape(), &[1, 2]);

        // Verificar contenido del último lote
        assert_eq!(batches[2][[0, 0]], 9.0);
        assert_eq!(batches[2][[0, 1]], 10.0);
    }

    #[test]
    fn test_forward_capa_lineal_relu() {
        // Batch de 2 muestras con dimensión 2
        let x: Array2<f32> = array![
            [1.0, -1.0],
            [2.0, 3.0]
        ];

        // Pesos [2, 2]
        let w: Array2<f32> = array![
            [2.0, -1.0],
            [1.0, 2.0]
        ];

        // Sesgo [2]
        let bias: Array1<f32> = array![0.5, -2.0];

        // x · w:
        // Fila 0: [1*2 + -1*1, 1*-1 + -1*2] = [2 - 1, -1 - 2] = [1.0, -3.0]
        // Fila 1: [2*2 + 3*1, 2*-1 + 3*2]   = [4 + 3, -2 + 6]  = [7.0, 4.0]
        //
        // + bias ([0.5, -2.0]):
        // Fila 0: [1.0 + 0.5, -3.0 - 2.0] = [1.5, -5.0]
        // Fila 1: [7.0 + 0.5, 4.0 - 2.0]  = [7.5, 2.0]
        //
        // ReLU (max(0, x)):
        // Fila 0: [1.5, 0.0]
        // Fila 1: [7.5, 2.0]

        let salida = forward_capa_lineal_relu(&x, &w, &bias);

        assert!((salida[[0, 0]] - 1.5).abs() < 1e-5);
        assert!((salida[[0, 1]] - 0.0).abs() < 1e-5);
        assert!((salida[[1, 0]] - 7.5).abs() < 1e-5);
        assert!((salida[[1, 1]] - 2.0).abs() < 1e-5);
    }
}
