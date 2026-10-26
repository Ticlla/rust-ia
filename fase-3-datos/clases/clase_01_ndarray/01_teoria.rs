//! ==============================================================================
//! CLASE 01: Álgebra Lineal y Tensores con `ndarray`
//! ==============================================================================
//!
//! En Inteligencia Artificial, las redes neuronales y los transformers son
//! fundamentalmente operaciones de álgebra lineal sobre tensores multidimensionales.
//!
//! En Python utilizamos `NumPy` (`np.ndarray`) o `PyTorch` (`torch.Tensor`).
//! En Rust, el estándar industrial para cálculo N-dimensional en CPU es `ndarray`.
//!
//! OBJETIVOS PEDAGÓGICOS:
//! 1. Entender la anatomía de un tensor en memoria: Shape, Strides y C-Contiguity.
//! 2. Dominar las Vistas Cero-Copia (`ArrayView`) y Slicing multidimensional (`s![]`).
//! 3. Realizar operaciones matriciales fundamentales en IA: Matmul (`dot`), Dot Product.
//! 4. Comprender el Broadcasting para sumar sesgos (bias) a lotes de predicciones.

use ndarray::{array, s, Array1, Array2, ArrayView2};

fn main() {
    println!("============================================================");
    println!("🧠 CLASE 01: Tensores y Álgebra Lineal con `ndarray`");
    println!("============================================================\n");

    demostrar_anatomia_memoria();
    demostrar_vistas_cero_copia();
    demostrar_multiplicacion_matrices();
    demostrar_broadcasting_en_redes_neuronales();

    println!("\n✅ Demostración teórica completada con éxito.");
}

/// ----------------------------------------------------------------------------
/// 1. ANATOMÍA DE UN TENSOR EN MEMORIA (Shape, Strides y Caché L1/L2)
/// ----------------------------------------------------------------------------
/// Un tensor 2D de [M, N] NO es un vector de vectores (`Vec<Vec<f32>>`).
/// Un `Vec<Vec<f32>>` dispersa los punteros por todo el Heap, provocando
/// incontables "Cache Misses" que destruyen el rendimiento de la CPU.
///
/// Un `Array2<f32>` almacena todos los elementos en un ÚNICO bloque contiguo plano.
/// ¿Cómo sabe dónde está el elemento (i, j)?
/// Índice en memoria = i * stride_filas + j * stride_columnas
fn demostrar_anatomia_memoria() {
    println!("--- 1. Anatomía en Memoria: Shape, Strides y Contigüidad ---");

    // Matriz de 2 filas x 3 columnas (ej: 2 vectores de embeddings de dimensión 3)
    let matriz: Array2<f32> = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ];

    println!("Matriz original:\n{:?}", matriz);
    println!("Forma (Shape): {:?}", matriz.shape()); // [2, 3]
    println!("Saltos (Strides): {:?}", matriz.strides()); // [3, 1]
    println!("¿Es contigua en formato C (Row-Major)?: {}", matriz.is_standard_layout());

    // Explicación de los Strides:
    // Stride [3, 1] significa:
    // - Para avanzar a la siguiente fila (+1 en i): saltar 3 posiciones en memoria plana.
    // - Para avanzar a la siguiente columna (+1 en j): saltar 1 posición en memoria plana.
    // Memoria física real en RAM: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] (100% contigua)

    // Transpuesta (Cero Copias)
    let transpuesta = matriz.t();
    println!("\nMatriz Transpuesta (sin copiar datos):\n{:?}", transpuesta);
    println!("Forma de la transpuesta: {:?}", transpuesta.shape()); // [3, 2]
    println!("Saltos de la transpuesta: {:?}", transpuesta.strides()); // [1, 3]
    println!("¿Es contigua en C?: {}", transpuesta.is_standard_layout());
    // 💡 La transpuesta no duplicó ni 1 solo byte en RAM; solo invirtió los strides!
    println!();
}

/// ----------------------------------------------------------------------------
/// 2. VISTAS CERO-COPIA (ArrayView) Y SLICING MULTIDIMENSIONAL
/// ----------------------------------------------------------------------------
/// En Python: `lote = dataset[0:32, :]`
/// En Rust: Usamos la macro `s![filas, columnas]` para obtener un `ArrayView`.
/// Un `ArrayView` es un puntero a los datos originales con un ciclo de vida `'a`.
/// Cero asignaciones en el Heap. Si el dataset pesa 10GB, el slice cuesta 0 bytes extra.
fn demostrar_vistas_cero_copia() {
    println!("--- 2. Vistas Cero-Copia (Slicing con s![...]) ---");

    // Supongamos un lote (batch) de 4 muestras con dimensión 4
    let mut dataset: Array2<f32> = array![
        [10.0, 11.0, 12.0, 13.0], // Muestra 0
        [20.0, 21.0, 22.0, 23.0], // Muestra 1
        [30.0, 31.0, 32.0, 33.0], // Muestra 2
        [40.0, 41.0, 42.0, 43.0]  // Muestra 3
    ];

    // Extraer un mini-batch con las primeras 2 muestras (filas 0..2) y todas las columnas (..)
    let mini_batch: ArrayView2<f32> = dataset.slice(s![0..2, ..]);
    println!("Mini-batch (filas 0..2):\n{:?}", mini_batch);

    // Extraer solo una columna (por ejemplo, la primera característica de cada muestra)
    let feature_0 = dataset.slice(s![.., 0]);
    println!("Primera característica de todas las muestras: {:?}", feature_0);

    // Modificación mutable in-place a través de una vista mutable (ArrayViewMut)
    // Multiplicamos por 2 la última fila en el dataset original
    {
        let mut ultima_fila = dataset.slice_mut(s![3, ..]);
        ultima_fila.mapv_inplace(|x| x * 2.0);
    }
    println!("Dataset tras mutar la última fila in-place:\n{:?}", dataset);
    println!();
}

/// ----------------------------------------------------------------------------
/// 3. ÁLGEBRA LINEAL: MULTIPLICACIÓN MATRICIAL (Y = X · W)
/// ----------------------------------------------------------------------------
/// Capa densa (Linear Layer) en Deep Learning:
/// Entradas X: [Batch_size, Input_dim]
/// Pesos W:   [Input_dim, Output_dim]
/// Salida Y:  [Batch_size, Output_dim]
fn demostrar_multiplicacion_matrices() {
    println!("--- 3. Multiplicación Matricial (Matmul / GEMM) ---");

    // Batch de 2 tokens, cada uno representado por un embedding de dimensión 3
    // Shape: [2, 3]
    let x: Array2<f32> = array![
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5]
    ];

    // Matriz de pesos de la capa: proyecta de dimensión 3 a dimensión 2
    // Shape: [3, 2]
    let w: Array2<f32> = array![
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]
    ];

    // Matmul: X · W mediante el método .dot()
    // Shape resultante: [2, 2]
    let y: Array2<f32> = x.dot(&w);

    println!("X (Entrada [2, 3]):\n{:?}", x);
    println!("W (Pesos [3, 2]):\n{:?}", w);
    println!("Y = X · W (Salida [2, 2]):\n{:?}", y);

    // Verificación manual:
    // Fila 0: [1.0*0.1 + 2.0*0.3 + 3.0*0.5, 1.0*0.2 + 2.0*0.4 + 3.0*0.6]
    //       = [0.1 + 0.6 + 1.5, 0.2 + 0.8 + 1.8] = [2.2, 2.8]
    assert!((y[[0, 0]] - 2.2).abs() < 1e-5);
    assert!((y[[0, 1]] - 2.8).abs() < 1e-5);
    println!("✓ Verificación matemática exacta confirmada.");
    println!();
}

/// ----------------------------------------------------------------------------
/// 4. BROADCASTING: SUMA DE SESGOS (BIAS) EN REDES NEURONALES
/// ----------------------------------------------------------------------------
/// En IA, casi siempre calculamos: Y = X · W + b
/// Donde Y es [Batch_size, Dim] y el sesgo 'b' es un vector 1D de tamaño [Dim].
/// El "Broadcasting" expande virtualmente el sesgo a lo largo del eje del Batch
/// sin duplicar memoria.
fn demostrar_broadcasting_en_redes_neuronales() {
    println!("--- 4. Broadcasting: Suma de Sesgos (Bias) ---");

    // Salida lineal previa: 2 muestras x 2 neuronas de salida
    let activaciones: Array2<f32> = array![
        [2.2, 2.8],
        [1.7, 2.2]
    ];

    // Vector de sesgo para las 2 neuronas
    let bias: Array1<f32> = array![10.0, -5.0];

    // En ndarray, sumamos el sesgo a lo largo de un eje específico (Axis(0) o Axis(1))
    // mediante sumas o broadcasting:
    // Para sumar un vector 1D a cada fila de una matriz 2D:
    let mut salida = activaciones.clone();
    salida = salida + &bias; // ndarray aplica broadcasting automático en la última dimensión

    println!("Activaciones antes del sesgo:\n{:?}", activaciones);
    println!("Sesgo a sumar:\n{:?}", bias);
    println!("Salida final (Activaciones + Sesgo):\n{:?}", salida);

    // Muestra 0: [2.2 + 10.0, 2.8 - 5.0] = [12.2, -2.2]
    assert!((salida[[0, 0]] - 12.2).abs() < 1e-5);
    assert!((salida[[0, 1]] - (-2.2)).abs() < 1e-5);
    println!("✓ Broadcasting completado sin asignar copias intermedias.");
}
