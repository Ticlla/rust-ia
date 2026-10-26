//! ==============================================================================
//! CLASE 01: Errores Comunes y Antipatrones con `ndarray`
//! ==============================================================================
//!
//! Al transicionar de NumPy/PyTorch a Rust `ndarray`, los desarrolladores de IA
//! suelen tropezar con diferencias clave de sintaxis, gestión de memoria caché
//! y las garantías del Borrow Checker sobre vistas mutables.

use ndarray::{array, Array2, ArrayView2};

fn main() {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Tensores y Álgebra con `ndarray`");
    println!("============================================================\n");

    error_1_multiplicacion_elementwise_vs_matricial();
    error_2_cache_trashing_recorrer_memoria_al_reves();
    error_3_clonar_vs_prestar_vistas_cero_copia();
    error_4_aliasing_y_vistas_mutables_en_rust();

    println!("\n✅ Todos los antipatrones analizados y resueltos correctamente.");
}

/// ----------------------------------------------------------------------------
/// ERROR 1: Confusión entre `*` (Element-wise) y `.dot()` (Matmul)
/// ----------------------------------------------------------------------------
/// En Python:
///   `A * B` es Hadamard (elemento a elemento).
///   `A @ B` o `np.dot(A, B)` es multiplicación de matrices.
///
/// En Rust `ndarray`:
///   `&A * &B` es Hadamard (las dimensiones deben ser exactamente iguales).
///   `A.dot(&B)` es multiplicación matricial ([M, K] x [K, N] -> [M, N]).
fn error_1_multiplicacion_elementwise_vs_matricial() {
    println!("--- Error 1: Hadamard (*) vs Matmul (.dot()) ---");

    let a: Array2<f32> = array![
        [1.0, 2.0],
        [3.0, 4.0]
    ];
    let b: Array2<f32> = array![
        [5.0, 6.0],
        [7.0, 8.0]
    ];

    // 1. Hadamard (Element-wise)
    let hadamard = &a * &b;
    println!("Hadamard (&a * &b):\n{:?}", hadamard);
    // [1*5, 2*6] = [5, 12]
    // [3*7, 4*8] = [21, 32]
    assert_eq!(hadamard[[0, 0]], 5.0);

    // 2. Matmul (.dot())
    let matmul = a.dot(&b);
    println!("Matmul (a.dot(&b)):\n{:?}", matmul);
    // Fila 0: [1*5 + 2*7, 1*6 + 2*8] = [5+14, 6+16] = [19, 22]
    assert_eq!(matmul[[0, 0]], 19.0);

    println!("👉 Regla: Usa `.dot()` para capas de redes neuronales y `*` para máscaras de atención.");
    println!();
}

/// ----------------------------------------------------------------------------
/// ERROR 2: Cache Trashing (Recorrer memoria a contramano de los Strides)
/// ----------------------------------------------------------------------------
/// En C-Layout (el estándar de ndarray y NumPy), los datos están guardados fila por fila.
///
/// [Fila 0: elem0, elem1, elem2...] [Fila 1: elem0, elem1...]
///
/// Si recorres columna por columna, en cada paso saltas a una dirección distante de RAM.
/// La CPU no puede predecir el acceso (Prefetcher falla) y vacías la memoria caché L1.
fn error_2_cache_trashing_recorrer_memoria_al_reves() {
    println!("--- Error 2: Rendimiento y Caché L1/L2 (Row-Major vs Col-Major) ---");

    let matriz = Array2::<f32>::from_elem((100, 100), 1.0);

    // Forma EFICIENTE (Cache-friendly): Iterar por filas (Row-Major)
    // Lee floats contiguos. 1 línea de caché (64 bytes) carga 16 floats juntos.
    let mut suma_optima = 0.0;
    for fila in matriz.rows() {
        for &val in fila {
            suma_optima += val;
        }
    }

    // Forma INEFICIENTE: Iterar por columnas saltando en memoria
    let mut suma_lenta = 0.0;
    for col in matriz.columns() {
        for &val in col {
            suma_lenta += val;
        }
    }

    assert_eq!(suma_optima, suma_lenta);
    println!("👉 Regla: Siempre itera sobre `rows()` o usa `.iter()` / iteradores vectorizados.");
    println!();
}

/// ----------------------------------------------------------------------------
/// ERROR 3: Clonar matrices gigantes para pasarlas a funciones
/// ----------------------------------------------------------------------------
/// Si un modelo de embeddings tiene una matriz de 100,000 x 768 (~307 MB en RAM),
/// hacer `.clone()` duplica 307 MB y bloquea el Garbage Collector / Asignador.
///
/// En Rust, pásalo como `&Array2<f32>` o como `ArrayView2<f32>`.
fn error_3_clonar_vs_prestar_vistas_cero_copia() {
    println!("--- Error 3: Duplicación innecesaria de tensores (Cero-Copia) ---");

    let embeddings = Array2::<f32>::zeros((500, 128));

    // Antipatrón: `inferir(embeddings.clone())`
    // Patrón Óptimo: Pasar una vista prestada
    fn inferir(vista: ArrayView2<f32>) -> usize {
        vista.nrows() // Procesar sin copiar
    }

    let filas = inferir(embeddings.view());
    println!("Procesadas {} filas con cero asignaciones en RAM.", filas);
    println!();
}

/// ----------------------------------------------------------------------------
/// ERROR 4: Aliasing y Vistas Mutables Solapadas
/// ----------------------------------------------------------------------------
/// En C o Python, dos hilos o dos funciones pueden modificar la misma matriz
/// al mismo tiempo, provocando "Data Races" imposibles de depurar.
///
/// Rust prohíbe tener dos `&mut` sobre la misma matriz o regiones coincidentes.
/// Para dividir una matriz mutablemente de forma segura, se usa `split_at(Axis(0), idx)`.
fn error_4_aliasing_y_vistas_mutables_en_rust() {
    println!("--- Error 4: Partición segura de matrices mutables ---");

    let mut matriz: Array2<f32> = array![
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ];

    // Dividir la matriz a la mitad de forma segura (2 filas arriba, 2 filas abajo)
    // El método `split_at` garantiza al compilador que las dos mitades NO se solapan.
    let (mut parte_superior, mut parte_inferior) = matriz.view_mut().split_at(ndarray::Axis(0), 2);

    parte_superior.mapv_inplace(|x| x * 10.0);
    parte_inferior.mapv_inplace(|x| x + 100.0);

    println!("Matriz modificada concurrentemente de forma segura:\n{:?}", matriz);
    println!("👉 Regla: Usa `.split_at(Axis, idx)` cuando quieras procesar mitades en paralelo.");
}
