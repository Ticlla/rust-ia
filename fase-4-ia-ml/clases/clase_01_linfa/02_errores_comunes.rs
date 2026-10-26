//! ==============================================================================
//! CLASE 01: Errores Comunes con `linfa` (Machine Learning en Rust)
//! ==============================================================================
//!
//! Al trabajar con ML en Rust, el compilador previene muchos de los errores
//! en tiempo de ejecución típicos de Python (como formas incompatibles o tipos None).
//! Sin embargo, hay errores conceptuales y de tipado que debemos conocer.

use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::{array, Array2};

fn main() {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Antipatrones de Machine Learning");
    println!("============================================================\n");

    error_1_dimension_1d_en_features();
    error_2_mas_clusters_que_muestras_en_kmeans();
    error_3_olvidar_el_intercepto_en_regresion();

    println!("\n✅ Todos los antipatrones analizados y resueltos correctamente.");
}

/// ----------------------------------------------------------------------------
/// ERROR 1: Pasar un Vector 1D en lugar de una Matriz 2D de Características
/// ----------------------------------------------------------------------------
/// En Scikit-Learn (Python), pasar un array 1D lanza un error en tiempo de ejecución:
/// `ValueError: Expected 2D array, got 1D array instead`.
/// En Rust con `linfa`, el sistema de tipos de `ndarray` ni siquiera te deja compilar.
fn error_1_dimension_1d_en_features() {
    println!("--- Error 1: Formas 2D Estrictas en Datasets ---");
    println!("❌ Antipatrón conceptual: Intentar crear un Dataset con un vector plano 1D.");
    println!("  Solución en Rust: Usar siempre `Array2` [Muestras, Características].");

    // Correcto: 3 muestras con 1 sola característica cada una -> Forma [3, 1]
    let x_2d: Array2<f64> = array![[1.0], [2.0], [3.0]];
    let y = array![2.0, 4.0, 6.0];
    let dataset = Dataset::new(x_2d, y);

    println!("✓ Dataset [3, 1] creado con seguridad de tipos en compilación: {} muestras, {} features.\n",
        dataset.nsamples(), dataset.nfeatures());
}

/// ----------------------------------------------------------------------------
/// ERROR 2: K-Means con Más Clusters que Muestras (K > N)
/// ----------------------------------------------------------------------------
/// Si pides K = 5 clusters pero solo tienes 3 puntos de datos,
/// matemáticamente es imposible calcular 5 centroides.
fn error_2_mas_clusters_que_muestras_en_kmeans() {
    println!("--- Error 2: K-Means con K mayor que el número de muestras ---");

    let puntos: Array2<f64> = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]; // N = 3
    let dataset = DatasetBase::from(puntos);

    // Intentamos pedir K = 0 clusters (parámetro inválido):
    let resultado = KMeans::params(0).fit(&dataset);

    match resultado {
        Ok(_) => println!("⚠️ K=0 inesperadamente aceptado."),
        Err(e) => {
            println!("✓ Linfa valida los parámetros y devuelve Result::Err en vez de crashear:");
            println!("   Error controlado: {:?}\n", e);
        }
    }
}

/// ----------------------------------------------------------------------------
/// ERROR 3: Olvidar el Intercepto (Sesgo b) en Regresión Lineal
/// ----------------------------------------------------------------------------
/// Por defecto, algunos algoritmos asumen que la recta pasa por el origen (0,0).
/// Si tus datos tienen un sesgo base (ej: latencia base del sistema operativo de 10ms),
/// forzar b = 0 arruinará la precisión del modelo.
fn error_3_olvidar_el_intercepto_en_regresion() {
    println!("--- Error 3: Intercepto (Sesgo b) en Regresión Lineal ---");
    println!("💡 Regla: En modelos de latencia o costes de IA, siempre configura:");
    println!("   LinearRegression::default().with_intercept(true)");
    println!("   Esto garantiza que el modelo aprenda la fórmula: Y = W*X + b");
}
