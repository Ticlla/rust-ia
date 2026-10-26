//! ==============================================================================
//! CLASE 01: Machine Learning Clásico con `linfa` (El Scikit-Learn de Rust)
//! ==============================================================================
//!
//! En esta clase aprendemos el ciclo de vida de Machine Learning en Rust:
//! 1. `Dataset`: Envoltorio tipado de características (`features`) y objetivos (`targets`).
//! 2. Algoritmos Supervisados: Regresión Lineal (`linfa-linear`).
//! 3. Algoritmos No Supervisados: K-Means Clustering (`linfa-clustering`).
//! 4. Inferencia y métricas de error (Mean Squared Error).

use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_linear::LinearRegression;
use ndarray::{array, Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("🤖 CLASE 01: MACHINE LEARNING CLÁSICO CON LINFA");
    println!("============================================================\n");

    demostrar_regresion_lineal()?;
    demostrar_clustering_kmeans()?;

    println!("✅ Demostración teórica de linfa completada con éxito.");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// 1. REGRESIÓN LINEAL SUPERVISADA
/// ----------------------------------------------------------------------------
/// Problema de IA: Predecir la latencia de inferencia de un LLM (en ms)
/// en base a dos variables:
///   - Columna 0: Longitud del prompt (número de tokens)
///   - Columna 1: Tamaño del batch (número de peticiones concurrentes)
fn demostrar_regresion_lineal() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 1. Regresión Lineal: Predicción de Latencia de Inferencia ---");

    // Matriz de entrenamiento X: [N = 6 muestras, D = 2 características]
    let x: Array2<f64> = array![
        [100.0, 1.0],  // 100 tokens, batch 1 -> Latencia real: 15.0 ms
        [200.0, 1.0],  // 200 tokens, batch 1 -> Latencia real: 25.0 ms
        [500.0, 1.0],  // 500 tokens, batch 1 -> Latencia real: 55.0 ms
        [100.0, 4.0],  // 100 tokens, batch 4 -> Latencia real: 30.0 ms
        [200.0, 4.0],  // 200 tokens, batch 4 -> Latencia real: 40.0 ms
        [500.0, 4.0],  // 500 tokens, batch 4 -> Latencia real: 70.0 ms
    ];

    // Vector de etiquetas reales Y: [N = 6 latencias en ms]
    let y: Array1<f64> = array![15.0, 25.0, 55.0, 30.0, 40.0, 70.0];

    // Construcción del Dataset en Linfa (X features + Y targets):
    let dataset = Dataset::new(x, y);
    println!("✓ Dataset creado: {} muestras con {} variables.", dataset.nsamples(), dataset.nfeatures());

    // Ajuste del modelo (Entrenamiento con Mínimos Cuadrados Ordinarios):
    let modelo = LinearRegression::default()
        .with_intercept(true)
        .fit(&dataset)?;

    println!("✓ Modelo de Regresión Lineal entrenado:");
    println!("   • Coeficientes (Pesos W): {:?}", modelo.params());
    println!("   • Sesgo (Intercepto b):   {:.4}", modelo.intercept());

    // Inferencia sobre un nuevo escenario no visto:
    // Prompt de 350 tokens con batch size de 2:
    let nuevo_escenario: Array2<f64> = array![[350.0, 2.0]];
    let prediccion = modelo.predict(&nuevo_escenario);

    println!("\n🔮 Inferencia:");
    println!("   Entrada: 350 tokens, Batch 2");
    println!("   Latencia estimada por el modelo: {:.2} ms\n", prediccion[0]);

    Ok(())
}

/// ----------------------------------------------------------------------------
/// 2. CLUSTERING NO SUPERVISADO CON K-MEANS
/// ----------------------------------------------------------------------------
/// Problema de IA: En Vector Databases (ej. FAISS / Qdrant), el algoritmo
/// IVFFlat usa K-Means para agrupar 1 millón de embeddings en K centroides.
/// En la búsqueda, el query solo se compara contra el centroide más cercano.
fn demostrar_clustering_kmeans() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 2. Clustering No Supervisado: Agrupamiento con K-Means ---");

    // Simulamos 6 vectores de embeddings en 2 dimensiones:
    // Grupo A: Embeddings de programación (cercanos a [1.0, 1.0])
    // Grupo B: Embeddings de gastronomía (cercanos a [9.0, 9.0])
    let embeddings: Array2<f64> = array![
        [1.1, 1.2], // Doc 0: Rust
        [0.9, 0.8], // Doc 1: Python
        [1.3, 1.0], // Doc 2: C++
        [8.9, 9.1], // Doc 3: Pizza
        [9.2, 8.8], // Doc 4: Pasta
        [9.0, 9.3], // Doc 5: Ensalada
    ];

    let dataset = DatasetBase::from(embeddings);

    // Entrenamos K-Means para descubrir K = 2 clusters:
    let n_clusters = 2;
    let modelo_kmeans = KMeans::params(n_clusters)
        .max_n_iterations(100)
        .tolerance(1e-4)
        .fit(&dataset)?;

    println!("✓ K-Means convergió exitosamente en {} centroides:", n_clusters);
    let centroides = modelo_kmeans.centroids();
    for (k, centroide) in centroides.rows().into_iter().enumerate() {
        println!("   Cluster {}: Centroide en {:?}", k, centroide.to_vec());
    }

    // Clasificar un nuevo embedding no visto:
    // "Compilador y punteros" -> vector [1.0, 0.9]
    let query_embedding: Array2<f64> = array![[1.0, 0.9]];
    let cluster_asignado = modelo_kmeans.predict(&query_embedding);

    println!("\n🔮 Clasificación No Supervisada de Query:");
    println!("   Query: [1.0, 0.9] (Semántica técnica/sistemas)");
    println!("   Cluster asignado por K-Means: #{}\n", cluster_asignado[0]);

    Ok(())
}
