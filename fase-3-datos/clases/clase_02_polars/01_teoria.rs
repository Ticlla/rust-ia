//! ==============================================================================
//! CLASE 02: Procesamiento Masivo de Datos con `Polars` y Apache Arrow
//! ==============================================================================
//!
//! En Machine Learning e Inteligencia Artificial, los datos casi nunca llegan
//! listos como matrices de números. Llegan como tablas gigantes con columnas de
//! texto, categorías, números y valores faltantes.
//!
//! En Python utilizamos `Pandas`. En Rust usamos `Polars`.
//! De hecho, la librería Polars de Python está escrita 100% en Rust.
//!
//! OBJETIVOS PEDAGÓGICOS:
//! 1. Entender la memoria Columnar (Apache Arrow) vs Fila por Fila.
//! 2. Dominar la API Perezosa (`LazyFrame`) y la optimización de consultas.
//! 3. Realizar Feature Engineering: filtros, transformaciones y agrupaciones.
//! 4. Construir el puente: Convertir columnas de Polars a un tensor `ndarray`.

use ndarray::Array2;
use polars::prelude::*;

fn main() -> PolarsResult<()> {
    println!("============================================================");
    println!("📊 CLASE 02: Procesamiento Masivo con Polars (Apache Arrow)");
    println!("============================================================\n");

    demostrar_memoria_columnar();
    demostrar_lazy_evaluation_y_optimizador()?;
    demostrar_feature_engineering()?;
    demostrar_puente_polars_a_ndarray()?;

    println!("\n✅ Demostración teórica de Polars completada con éxito.");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// 1. ARQUITECTURA APACHE ARROW: MEMORIA COLUMNAR
/// ----------------------------------------------------------------------------
/// En `ndarray` guardábamos matrices por filas (Row-Major).
/// En `Polars`, los datos se guardan estrictamente por COLUMNAS continuas en RAM.
///
/// Si una tabla tiene 50 columnas y 10 millones de filas:
/// - En Pandas (filas): Para calcular el promedio de 1 columna, la CPU tiene
///   que leer toda la tabla en RAM (desperdicio masivo de memoria y ancho de banda).
/// - En Polars (columnas): La columna entera está pegada en un solo bloque contiguo.
///   La CPU carga solo esa columna con instrucciones vectoriales SIMD.
fn demostrar_memoria_columnar() {
    println!("--- 1. Arquitectura Columnar: Series y DataFrames ---");

    // Una 'Series' es una columna fuertemente tipada y contigua en memoria Arrow:
    let modelos = Series::new("modelo".into(), &["Llama-3-8B", "Mistral-7B", "Gemma-7B", "Phi-3-Mini"]);
    let latencias = Series::new("latencia_ms".into(), &[45.2, 38.1, 41.5, 18.3]);
    let accuracy = Series::new("accuracy".into(), &[0.85, 0.82, 0.81, 0.76]);

    // Un DataFrame es una colección de columnas continuas con la misma altura:
    let n_filas = 4;
    let df = DataFrame::new(n_filas, vec![modelos.into(), latencias.into(), accuracy.into()]).unwrap();

    println!("DataFrame creado en memoria columnar Arrow:\n{}", df);
    println!("Número de filas: {}, Número de columnas: {}", df.height(), df.width());
    println!();
}

/// ----------------------------------------------------------------------------
/// 2. LA API PEREZOSA (LazyFrame) Y EL OPTIMIZADOR DE CONSULTAS
/// ----------------------------------------------------------------------------
/// En Pandas cada línea de código se ejecuta de inmediato (Eager), duplicando memoria.
/// En Polars, con `.lazy()` creamos un PLAN de ejecución.
///
/// El optimizador de Polars aplica:
/// - Predicate Pushdown: Mueve los filtros 'filter()' lo más cerca posible de la
///   lectura para descartar filas de inmediato.
/// - Projection Pushdown: Descarta del disco todas las columnas que no vas a usar.
fn demostrar_lazy_evaluation_y_optimizador() -> PolarsResult<()> {
    println!("--- 2. Lazy Evaluation (LazyFrame): Optimización Inteligente ---");

    let df = df!(
        "model_id" => &[1, 2, 3, 4, 5],
        "nombre" => &["GPT-4o", "Claude-3.5", "Llama-3-70B", "Mistral-Large", "Mini-Model"],
        "costo_por_1k_tokens" => &[0.005, 0.003, 0.0008, 0.002, 0.0001],
        "eval_score" => &[92.5, 93.1, 88.4, 87.9, 65.0]
    )?;

    // Construimos la consulta sin ejecutarla todavía:
    let plan_perezoso = df
        .lazy()
        // 1. Filtrar modelos de alto rendimiento (score >= 88.0)
        .filter(col("eval_score").gt_eq(88.0))
        // 2. Proyectar solo el nombre y el costo
        .select([col("nombre"), col("costo_por_1k_tokens"), col("eval_score")]);

    println!("💡 Plan de consulta construido (aún no se ejecutó en RAM).");

    // .collect() dispara el motor de ejecución multihilo en Rust:
    let resultado = plan_perezoso.collect()?;
    println!("Resultado tras optimización y ejecución multihilo:\n{}", resultado);
    println!();

    Ok(())
}

/// ----------------------------------------------------------------------------
/// 3. FEATURE ENGINEERING PARA MACHINE LEARNING
/// ----------------------------------------------------------------------------
/// Preparar datos para IA requiere normalizar columnas continuas.
/// Ejemplo: Escalado Min-Max: x_norm = (x - min) / (max - min)
fn demostrar_feature_engineering() -> PolarsResult<()> {
    println!("--- 3. Feature Engineering: Normalización de Columnas ---");

    let dataset = df!(
        "id" => &[101, 102, 103, 104],
        "tamano_contexto" => &[2048.0, 4096.0, 8192.0, 16384.0],
        "tps" => &[120.0, 95.0, 60.0, 30.0] // Tokens Por Segundo
    )?;

    println!("Dataset original antes de normalizar:\n{}", dataset);

    // En Polars usamos 'with_columns' con expresiones matemáticas vectorizadas:
    let min_ctx = col("tamano_contexto").min();
    let max_ctx = col("tamano_contexto").max();

    let transformado = dataset
        .lazy()
        .with_columns([
            // Normalización Min-Max de contexto a rango [0.0, 1.0]
            ((col("tamano_contexto") - min_ctx.clone()) / (max_ctx - min_ctx))
                .alias("contexto_normalizado"),
            // Relación de eficiencia: tps por cada 1,000 tokens
            (col("tps") / (col("tamano_contexto") / lit(1000.0)))
                .alias("eficiencia_relativa")
        ])
        .collect()?;

    println!("Dataset con nuevas características para entrenar:\n{}", transformado);
    println!();

    Ok(())
}

/// ----------------------------------------------------------------------------
/// 4. EL PUENTE: CONVERTIR UN DATAFRAME DE POLARS A UN TENSOR NDARRAY
/// ----------------------------------------------------------------------------
/// Una vez limpiado el dataset con Polars, necesitamos alimentar la red neuronal
/// con tensores de `ndarray::Array2<f32>` (lo que aprendimos en la Clase 01).
fn demostrar_puente_polars_a_ndarray() -> PolarsResult<()> {
    println!("--- 4. Puente: De DataFrame (Polars) a Tensor (ndarray) ---");

    let df_features = df!(
        "f1" => &[0.1f32, 0.4f32, 0.7f32],
        "f2" => &[0.2f32, 0.5f32, 0.8f32],
        "f3" => &[0.3f32, 0.6f32, 0.9f32]
    )?;

    println!("Características tabulares en Polars:\n{}", df_features);

    // Extraemos la matriz convirtiendo las columnas numéricas:
    let n_filas = df_features.height();
    let n_cols = df_features.width();

    let mut datos_planos: Vec<f32> = Vec::with_capacity(n_filas * n_cols);

    // Iteramos por filas extrayendo los valores contiguos:
    for i in 0..n_filas {
        for col_name in df_features.get_column_names() {
            let val = df_features.column(col_name)?.f32()?.get(i).unwrap();
            datos_planos.push(val);
        }
    }

    // Construimos el Array2 de ndarray:
    let tensor: Array2<f32> = Array2::from_shape_vec((n_filas, n_cols), datos_planos).unwrap();

    println!("Tensor de ndarray listo para alimentar el modelo:\n{:?}", tensor);
    println!("Shape del tensor: {:?}", tensor.shape());
    println!("✓ Datos listos para la capa densa Y = X · W + b.");

    Ok(())
}
