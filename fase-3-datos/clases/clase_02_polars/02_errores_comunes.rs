//! ==============================================================================
//! CLASE 02: Errores Comunes y Antipatrones con `Polars`
//! ==============================================================================
//!
//! Al venir de Pandas, los ingenieros suelen trasladar vicios de Python a Polars:
//! iterar filas manualmente, abusar de la API Eager y no manejar correctamente
//! los valores nulos de Apache Arrow antes de pasar los datos a una red neuronal.

use polars::prelude::*;

fn main() -> PolarsResult<()> {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Antipatrones en Polars y DataFrames");
    println!("============================================================\n");

    error_1_antipatron_iterrows_vs_expresiones_vectorizadas()?;
    error_2_eager_vs_lazy_en_datasets_grandes()?;
    error_3_olvidar_collect_en_lazyframe()?;
    error_4_manejo_de_nulos_antes_de_alimentar_ia()?;

    println!("\n✅ Todos los antipatrones analizados y resueltos correctamente.");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 1: Iterar filas manualmente (El vicio de `iterrows()` de Pandas)
/// ----------------------------------------------------------------------------
/// En Python se suele hacer: `for row in df.iterrows(): ...`
/// En Polars, iterar fila por fila destruye la velocidad porque rompe la
/// arquitectura columnar de Arrow y desactiva las instrucciones vectoriales SIMD.
fn error_1_antipatron_iterrows_vs_expresiones_vectorizadas() -> PolarsResult<()> {
    println!("--- Error 1: Antipatrón iterrows() vs Expresiones Vectorizadas ---");

    let df = df!(
        "precio_usd" => &[100.0, 250.0, 50.0],
        "descuento" => &[0.10, 0.20, 0.05]
    )?;

    // ❌ Antipatrón LENTO: Bucle manual por filas
    // let mut total = Vec::new();
    // for i in 0..df.height() { ... }

    //  Patrón ÓPTIMO: Expresión vectorizada en Polars (corre a velocidad C/SIMD)
    let calculado = df
        .lazy()
        .with_column(
            (col("precio_usd") * (lit(1.0) - col("descuento"))).alias("precio_final")
        )
        .collect()?;

    println!("Resultado calculado vectorialmente:\n{}", calculado);
    println!("👉 Regla: NUNCA iteres filas. Escribe expresiones con `col(...)`.\n");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 2: Encadenar operaciones Eager en vez de usar LazyFrame
/// ----------------------------------------------------------------------------
/// En Pandas: cada paso crea un nuevo DataFrame en RAM.
/// Si haces `df.filter().select().sort()`, creas 3 copias intermedias en memoria.
fn error_2_eager_vs_lazy_en_datasets_grandes() -> PolarsResult<()> {
    println!("--- Error 2: Eager (Copias en RAM) vs Lazy (Optimizador) ---");

    let df = df!(
        "usuario" => &["Alice", "Bob", "Charlie", "David"],
        "tokens_consumidos" => &[1500, 200, 8500, 400],
        "activo" => &[true, false, true, false]
    )?;

    //  Patrón Óptimo: .lazy() permite al optimizador fusionar filtros y proyecciones
    let optimizado = df
        .lazy()
        .filter(col("activo").eq(lit(true)))
        .filter(col("tokens_consumidos").gt(1000))
        .select([col("usuario"), col("tokens_consumidos")])
        .collect()?;

    println!("Resultado optimizado con Predicate Pushdown:\n{}", optimizado);
    println!("👉 Regla: Para cualquier pipeline de datos, comienza con `.lazy()`.\n");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 3: Olvidar `.collect()?` en un LazyFrame
/// ----------------------------------------------------------------------------
/// Un `LazyFrame` no contiene datos cargados; es solo un plano arquitectónico.
fn error_3_olvidar_collect_en_lazyframe() -> PolarsResult<()> {
    println!("--- Error 3: Confundir un LazyFrame con un DataFrame evaluado ---");

    let df = df!("x" => &[1, 2, 3])?;

    let lazy = df.lazy().filter(col("x").gt(1));

    // Si inspeccionas `lazy.describe_plan()`, Polars imprime el ÁRBOL LÓGICO:
    let plan = lazy.describe_plan()?;
    println!("Representación de un LazyFrame (árbol de ejecución):\n{}", plan);

    // Para materializar los datos reales necesitas .collect():
    let evaluado = lazy.collect()?;
    println!("DataFrame real materializado con .collect():\n{}", evaluado);
    println!("👉 Regla: Un LazyFrame es la receta; `.collect()` es cocinar el plato.\n");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 4: Enviar valores nulos (`Null`) directamente a una Red Neuronal
/// ----------------------------------------------------------------------------
/// En Arrow, un valor nulo se representa con un bitmap de validez.
/// Si intentas convertir una columna con `Null` a un tensor `Array2<f32>`,
/// la red neuronal producirá `NaN` que destruirán los gradientes.
fn error_4_manejo_de_nulos_antes_de_alimentar_ia() -> PolarsResult<()> {
    println!("--- Error 4: Valores Nulos (Nulls) en Pipelines de IA ---");

    // Dataset con valores faltantes (nulls)
    let df_con_nulos = df!(
        "latencia" => &[Some(12.5), None, Some(15.2), None, Some(9.8)]
    )?;

    println!("Columna con nulos (prohibido para redes neuronales):\n{}", df_con_nulos);

    //  Imputar nulos con la media antes de enviar al modelo:
    let media = col("latencia").mean();
    let saneado = df_con_nulos
        .lazy()
        .with_column(
            col("latencia").fill_null(media).alias("latencia_saneada")
        )
        .collect()?;

    println!("Columna saneada con imputación de la media:\n{}", saneado);
    println!("👉 Regla: Siempre sanea con `.fill_null(...)` antes de pasar a tensores.");
    Ok(())
}
