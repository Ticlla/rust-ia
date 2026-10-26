//! ==============================================================================
//! CLASE 02: Ejercicios Prácticos con `Polars` para IA
//! ==============================================================================
//!
//! En este laboratorio resolverás 3 tareas críticas en pipelines de datos de IA:
//! 1. Filtrado y Ranking de Modelos con la API Lazy.
//! 2. Saneamiento e Imputación de Valores Faltantes (Nulls).
//! 3. Extracción de un Tensor `Array2<f32>` a partir de un DataFrame de Polars.

use ndarray::Array2;
use polars::prelude::*;

fn main() {
    println!("============================================================");
    println!("🧪 EJERCICIOS CLASE 02: Polars y Feature Engineering para IA");
    println!("============================================================\n");
    println!("Ejecuta 'cargo test --bin c02_ejercicios' para evaluar tus soluciones.");
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 1: Filtrado y Ranking de Modelos Eficientes con LazyFrame
/// ----------------------------------------------------------------------------
/// Dado un DataFrame de modelos con columnas:
///   "nombre" (str), "accuracy" (f64), "latencia_ms" (f64)
///
/// Implementa una función que usando la API Lazy:
/// 1. Filtre solo los modelos con `accuracy >= 0.80` y `latencia_ms <= 50.0`.
/// 2. Cree una nueva columna llamada `"eficiencia"` calculada como:
///    accuracy / (latencia_ms / 1000.0)
/// 3. Retorne el DataFrame evaluado.
pub fn filtrar_y_calcular_eficiencia(df: &DataFrame) -> PolarsResult<DataFrame> {
    df.clone()
        .lazy()
        .filter(col("accuracy").gt_eq(0.80))
        .filter(col("latencia_ms").lt_eq(50.0))
        .with_column(
            (col("accuracy") / (col("latencia_ms") / lit(1000.0))).alias("eficiencia")
        )
        .collect()
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 2: Saneamiento e Imputación de Nulos en Embeddings/Métricas
/// ----------------------------------------------------------------------------
/// Si una columna de características tiene valores nulos (`Null`), la red
/// neuronal generará `NaN`.
/// Implementa una función que reemplace todos los nulos de la columna especificada
/// por un valor constante de relleno (por ejemplo 0.0 o la media).
pub fn imputar_nulos_columna(
    df: &DataFrame,
    columna: &str,
    valor_relleno: f64,
) -> PolarsResult<DataFrame> {
    df.clone()
        .lazy()
        .with_column(
            col(columna).fill_null(lit(valor_relleno)).alias(columna)
        )
        .collect()
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 3: El Puente Polars -> ndarray (Extracción de Tensores)
/// ----------------------------------------------------------------------------
/// Extrae las columnas indicadas en `columnas_features` de un DataFrame de Polars
/// y constrúyelas en un tensor 2D continuo `Array2<f32>` de `ndarray`.
pub fn dataframe_a_tensor_features(
    df: &DataFrame,
    columnas_features: &[&str],
) -> PolarsResult<Array2<f32>> {
    let n_filas = df.height();
    let n_cols = columnas_features.len();
    let mut buffer = Vec::with_capacity(n_filas * n_cols);

    for i in 0..n_filas {
        for &col_name in columnas_features {
            let col = df.column(col_name)?;
            let val = col.f32()?.get(i).unwrap();
            buffer.push(val);
        }
    }

    Ok(Array2::from_shape_vec((n_filas, n_cols), buffer).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filtrar_y_calcular_eficiencia() {
        let df = df!(
            "nombre" => &["Model-A", "Model-B", "Model-C", "Model-D"],
            "accuracy" => &[0.85, 0.70, 0.90, 0.82],
            "latencia_ms" => &[40.0, 30.0, 100.0, 25.0]
        )
        .unwrap();

        let res = filtrar_y_calcular_eficiencia(&df).unwrap();

        // Model-B descartado (accuracy < 0.80)
        // Model-C descartado (latencia > 50.0)
        // Quedan Model-A y Model-D:
        assert_eq!(res.height(), 2);

        let nombres = res.column("nombre").unwrap();
        let nombres_str: Vec<&str> = nombres.str().unwrap().iter().flatten().collect();
        assert_eq!(nombres_str, vec!["Model-A", "Model-D"]);

        // Eficiencia de Model-D: 0.82 / (25 / 1000) = 0.82 / 0.025 = 32.8
        let eficiencia = res.column("eficiencia").unwrap();
        let val_d = eficiencia.f64().unwrap().get(1).unwrap();
        assert!((val_d - 32.8).abs() < 1e-4);
    }

    #[test]
    fn test_imputar_nulos() {
        let df = df!(
            "embedding_loss" => &[Some(0.5), None, Some(0.2), None]
        )
        .unwrap();

        let saneado = imputar_nulos_columna(&df, "embedding_loss", 0.0).unwrap();
        let col = saneado.column("embedding_loss").unwrap();

        assert_eq!(col.null_count(), 0);
        let vals: Vec<f64> = col.f64().unwrap().into_no_null_iter().collect();
        assert_eq!(vals, vec![0.5, 0.0, 0.2, 0.0]);
    }

    #[test]
    fn test_dataframe_a_tensor_features() {
        let df = df!(
            "f1" => &[1.0f32, 3.0f32],
            "f2" => &[2.0f32, 4.0f32],
            "etiqueta_texto" => &["clase_a", "clase_b"]
        )
        .unwrap();

        let tensor = dataframe_a_tensor_features(&df, &["f1", "f2"]).unwrap();

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor[[0, 0]], 1.0);
        assert_eq!(tensor[[0, 1]], 2.0);
        assert_eq!(tensor[[1, 0]], 3.0);
        assert_eq!(tensor[[1, 1]], 4.0);
    }
}
