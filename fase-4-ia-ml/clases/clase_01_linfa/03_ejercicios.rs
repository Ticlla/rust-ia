//! ==============================================================================
//! CLASE 01: Ejercicios Prácticos de Machine Learning con `linfa`
//! ==============================================================================
//!
//! En este laboratorio implementarás y validarás:
//! 1. `entrenar_regresor_lineal`: Ajustar un modelo lineal con intercepto y hacer inferencias.
//! 2. `clasificar_con_kmeans`: Entrenar K-Means y asignar clusters a nuevos vectores.
//! 3. `calcular_error_cuadratico_medio`: Función matemática de evaluación (MSE).

use linfa::prelude::*;
use linfa_clustering::KMeans;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};

fn main() {
    println!("============================================================");
    println!("🧪 EJERCICIOS CLASE 01: Machine Learning con linfa");
    println!("============================================================\n");
    println!("Ejecuta 'cargo test --bin c01_ejercicios' para evaluar tus soluciones.");
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 1: Entrenar un Regresor Lineal y Predecir
/// ----------------------------------------------------------------------------
/// Dado un Dataset de características y etiquetas, ajusta una regresión lineal
/// con intercepto y predice la salida para `nuevas_entradas`.
pub fn entrenar_y_predecir_lineal(
    x_train: Array2<f64>,
    y_train: Array1<f64>,
    x_test: &Array2<f64>,
) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    let dataset = Dataset::new(x_train, y_train);
    let modelo = LinearRegression::default()
        .with_intercept(true)
        .fit(&dataset)?;

    let predicciones = modelo.predict(x_test);
    Ok(predicciones)
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 2: Entrenar K-Means y Agrupar Vectores
/// ----------------------------------------------------------------------------
/// Entrena un modelo K-Means con `k` clusters sobre la matriz de vectores
/// y retorna el índice del cluster asignado a `query`.
pub fn entrenar_kmeans_y_asignar(
    puntos: Array2<f64>,
    k: usize,
    query: &Array2<f64>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let dataset = DatasetBase::from(puntos);
    let modelo = KMeans::params(k)
        .max_n_iterations(200)
        .tolerance(1e-4)
        .fit(&dataset)?;

    let asignaciones = modelo.predict(query);
    Ok(asignaciones[0])
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 3: Calcular Mean Squared Error (MSE)
/// ----------------------------------------------------------------------------
/// MSE = (1 / N) * sum((y_pred[i] - y_real[i])^2)
pub fn calcular_mse(y_pred: &Array1<f64>, y_real: &Array1<f64>) -> f64 {
    assert_eq!(y_pred.len(), y_real.len(), "Longitudes incompatibles");
    let mut suma_error = 0.0;
    for i in 0..y_pred.len() {
        let diff = y_pred[i] - y_real[i];
        suma_error += diff * diff;
    }
    suma_error / (y_pred.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_entrenar_regresor_lineal() -> Result<(), Box<dyn std::error::Error>> {
        // Datos: y = 2.0 * x + 1.0
        let x_train = array![[1.0], [2.0], [3.0], [4.0]];
        let y_train = array![3.0, 5.0, 7.0, 9.0];

        let x_test = array![[5.0], [10.0]];
        let preds = entrenar_y_predecir_lineal(x_train, y_train, &x_test)?;

        // Para x = 5.0 -> y = 11.0
        // Para x = 10.0 -> y = 21.0
        assert!((preds[0] - 11.0).abs() < 1e-4);
        assert!((preds[1] - 21.0).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_kmeans_separacion_clusters() -> Result<(), Box<dyn std::error::Error>> {
        // Dos grupos claramente separados
        let puntos = array![
            [0.1, 0.1],
            [0.2, 0.1],
            [0.1, 0.2],
            [10.1, 10.1],
            [10.2, 10.0],
            [10.0, 10.2]
        ];

        let q1 = array![[0.15, 0.15]]; // Cerca del grupo 0
        let q2 = array![[10.15, 10.05]]; // Cerca del grupo 1

        let c1 = entrenar_kmeans_y_asignar(puntos.clone(), 2, &q1)?;
        let c2 = entrenar_kmeans_y_asignar(puntos, 2, &q2)?;

        // Deben pertenecer a clusters diferentes
        assert_ne!(c1, c2);
        Ok(())
    }

    #[test]
    fn test_calculo_mse() {
        let y_pred = array![2.0, 4.0, 6.0];
        let y_real = array![2.0, 5.0, 5.0];

        // Errores: (0)^2 + (-1)^2 + (1)^2 = 0 + 1 + 1 = 2
        // MSE = 2 / 3 = 0.6666...
        let mse = calcular_mse(&y_pred, &y_real);
        assert!((mse - (2.0 / 3.0)).abs() < 1e-5);
    }
}
