//! ============================================================================
//! 🎯 FASE 2 - CLASE 10: Suite de Pruebas Unitarias e Integración del Proyecto
//! ============================================================================
//!
//! Ejecutar el banco de pruebas completo:
//!     cargo test --bin c10_ejercicios
//!
//! Ejecutar el ejecutable de pruebas:
//!     cargo run --bin c10_ejercicios

// Importamos los componentes del proyecto:
#[path = "01_teoria.rs"]
mod proyecto;
use proyecto::*;

fn main() {
    println!("============================================================");
    println!("🎯 SUITE DE VALIDACIÓN: Proyecto Integrador de la Fase 2");
    println!("============================================================\n");
    println!("Ejecuta las pruebas unitarias e integración automáticas con:");
    println!("    cargo test --bin c10_ejercicios\n");

    // Pequeño test rápido en terminal:
    let tensor = TensorBuffer::from_slice(&[1.0, 2.0, 3.0]);
    let modelo = ModeloInferencia { version: 1, pesos: tensor };

    let resultado = modelo.predecir_paralelo(&[4.0, 5.0, 6.0]);
    println!("Prueba de inferencia paralela: 1*4 + 2*5 + 3*6 = {:.2}", resultado);
    assert_eq!(resultado, 32.0);

    println!("✅ Verificación completada con éxito.");
}

// =============================================================================
// PRUEBAS DE INTEGRACIÓN ASÍNCRONAS Y CONCURRENTES
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State;
    use axum::Json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, RwLock};
    use tokio::sync::mpsc;

    #[test]
    fn test_tensor_buffer_raii_y_dimensiones() {
        let datos = [0.1, 0.2, 0.3, 0.4];
        let tensor = TensorBuffer::from_slice(&datos);

        assert_eq!(tensor.len(), 4);
        assert!(!tensor.is_empty());
        assert_eq!(tensor.as_slice(), &datos);
        // Al salir del scope, 'Drop' se ejecuta automáticamente sin fugas.
    }

    #[test]
    fn test_inferencia_paralela_con_rayon() {
        let pesos = TensorBuffer::from_slice(&[0.5, -1.0, 2.0]);
        let modelo = ModeloInferencia { version: 1, pesos };

        let x = [10.0, 5.0, 2.0];
        let pred = modelo.predecir_paralelo(&x);

        // 10*0.5 + 5*(-1.0) + 2*2.0 = 5.0 - 5.0 + 4.0 = 4.0
        assert!((pred - 4.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_pipeline_asincrono_hot_reload_y_concurrencia() {
        let (tx, _rx) = mpsc::channel(5);
        let estado = Arc::new(AppState {
            modelo: Arc::new(RwLock::new(ModeloInferencia {
                version: 1,
                pesos: TensorBuffer::from_slice(&[1.0, 2.0]),
            })),
            total_inferencias: Arc::new(AtomicUsize::new(0)),
            dataloader_tx: tx,
        });

        // 1. Inferencia inicial con v1:
        let (_, Json(res1)) = predict_handler(
            State(estado.clone()),
            Json(PredictRequest { features: vec![3.0, 4.0] }),
        )
        .await;

        // 3*1.0 + 4*2.0 = 11.0
        assert_eq!(res1.prediccion, 11.0);
        assert_eq!(res1.modelo_version, 1);
        assert_eq!(estado.total_inferencias.load(Ordering::Relaxed), 1);

        // 2. Hot-reload a v2 con pesos [10.0, 10.0]:
        let (_, Json(reload_res)) = hot_reload_handler(
            State(estado.clone()),
            Json(ReloadRequest { nuevos_pesos: vec![10.0, 10.0] }),
        )
        .await;

        assert_eq!(reload_res["nueva_version"], 2);

        // 3. Segunda inferencia (debe usar los nuevos pesos inmediatamente):
        let (_, Json(res2)) = predict_handler(
            State(estado.clone()),
            Json(PredictRequest { features: vec![3.0, 4.0] }),
        )
        .await;

        // 3*10.0 + 4*10.0 = 70.0
        assert_eq!(res2.prediccion, 70.0);
        assert_eq!(res2.modelo_version, 2);
        assert_eq!(estado.total_inferencias.load(Ordering::Relaxed), 2);
    }
}
