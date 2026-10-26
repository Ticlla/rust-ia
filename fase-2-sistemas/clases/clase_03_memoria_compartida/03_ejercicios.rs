//! ============================================================================
//! 🎯 FASE 2 - CLASE 03: Retos Prácticos con `Arc<Mutex<T>>`
//! ============================================================================
//!
//! Ejecutar el reto:
//!     cargo run --bin c03_ejercicios
//!
//! Ejecutar las pruebas unitarias:
//!     cargo test --bin c03_ejercicios

use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    println!("========================================================");
    println!("🎯 RETO PRÁCTICO: Servidor de Parámetros Concurrente");
    println!("========================================================\n");

    let pesos_iniciales = vec![1.0, 2.0, 3.0];
    println!("Pesos iniciales antes de entrenar: {:?}", pesos_iniciales);

    // 4 gradientes calculados por 4 workers en paralelo:
    let gradientes_workers = vec![
        vec![0.1, 0.1, 0.1],
        vec![0.2, 0.2, 0.2],
        vec![-0.1, -0.1, -0.1],
        vec![0.05, 0.05, 0.05],
    ];

    let lr = 1.0; // learning rate = 1.0 para verificar sumas directas
    let pesos_actualizados = entrenar_paralelo_arc_mutex(pesos_iniciales, gradientes_workers, lr);

    println!("Pesos finales tras 4 workers concurrentes: {:?}", pesos_actualizados);
    println!("Esperado: [1.0 - 0.25 = 0.75, 2.0 - 0.25 = 1.75, 3.0 - 0.25 = 2.75]");

    println!("\n✅ Reto ejecutado exitosamente.");
}

/// Aplica gradientes de múltiples trabajadores concurrentes sobre un modelo
/// protegido por `Arc<Mutex<Vec<f32>>>`.
/// Fórmula de actualización: W = W - lr * grad
pub fn entrenar_paralelo_arc_mutex(
    pesos_iniciales: Vec<f32>,
    gradientes_por_worker: Vec<Vec<f32>>,
    lr: f32,
) -> Vec<f32> {
    let dim = pesos_iniciales.len();
    let modelo = Arc::new(Mutex::new(pesos_iniciales));
    let mut handles = Vec::new();

    for gradientes in gradientes_por_worker {
        assert_eq!(gradientes.len(), dim, "Dimensiones de gradientes incompatibles");
        let modelo_clon = Arc::clone(&modelo);

        let h = thread::spawn(move || {
            // Adquirimos el cerrojo exclusivo:
            let mut pesos = modelo_clon.lock().unwrap();
            for (w, &g) in pesos.iter_mut().zip(gradientes.iter()) {
                *w -= lr * g;
            }
        });
        handles.push(h);
    }

    for h in handles {
        h.join().unwrap();
    }

    // Extraemos el valor final. Como todos los hilos terminaron,
    // 'Arc::try_unwrap' nos permite recuperar el Mutex y su contenido sin clonar:
    let mutex_final = Arc::try_unwrap(modelo).expect("Quedaban hilos vivos con referencias al Arc");
    mutex_final.into_inner().unwrap()
}

// =============================================================================
// PRUEBAS UNITARIAS AUTOMÁTICAS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actualizacion_concurrente_correcta() {
        let pesos = vec![10.0, 20.0];
        let gradientes = vec![
            vec![1.0, 2.0],  // Worker 1
            vec![3.0, 4.0],  // Worker 2
            vec![-1.0, -2.0] // Worker 3
        ];
        // Suma neta de gradientes: [3.0, 4.0]
        // Con lr = 0.5:
        // W[0] = 10.0 - 0.5 * 3.0 = 8.5
        // W[1] = 20.0 - 0.5 * 4.0 = 18.0
        let resultado = entrenar_paralelo_arc_mutex(pesos, gradientes, 0.5);

        assert!((resultado[0] - 8.5).abs() < 1e-6);
        assert!((resultado[1] - 18.0).abs() < 1e-6);
    }

    #[test]
    fn test_sin_gradientes() {
        let pesos = vec![5.0, 6.0];
        let resultado = entrenar_paralelo_arc_mutex(pesos, vec![], 0.1);
        assert_eq!(resultado, vec![5.0, 6.0]);
    }
}
