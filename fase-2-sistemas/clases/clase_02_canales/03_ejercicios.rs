//! ============================================================================
//! 🎯 FASE 2 - CLASE 02: Retos Prácticos con Canales (`mpsc`)
//! ============================================================================
//!
//! Ejecutar el reto:
//!     cargo run --bin c02_ejercicios
//!
//! Ejecutar las pruebas unitarias:
//!     cargo test --bin c02_ejercicios

use std::sync::mpsc;
use std::thread;

fn main() {
    println!("========================================================");
    println!("🎯 RETO PRÁCTICO: DataLoader Paralelo de Normalización");
    println!("========================================================\n");

    let trozos_datos = vec![
        vec![10.0, 20.0, 30.0],
        vec![40.0, 50.0, 60.0],
        vec![70.0, 80.0, 90.0],
    ];

    println!("Cargando y normalizando 3 trozos de datos en hilos paralelos...");
    let resultado = cargar_y_normalizar_paralelo(trozos_datos);

    println!("Total lotes procesados por el canal: {}", resultado.len());
    for (id, lote) in &resultado {
        println!("  Lote #{id} normalizado: {:?}", lote);
    }

    println!("\n✅ Reto ejecutado exitosamente.");
}

/// Normaliza trozos de datos en hilos paralelos y los envía por un canal MPSC.
/// Cada hilo recibe un vector crudo, normaliza dividiendo entre 100.0,
/// y envía la tupla `(id, vector_normalizado)` al hilo principal.
pub fn cargar_y_normalizar_paralelo(trozos: Vec<Vec<f32>>) -> Vec<(usize, Vec<f32>)> {
    let (tx, rx) = mpsc::channel();

    for (id, mut datos) in trozos.into_iter().enumerate() {
        let tx_clon = tx.clone();
        thread::spawn(move || {
            // Normalización simple in-place: valor / 100.0
            for val in &mut datos {
                *val /= 100.0;
            }
            tx_clon.send((id, datos)).unwrap();
        });
    }

    // Cerramos el transmisor de main para que 'rx' no espere infinitamente:
    drop(tx);

    let mut recolectados = Vec::new();
    for elemento in rx {
        recolectados.push(elemento);
    }

    // Ordenamos por ID para consistencia
    recolectados.sort_by_key(|&(id, _)| id);
    recolectados
}

// =============================================================================
// PRUEBAS UNITARIAS AUTOMÁTICAS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader_paralelo_correcto() {
        let datos = vec![
            vec![100.0, 200.0],
            vec![50.0],
        ];

        let resultado = cargar_y_normalizar_paralelo(datos);

        assert_eq!(resultado.len(), 2);
        assert_eq!(resultado[0], (0, vec![1.0, 2.0]));
        assert_eq!(resultado[1], (1, vec![0.5]));
    }

    #[test]
    fn test_dataloader_vacio() {
        let datos: Vec<Vec<f32>> = Vec::new();
        let resultado = cargar_y_normalizar_paralelo(datos);
        assert!(resultado.is_empty());
    }
}
