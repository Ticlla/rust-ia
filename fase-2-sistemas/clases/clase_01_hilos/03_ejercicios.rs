//! ============================================================================
//! 🎯 FASE 2 - CLASE 01: Retos Prácticos con Hilos
//! ============================================================================
//!
//! Ejecutar el reto:
//!     cargo run --bin c01_ejercicios
//!
//! Ejecutar las pruebas unitarias automáticas:
//!     cargo test --bin c01_ejercicios

use std::thread;

fn main() {
    println!("========================================================");
    println!("🎯 RETO PRÁCTICO: Producto Punto Paralelo en 2 Hilos");
    println!("========================================================\n");

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

    let resultado = producto_punto_paralelo(&x, &w);
    println!("Entradas (x) : {:?}", x);
    println!("Pesos (w)    : {:?}", w);
    println!("Producto punto paralelo = {resultado:.4}");
    println!("(Esperado: 1*0.5 + 2*0.5 + 3*0.5 + 4*0.5 + 5*0.5 + 6*0.5 = 10.5)");

    println!("\n✅ Reto ejecutado exitosamente.");
}

/// Divide la computación del producto punto entre dos hilos del CPU.
/// Hilo 1: procesa la primera mitad del vector.
/// Hilo 2: procesa la segunda mitad del vector.
/// Retorna la suma total de ambos resultados.
pub fn producto_punto_paralelo(x: &[f32], w: &[f32]) -> f32 {
    assert_eq!(x.len(), w.len(), "Los vectores deben tener la misma dimensión");
    if x.is_empty() {
        return 0.0;
    }

    let mid = x.len() / 2;

    // Convertimos a vectores independientes para que cada hilo tome posesión con 'move':
    let x_part1 = x[..mid].to_vec();
    let w_part1 = w[..mid].to_vec();

    let x_part2 = x[mid..].to_vec();
    let w_part2 = w[mid..].to_vec();

    // Hilo 1: primera mitad
    let handle1 = thread::spawn(move || {
        x_part1.iter().zip(w_part1.iter()).map(|(&a, &b)| a * b).sum::<f32>()
    });

    // Hilo 2: segunda mitad
    let handle2 = thread::spawn(move || {
        x_part2.iter().zip(w_part2.iter()).map(|(&a, &b)| a * b).sum::<f32>()
    });

    // Esperamos a ambos hilos y sumamos los resultados parciales:
    let parcial1 = handle1.join().unwrap();
    let parcial2 = handle2.join().unwrap();

    parcial1 + parcial2
}

// =============================================================================
// PRUEBAS UNITARIAS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_producto_punto_paralelo_correcto() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 2.0, 2.0, 2.0];
        // 1*2 + 2*2 + 3*2 + 4*2 = 2 + 4 + 6 + 8 = 20.0
        assert_eq!(producto_punto_paralelo(&a, &b), 20.0);
    }

    #[test]
    fn test_producto_punto_vacio() {
        assert_eq!(producto_punto_paralelo(&[], &[]), 0.0);
    }
}
