//! 🦀 CLASE 03: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c03_ejercicios

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 03 [EJERCICIOS PRÁCTICOS]: Funciones");
    println!("========================================================\n");

    // Reto 1: Función Escalón (Heaviside)
    println!("--- Reto 1: Heaviside Step Function ---");
    let val_neg = escalon(-2.5);
    let val_pos = escalon(1.8);
    println!("Escalón(-2.5) = {}, Escalón(1.8) = {}", val_neg, val_pos);
    assert_eq!(val_neg, 0.0);
    assert_eq!(val_pos, 1.0);
    println!("✅ Reto 1 superado!\n");

    // Reto 2: Pérdida Cuadrática Simple
    println!("--- Reto 2: Función de Pérdida Cuadrática ---");
    let loss = perdida_cuadratica(3.5, 3.0);
    println!("Pérdida (y_pred=3.5, y_real=3.0) = {:.4}", loss);
    assert!((loss - 0.25).abs() < 1e-6);
    println!("✅ Reto 2 superado!\n");

    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 03 FUERON SUPERADOS!");
    println!("========================================================");
}

fn escalon(x: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

fn perdida_cuadratica(y_pred: f32, y_real: f32) -> f32 {
    let diff = y_pred - y_real;
    diff * diff
}
