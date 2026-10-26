//! 🦀 CLASE 04: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c04_ejercicios

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 04 [EJERCICIOS PRÁCTICOS]: Control de Flujo");
    println!("========================================================\n");

    // Reto 1: Early Stopping (Parada Temprana)
    println!("--- Reto 1: Early Stopping cuando Loss < 0.05 ---");
    let historial_loss = [1.20, 0.85, 0.40, 0.12, 0.03, 0.02, 0.01];
    let mut epocas_ejecutadas = 0;

    for &loss in &historial_loss {
        epocas_ejecutadas += 1;
        if loss < 0.05 {
            println!("  [Early Stopping] Criterio cumplido en época {} con loss {:.2}", epocas_ejecutadas, loss);
            break;
        }
    }
    assert_eq!(epocas_ejecutadas, 5);
    println!("✅ Reto 1 superado!\n");

    // Reto 2: Suma de épocas con 'while'
    println!("--- Reto 2: Contador de Épocas ---");
    let mut contador = 0;
    while contador < 10 {
        contador += 1;
    }
    assert_eq!(contador, 10);
    println!("✅ Reto 2 superado!\n");

    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 04 FUERON SUPERADOS!");
    println!("========================================================");
}
