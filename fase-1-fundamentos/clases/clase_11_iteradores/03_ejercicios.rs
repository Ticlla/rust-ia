//! 🦀 CLASE 11: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c11_ejercicios

fn calcular_mse(y_pred: &[f32], y_true: &[f32]) -> f32 {
    let suma_cuadrados: f32 = y_pred.iter()
        .zip(y_true.iter())
        .map(|(&p, &t)| {
            let diff = p - t;
            diff * diff
        })
        .sum();
    suma_cuadrados / y_pred.len() as f32
}

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 11 [EJERCICIOS PRÁCTICOS]: Iteradores");
    println!("========================================================\n");

    let preds = [2.0, 4.0, 6.0];
    let reales = [2.5, 3.5, 6.0];
    let mse = calcular_mse(&preds, &reales);

    println!("MSE funcional calculado: {:.4}", mse);
    assert!((mse - 0.16666).abs() < 1e-4);

    println!("✅ Reto superado con éxito!\n");
    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 11 FUERON SUPERADOS!");
    println!("========================================================");
}
