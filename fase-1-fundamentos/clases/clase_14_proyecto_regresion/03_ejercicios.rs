//! 🦀 CLASE 14: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c14_ejercicios

fn calcular_r2(y_real: &[f32], y_pred: &[f32]) -> f32 {
    let media_y: f32 = y_real.iter().sum::<f32>() / y_real.len() as f32;
    let ss_tot: f32 = y_real.iter().map(|&y| (y - media_y).powi(2)).sum();
    let ss_res: f32 = y_real.iter().zip(y_pred.iter()).map(|(&y, &p)| (y - p).powi(2)).sum();
    1.0 - (ss_res / ss_tot)
}

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 14 [EJERCICIOS PRÁCTICOS]: Métrica R² (Coeficiente de Determinación)");
    println!("========================================================\n");

    let y_real = [3.0, 5.0, 7.0, 9.0];
    let y_pred = [2.9, 5.1, 7.0, 8.9]; // Predicciones casi perfectas
    let r2 = calcular_r2(&y_real, &y_pred);

    println!("Métrica R² calculada: {:.4} (Máximo teórico: 1.0)", r2);
    assert!(r2 > 0.98);

    println!("✅ Reto de métricas R² superado con éxito!\n");
    println!("========================================================");
    println!("🎉 ¡HAS COMPLETADO EL PROYECTO Y LA FASE 1 AL 100%!");
    println!("========================================================");
}
