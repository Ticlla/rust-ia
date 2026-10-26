//! 🦀 CLASE 14: LABORATORIO DE ERRORES COMUNES - Machine Learning
//! Ejecutar con: cargo run --bin c14_errores

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 14 [LABORATORIO DE ERRORES]: Machine Learning");
    println!("========================================================\n");

    println!("--- Error 1: Learning Rate Explosivo (Divergencia numérica / NaN) ---");
    println!("❌ Si usas un lr demasiado alto (ej. 10.0), los pesos explotan hacia infinito o NaN.");
    println!("✅ Solución: Monitorizar la pérdida en cada época y reducir el learning rate si diverge.\n");

    println!("--- Error 2: Muestras y Etiquetas con Dimensiones Inconsistentes ---");
    println!("❌ Entrenar con X de tamaño N y etiquetas Y de tamaño M provoca pánico en lenguajes desprotegidos.");
    println!("✅ Solución: Validar dimensiones con Result<T, String> antes de computar gradientes.\n");

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}
