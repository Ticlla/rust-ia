//! 🦀 CLASE 04: LABORATORIO DE ERRORES COMUNES - Control de Flujo
//! Ejecutar con: cargo run --bin c04_errores

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 04 [LABORATORIO DE ERRORES]: Control de Flujo");
    println!("========================================================\n");

    println!("--- Error 1: Bucle infinito por comparación flotante estricta (loss == 0.0) ---");
    println!("❌ En computación numérica, los floats raramente son exactamente 0.0.");
    println!("   while loss != 0.0 {{ ... }} suele ser un bucle infinito.");
    println!("✅ Solución: Usa umbrales de tolerancia épsilon (ej. loss < 1e-4):");
    let mut loss: f32 = 1.0;
    while loss > 1e-3 {
        loss *= 0.5;
    }
    println!("   Convergencia robusta por umbral épsilon: {}\n", loss);

    println!("--- Error 2: Tipos incompatibles en múltiples 'break' ---");
    println!("❌ Si un break retorna 1.0 (f32) y otro retorna 2 (entero), el compilador falla.");
    println!("✅ Solución: Todos los puntos de salida 'break' deben retornar el mismo tipo.");

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}
