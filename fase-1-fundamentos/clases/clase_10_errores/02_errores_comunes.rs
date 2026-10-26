//! 🦀 CLASE 10: LABORATORIO DE ERRORES COMUNES - Option y Result
//! Ejecutar con: cargo run --bin c10_errores

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 10 [LABORATORIO DE ERRORES]: Option y Result");
    println!("========================================================\n");

    println!("--- Error 1: Pánico por .unwrap() incondicional ---");
    println!("❌ Si haces let val = opt.unwrap(); sobre un None o Err, el programa colapsa en pánico.");
    println!("✅ Solución: Usar match, unwrap_or(valor_defecto), o '?' en funciones fallibles:");
    let opt_vacio: Option<f32> = None;
    let fallback = opt_vacio.unwrap_or(0.0);
    println!("   Rescate seguro con unwrap_or: {}\n", fallback);

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}
