//! 🦀 CLASE 07: LABORATORIO DE ERRORES COMUNES - Slices
//! Ejecutar con: cargo run --bin c07_errores

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 07 [LABORATORIO DE ERRORES]: Slices");
    println!("========================================================\n");

    println!("--- Error 1: Pánico por Rango Fuera de Límites (Out of bounds) ---");
    println!("❌ Si haces: let s = &datos[0..10]; en un arreglo de 5 elementos, el programa hace panic.");
    println!("✅ Solución: Validar la longitud o usar métodos seguros con .get(rango):");
    let datos = [1.0, 2.0, 3.0];
    if let Some(sub) = datos.get(0..2) {
        println!("   Extracción segura con .get: {:?}\n", sub);
    }

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}
