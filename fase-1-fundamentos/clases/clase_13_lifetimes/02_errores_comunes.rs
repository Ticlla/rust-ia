//! 🦀 CLASE 13: LABORATORIO DE ERRORES COMUNES - Lifetimes
//! Ejecutar con: cargo run --bin c13_errores

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 13 [LABORATORIO DE ERRORES]: Lifetimes ('a)");
    println!("========================================================\n");

    println!("--- Error 1: error[E0597] - Borrowed value does not live long enough ---");
    println!("❌ Si una vista apunta a un vector dentro de un bloque {{ }} y el vector muere,");
    println!("   intentar leer la vista fuera causa error[E0597].");
    println!("✅ Solución: El dueño de los datos debe vivir al menos tanto como la vista:");

    let datos_seguros = vec![10.0, 20.0];
    let ref_segura = &datos_seguros;
    println!("   Referencia con ámbito seguro garantizada: {:?}\n", ref_segura);

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}
