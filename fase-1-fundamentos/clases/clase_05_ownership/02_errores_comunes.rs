//! 🦀 CLASE 05: LABORATORIO DE ERRORES COMUNES - Ownership
//! Ejecutar con: cargo run --bin c05_errores

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 05 [LABORATORIO DE ERRORES]: Ownership (Move Semantics)");
    println!("========================================================\n");

    println!("--- Error 1: error[E0382] - Borrow of moved value ---");
    println!("❌ Código erróneo:");
    println!("   let dataset = vec![1.0, 2.0, 3.0];");
    println!("   let consumer = dataset; // MOVE");
    println!("   println!(\"{{:?}}\", dataset); // 💥 ¡dataset ya no es dueño!");
    println!("\n💡 ¿Por qué ocurre?:");
    println!("   Para evitar doble liberación de memoria (Double Free), Rust invalida 'dataset'.");
    println!("✅ Solución: Si necesitas que ambos lean, usa referencias (&) o clona (.clone()).");

    let dataset = vec![1.0, 2.0, 3.0];
    let consumer = &dataset; // Préstamo con & en lugar de MOVE
    println!("   Solución con préstamo &: dataset={:?}, ref={:?}\n", dataset, consumer);

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}
