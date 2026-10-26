//! 🦀 CLASE 06: LABORATORIO DE ERRORES COMUNES - Borrowing
//! Ejecutar con: cargo run --bin c06_errores

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 06 [LABORATORIO DE ERRORES]: Borrowing");
    println!("========================================================\n");

    println!("--- Error 1: error[E0502] - Conflicto entre &mut y & simultáneos ---");
    println!("❌ Código erróneo:");
    println!("   let mut buffer = vec![1.0, 2.0];");
    println!("   let ref_lectura = &buffer;");
    println!("   buffer.push(3.0); // 💥 Modifica mientras ref_lectura está viva");
    println!("   println!(\"{{:?}}\", ref_lectura);");
    println!("\n💡 ¿Por qué ocurre?:");
    println!("   Si el buffer se reubica en memoria al crecer, ref_lectura apuntaría a memoria inválida.");
    println!("✅ Solución: Terminar de usar la lectura antes de pedir un préstamo mutable:\n");

    let mut buffer = vec![1.0, 2.0];
    let longitud = buffer.len(); // Leemos y terminamos
    buffer.push(3.0);            // Ahora es seguro modificar
    println!("   Solución: longitud previa = {}, nuevo buffer = {:?}\n", longitud, buffer);

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}
