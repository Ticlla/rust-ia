//! 🦀 CLASE 11: LABORATORIO DE ERRORES COMUNES - Iteradores
//! Ejecutar con: cargo run --bin c11_errores

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 11 [LABORATORIO DE ERRORES]: Iteradores");
    println!("========================================================\n");

    println!("--- Error 1: Olvidar .collect() para materializar el iterador ---");
    println!("❌ Los iteradores en Rust son perezosos (lazy). Si solo haces .map() no ocurre ningún cálculo.");
    println!("✅ Solución: Usar un consumidor terminal como .sum(), .count(), o .collect():");

    let v = vec![1, 2, 3];
    let duplicados: Vec<i32> = v.iter().map(|&x| x * 2).collect();
    println!("   Colección materializada con .collect(): {:?}\n", duplicados);

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}
