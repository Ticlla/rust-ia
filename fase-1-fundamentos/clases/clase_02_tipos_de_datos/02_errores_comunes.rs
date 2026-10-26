//! 🦀 CLASE 02: LABORATORIO DE ERRORES COMUNES - Tipos de Datos
//! Ejecutar con: cargo run --bin c02_errores

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 02 [LABORATORIO DE ERRORES]: Tipos de Datos");
    println!("========================================================\n");

    // Error 1: Operaciones matemáticas entre tipos heterogéneos sin cast
    println!("--- Error 1: error[E0308] - Mismatched types en operaciones ---");
    println!("❌ Código erróneo:");
    println!("   let a: f32 = 10.5;");
    println!("   let b: usize = 2;");
    println!("   let c = a / b; // 💥 Rust no hace casting implícito");
    println!("✅ Solución: Cast explícito con 'as':");
    let a: f32 = 10.5;
    let b: usize = 2;
    let c = a / (b as f32);
    println!("   Resultado correcto con 'as f32': {}\n", c);

    // Error 2: Mezclar tipos en un array fijo
    println!("--- Error 2: error[E0308] - Arrays requieren tipos homogéneos ---");
    println!("❌ Código erróneo: let datos = [1.0, 2]; // 💥 f32 mezclado con entero");
    println!("✅ Solución: Asegurar que todos tengan el mismo tipo numérico:");
    let datos_corregidos = [1.0, 2.0];
    println!("   Array homogéneo: {:?}\n", datos_corregidos);

    // Error 3: Desbordamiento de enteros (Integer Overflow)
    println!("--- Error 3: Desbordamiento en modo debug (u8: 0..255) ---");
    println!("❌ Un píxel u8 no puede superar 255. Si sumas 200 + 100 en u8 causa pánico en debug.");
    println!("✅ Solución: Convertir a un tipo mayor (u16 o f32) antes de operar:");
    let pixel1: u8 = 200;
    let pixel2: u8 = 100;
    let suma_segura: u16 = (pixel1 as u16) + (pixel2 as u16);
    println!("   Suma segura en u16: {}\n", suma_segura);

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}
