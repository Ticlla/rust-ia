//! 🦀 CLASE 03: LABORATORIO DE ERRORES COMUNES - Funciones
//! Ejecutar con: cargo run --bin c03_errores

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 03 [LABORATORIO DE ERRORES]: Funciones y Expresiones");
    println!("========================================================\n");

    println!("--- Error 1: error[E0308] - Punto y coma ';' accidental en el retorno ---");
    println!("❌ Si escribes: fn suma(a: f32, b: f32) -> f32 {{ a + b; }}");
    println!("   El ';' convierte la expresión en sentencia, devolviendo () [unit].");
    println!("✅ Solución: Remueve el ';' final o usa 'return a + b;':");
    let res = suma_correcta(3.0, 5.0);
    println!("   Resultado correcto: {}\n", res);

    println!("--- Error 2: error[E0308] - Ramas if/else con tipos discordantes ---");
    println!("❌ Código erróneo: if condicion {{ 1.0 }} else {{ 0 }}; // f32 vs entero");
    println!("✅ Solución: Ambas ramas de un if/else deben devolver idéntico tipo numérico:");
    let branch_res = if true { 1.0 } else { 0.0 };
    println!("   Resultado con tipos coherentes: {}\n", branch_res);

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}

fn suma_correcta(a: f32, b: f32) -> f32 {
    a + b
}
