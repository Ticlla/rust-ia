//! ============================================================================
//! 🦀 CLASE 01: LABORATORIO DE ERRORES COMUNES - Variables y Mutabilidad
//! ============================================================================
//!
//! Este archivo explica los errores más frecuentes que cometen los programadores
//! novatos en Rust al trabajar con variables, y cómo interpretarlos y resolverlos.
//!
//! Ejecutar con:
//!     cargo run --bin c01_errores

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 01 [LABORATORIO DE ERRORES]: Variables y Mutabilidad");
    println!("========================================================\n");

    // -------------------------------------------------------------------------
    // ERROR 1: Reasignar a una variable inmutable
    // Código de error de Rust: [E0384]
    // -------------------------------------------------------------------------
    println!("--- Error 1: error[E0384] - cannot assign twice to immutable variable ---");
    println!("❌ Código que causa el error:");
    println!("   let learning_rate = 0.01;");
    println!("   learning_rate = 0.005; // 💥 ¡El compilador rechaza esto!");
    println!("\n💡 ¿Por qué ocurre?:");
    println!("   En Rust, 'let' hace que la variable sea de solo lectura por defecto.");
    println!("✅ ¿Cómo se soluciona?:");
    println!("   Agrega 'mut' en la declaración inicial: let mut learning_rate = 0.01;\n");

    // Demostración de la solución correcta:
    let mut learning_rate_corregido = 0.01;
    println!("   Valor inicial : {}", learning_rate_corregido);
    learning_rate_corregido = 0.005;
    println!("   Valor reasignado: {}\n", learning_rate_corregido);

    // -------------------------------------------------------------------------
    // ERROR 2: Olvidar la anotación de tipo en una constante
    // Código de error de Rust: [E0282]
    // -------------------------------------------------------------------------
    println!("--- Error 2: error[E0282] - type annotations needed for `const` ---");
    println!("❌ Código que causa el error:");
    println!("   const BATCH_SIZE = 64; // 💥 ¡Error: falta el tipo!");
    println!("\n💡 ¿Por qué ocurre?:");
    println!("   A diferencia de 'let', las constantes 'const' NUNCA infieren el tipo.");
    println!("   El compilador exige que declares explícitamente el tipo exacto.");
    println!("✅ ¿Cómo se soluciona?:");
    println!("   Indica el tipo explícito: const BATCH_SIZE: usize = 64;\n");

    // Demostración de la solución correcta:
    const BATCH_SIZE_CORREGIDO: usize = 64;
    println!("   Demostración en vivo: BATCH_SIZE_CORREGIDO = {}\n", BATCH_SIZE_CORREGIDO);

    // -------------------------------------------------------------------------
    // ERROR 3: Cambiar el tipo de una variable mutable en una reasignación
    // Código de error de Rust: [E0308] - mismatched types
    // -------------------------------------------------------------------------
    println!("--- Error 3: error[E0308] - mismatched types (mut no cambia de tipo) ---");
    println!("❌ Código que causa el error:");
    println!("   let mut input_dim = \"512\";");
    println!("   input_dim = 512; // 💥 ¡Error! 'input_dim' fue tipada como &str, no puede recibir un entero.");
    println!("\n💡 ¿Por qué ocurre?:");
    println!("   'mut' permite cambiar el VALOR, pero jamás el TIPO de dato.");
    println!("✅ ¿Cómo se soluciona?:");
    println!("   Usa Shadowing (un nuevo 'let') en lugar de reasignación con '=':");
    println!("   let input_dim = \"512\";");
    println!("   let input_dim: usize = input_dim.parse().unwrap();\n");

    // Demostración de la solución correcta con Shadowing:
    let input_dim = "512";
    let input_dim: usize = input_dim.parse().unwrap();
    println!("   Demostración en vivo con Shadowing: input_dim transformado a entero = {}\n", input_dim);

    // -------------------------------------------------------------------------
    // ERROR 4: Usar una variable potencialmente no inicializada
    // Código de error de Rust: [E0381] - used binding isn't initialized
    // -------------------------------------------------------------------------
    println!("--- Error 4: error[E0381] - used binding isn't initialized ---");
    println!("❌ Código que causa el error:");
    println!("   let pesos_iniciales: f32;");
    println!("   println!(\"{{pesos_iniciales}}\"); // 💥 ¡Error: la variable no tiene valor!");
    println!("\n💡 ¿Por qué ocurre?:");
    println!("   En C/C++, esto leería basura que ya estaba en la memoria.");
    println!("   En Rust, está estrictamente prohibido leer memoria sin inicializar.");
    println!("✅ ¿Cómo se soluciona?:");
    println!("   Asigna siempre un valor inicial: let pesos_iniciales: f32 = 0.0;\n");

    let pesos_iniciales = 0.0;
    println!("   Demostración en vivo: pesos_iniciales = {}\n", pesos_iniciales);

    println!("========================================================");
    println!("🎉 ¡Has completado el laboratorio de errores de la Clase 01!");
    println!("👉 Pasa a resolver los retos con: cargo run --bin c01_ejercicios");
    println!("========================================================");
}
