//! ============================================================================
//! 🛠️ FASE 2 - CLASE 09: Laboratorio de Errores Comunes con `unsafe` y Punteros
//! ============================================================================
//!
//! Este laboratorio explora los 3 peligros mortales de memoria al trabajar
//! con punteros crudos y librerías nativas en C/CUDA.
//!
//! Para ejecutar este laboratorio:
//!     cargo run --bin c09_errores

fn main() {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: unsafe y Punteros Crudos");
    println!("============================================================\n");

    // =========================================================================
    // ERROR COMÚN 1: Desreferenciar un Puntero Nulo (Segmentation Fault)
    // =========================================================================
    println!("--- Error 1: Desreferenciar un Puntero Nulo (Null Pointer) ---");
    println!("En Safe Rust, una referencia (&T) NUNCA puede ser nula (no existe null).");
    println!("Pero los punteros crudos (*const T) de C/CUDA SÍ pueden ser nulos:");
    println!("    let ptr_nulo: *const f32 = std::ptr::null();");
    println!("    unsafe {{ println!(\"{{}}\", *ptr_nulo); }} // 💀 SEGMENTATION FAULT (CRASH)");
    println!("\n👉 REGLA DE ORO: Verifica siempre con '.is_null()' antes de desreferenciar:\n");
    println!("    if !ptr_nulo.is_null() {{");
    println!("        unsafe {{ println!(\"{{}}\", *ptr_nulo); }}");
    println!("    }}\n");

    // =========================================================================
    // ERROR COMÚN 2: Violación de Aliasing (Crear dos &mut al mismo tiempo)
    // =========================================================================
    println!("--- Error 2: Violación de Aliasing Mutable con Punteros ---");
    println!("El optimizador de Rust (LLVM) asume que una referencia mutable (&mut T)");
    println!("es el ÚNICO puntero que tiene acceso a esa memoria.");
    println!("Si mediante punteros crudos creas dos '&mut' apuntando a la misma dirección:");
    println!("    let mut x = 10.0;");
    println!("    let p = &mut x as *mut f64;");
    println!("    unsafe {{");
    println!("        let r1 = &mut *p;");
    println!("        let r2 = &mut *p; // ⚠️ UNDEFINED BEHAVIOR (Comportamiento Indefinido)");
    println!("    }}");
    println!("👉 LECCIÓN: 'unsafe' no te autoriza a romper las reglas fundamentales de aliasing.");
    println!("   El compilador podría generar optimizaciones de CPU que corrompen el cálculo.\n");

    // =========================================================================
    // ERROR COMÚN 3: Memory Leak por olvidar dealloc en Drop
    // =========================================================================
    println!("--- Error 3: Fugas de Memoria por olvidar 'Drop' en Buffers ---");
    println!("Si pides memoria manual con 'std::alloc::alloc' y no implementas 'Drop':");
    println!("Rust NO sabe cuántos bytes pediste ni cómo liberarlos.");
    println!("Cada vez que crees un tensor, esa memoria RAM se quedará atrapada para siempre");
    println!("hasta que tu servidor se quede sin RAM (Out of Memory).");
    println!("👉 REGLA: Si usas 'alloc', implementa SIEMPRE el trait 'Drop' con 'dealloc'.\n");

    println!("✅ Todos los casos de error de bajo nivel comprendidos.");
}
