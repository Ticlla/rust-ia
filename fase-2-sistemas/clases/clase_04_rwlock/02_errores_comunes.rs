//! ============================================================================
//! 🛠️ FASE 2 - CLASE 04: Laboratorio de Errores Comunes con `RwLock`
//! ============================================================================
//!
//! Este laboratorio explora los 3 errores más comunes y peligrosos al utilizar
//! `RwLock` en servidores de IA y cómo prevenirlos.
//!
//! Para ejecutar este laboratorio:
//!     cargo run --bin c04_errores

fn main() {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: RwLock en Servidores de IA");
    println!("============================================================\n");

    // =========================================================================
    // ERROR COMÚN 1: Deadlock por intentar Upgrading (Read -> Write) en el mismo hilo
    // =========================================================================
    println!("--- Error 1: Deadlock por solicitar Write teniendo Read activo ---");
    println!("Si en el mismo hilo escribes:");
    println!("    let lock = RwLock::new(vec![1.0, 2.0]);");
    println!("    let lector = lock.read().unwrap();");
    println!("    let mut escritor = lock.write().unwrap(); // 💀 DEADLOCK!");
    println!("\n¿Por qué se congela el programa?");
    println!("Porque 'write()' exige que NO haya ningún lector activo.");
    println!("El escritor se queda esperando a que 'lector' se destruya...");
    println!("¡pero 'lector' nunca se destruirá porque el hilo se congeló en la línea del write!");
    println!("👉 REGLA DE ORO: 'RwLock' de Rust NO permite escalada (upgrading) automática.");
    println!("👉 SOLUCIÓN: Suelta el lector antes de pedir el escritor: 'drop(lector);'\n");

    // =========================================================================
    // ERROR COMÚN 2: Mantener un ReadGuard abierto durante Operaciones Lentas
    // =========================================================================
    println!("--- Error 2: Bloqueo de Escritores por Lecturas Prolongadas ---");
    println!("Si escribes:");
    println!("    let modelo = lock.read().unwrap();");
    println!("    descargar_dataset_de_internet(); // ❌ Operación de 30 segundos");
    println!("    modelo.predecir();");
    println!("\n¿Por qué es un error grave en producción?");
    println!("Porque mientras 'modelo' esté en el scope, NINGÚN hilo podrá actualizar");
    println!("los pesos del modelo. Tu proceso de re-entrenamiento se congelará durante 30s.");
    println!("👉 SOLUCIÓN: Limita el scope de 'lock.read()' únicamente al cálculo rápido:");
    println!("    let resultado = {{");
    println!("        let m = lock.read().unwrap();");
    println!("        m.predecir(&entrada)");
    println!("    }}; // 💡 El candado se libera aquí de inmediato!\n");

    // =========================================================================
    // ERROR COMÚN 3: Elegir RwLock cuando Mutex es mejor
    // =========================================================================
    println!("--- Error 3: Usar RwLock cuando hay muchas escrituras ---");
    println!("'RwLock' tiene un costo de CPU mayor que 'Mutex' para gestionar lectores concurrentes.");
    println!("* Si tu carga es 95% Lecturas (Inferencia) y 5% Escrituras (Checkpoints): RwLock es el REY.");
    println!("* Si tu carga es 50% Escrituras (Entrenamiento con gradientes constantes): Mutex es MUCHO más rápido.");
    println!("👉 REGLA: Si escribes frecuentemente, usa 'Mutex'. Si lees masivamente y escribes rara vez, usa 'RwLock'.\n");

    println!("✅ Todos los casos de error comprendidos.");
}
