//! ============================================================================
//! 🧪 CASO 3: Rc (Monohilo) vs Arc (Multihilo Atómico) y los Traits Send/Sync
//! ============================================================================
//!
//! Ejecutar este caso:
//!     cargo run --bin c03_caso3
//!
//! CONTEXTO EN IA:
//! Un programador habituado a Rust en un solo hilo usa el patrón `Rc<RefCell<T>>`
//! para tener múltiples dueños y mutar datos. Pero cuando intenta enviar eso a un
//! hilo de fondo (por ejemplo, para guardar un checkpoint del modelo en disco),
//! el compilador lo bloquea.

use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    println!("============================================================");
    println!("🧪 CASO 3: 'trait Send is not implemented' [E0277] vs Arc<Mutex>");
    println!("============================================================\n");

    // =========================================================================
    // PARTE A: ¿CUÁL ES EL ERROR?
    // =========================================================================
    println!("--- ❌ PARTE A: Intentar usar Rc<RefCell<T>> entre hilos ---");
    println!("Si intentas escribir:");
    println!("    use std::rc::Rc;");
    println!("    use std::cell::RefCell;");
    println!("    let pesos = Rc::new(RefCell::new(vec![1.0, 2.0]));");
    println!("    let clon = Rc::clone(&pesos);");
    println!("    thread::spawn(move || {{");
    println!("        clon.borrow_mut()[0] += 10.0;");
    println!("    }}); // ❌ ERROR!\n");

    println!("¿Qué dice el compilador de Rust?");
    println!("    error[E0277]: `Rc<RefCell<Vec<f64>>>` cannot be sent between threads safely");
    println!("    the trait `Send` is not implemented for `Rc<RefCell<Vec<f64>>>`\n");

    println!("¿Por qué falla a nivel de Hardware?");
    println!("* 'Rc' incrementa su contador con un simple 'contador += 1' de CPU.");
    println!("* Si dos núcleos ejecutan 'contador += 1' al mismo milisegundo:");
    println!("  - Ambos leen contador = 1.");
    println!("  - Ambos escriben contador = 2 (cuando debería ser 3!).");
    println!("  - El contador queda desincronizado, y cuando un hilo termina, destruye");
    println!("    la memoria antes de tiempo, causando un desastroso 'Use-After-Free'.");
    println!("* Rust evita esto negándole a 'Rc' el trait 'Send'. ¡El código ni compila!\n");

    // =========================================================================
    // PARTE B: LA SOLUCIÓN CORRECTA: Arc<Mutex<T>>
    // =========================================================================
    println!("--- ✅ PARTE B: La Solución con Arc<Mutex<T>> (Instrucciones Atómicas) ---");

    // Reemplazamos:
    // Rc      -> Arc   (Atomic Reference Counting: usa 'LOCK XADD' a nivel CPU)
    // RefCell -> Mutex (Cerrojo con exclusión mutua para multihilo)
    let checkpoint_pesos = Arc::new(Mutex::new(vec![0.42, 0.88, -0.15]));

    let pesos_worker = Arc::clone(&checkpoint_pesos);

    let h = thread::spawn(move || {
        // Hilo en segundo plano: guarda o ajusta el checkpoint
        let mut datos = pesos_worker.lock().unwrap();
        println!("  [Hilo Worker] Accediendo a los pesos de forma segura...");
        datos[0] = 0.99; // Modificación sincronizada
        println!("  [Hilo Worker] Nuevo peso establecido: {:.2}", datos[0]);
    });

    h.join().unwrap();

    let datos_finales = checkpoint_pesos.lock().unwrap();
    println!("\n[Hilo Principal] Pesos leídos tras el guardado: {:?}", *datos_finales);
    println!("💡 Como 'Arc' y 'Mutex' implementan 'Send' y 'Sync', Rust garantiza 0 colisiones.");
    println!("\n✅ Caso 3 completado con éxito.");
}

// =============================================================================
// CÓDIGO CON ERROR PARA DESCOMENTAR Y COMPROBAR:
// =============================================================================
// Si quieres ver al compilador fallar en vivo, descomenta este bloque:
//
// use std::rc::Rc;
// use std::cell::RefCell;
// fn codigo_invalido() {
//     let pesos = Rc::new(RefCell::new(vec![1.0, 2.0]));
//     let clon = Rc::clone(&pesos);
//     thread::spawn(move || {
//         clon.borrow_mut()[0] += 10.0; // ❌ error[E0277]: trait Send not implemented
//     });
// }
