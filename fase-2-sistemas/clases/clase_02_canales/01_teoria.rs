//! ============================================================================
//! 🦀 FASE 2 - CLASE 02: Canales de Comunicación (`mpsc`) y DataLoaders en IA
//! ============================================================================
//!
//! En esta clase aprenderemos:
//! 1. Qué es un canal `mpsc` (Multiple Producer, Single Consumer).
//! 2. `tx.send()`: cómo el Ownership viaja por el canal hacia otro hilo.
//! 3. `rx.recv()` vs iterar sobre `rx` con un bucle `for`.
//! 4. Múltiples productores clonando el transmisor (`tx.clone()`).
//! 5. Caso real en IA: DataLoader paralelo (2 hilos cargan datos mientras 1 entrena).
//!
//! Para ejecutar este archivo:
//!     cargo run --bin c02_teoria

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    println!("========================================================");
    println!("📦 FASE 2 - CLASE 02: Canales mpsc y DataLoaders en IA");
    println!("========================================================\n");

    // =========================================================================
    // 1. CANAL BÁSICO ENTRE DOS HILOS
    // =========================================================================
    println!("--- 1. Enviar un tensor por el canal ---");

    // Creamos el canal:
    // tx = Transmitter (Transmisor / Emisor)
    // rx = Receiver (Receptor)
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let tensor = vec![0.15, -0.42, 0.88, 0.03];
        println!("  [Hilo Productor] Enviando tensor de pesos a la cinta...");
        
        // 'send()' toma posesión (Move) de 'tensor' y lo deposita en el canal:
        tx.send(tensor).unwrap();
        // Si intentáramos usar 'tensor' aquí, Rust daría error de Ownership.
    });

    // En el hilo principal (Consumidor), '.recv()' bloquea hasta recibir el dato:
    let tensor_recibido = rx.recv().unwrap();
    println!("  [Hilo Principal] Tensor recibido exitosamente: {:?}\n", tensor_recibido);

    // =========================================================================
    // 2. MÚLTIPLES PRODUCTORES: EL PATRÓN DATALOADER
    // =========================================================================
    // En PyTorch, un DataLoader suele usar 'num_workers = 2' o más.
    // Varios hilos leen archivos del disco y todos envían a la misma GPU/entrenador.
    println!("--- 2. DataLoader Paralelo con Múltiples Productores (MPSC) ---");

    let (tx_lotes, rx_lotes) = mpsc::channel();

    // Trabajador 1 (Worker #1): Lee del disco "disco_ssd_1.csv"
    let tx1 = tx_lotes.clone(); // 💡 ¡Clonamos el transmisor para el Worker 1!
    thread::spawn(move || {
        let archivos_worker1 = ["lote_1.csv", "lote_2.csv"];
        for archivo in archivos_worker1 {
            thread::sleep(Duration::from_millis(50)); // Simula lectura de disco
            let batch = vec![1.0, 2.0, 3.0]; // Datos cargados
            println!("  [Worker 1] Cargado {archivo}");
            tx1.send((1, archivo, batch)).unwrap();
        }
    });

    // Trabajador 2 (Worker #2): Lee del disco "disco_ssd_2.csv"
    let tx2 = tx_lotes.clone(); // 💡 ¡Clonamos el transmisor para el Worker 2!
    thread::spawn(move || {
        let archivos_worker2 = ["lote_3.csv", "lote_4.csv"];
        for archivo in archivos_worker2 {
            thread::sleep(Duration::from_millis(70)); // Simula lectura de disco
            let batch = vec![4.0, 5.0, 6.0]; // Datos cargados
            println!("  [Worker 2] Cargado {archivo}");
            tx2.send((2, archivo, batch)).unwrap();
        }
    });

    // ⚠️ DETALLE CRÍTICO DE RUST:
    // Debemos destruir el 'tx_lotes' original de main.
    // ¿Por qué? Porque el canal solo se cierra cuando TODOS los transmisores mueren.
    // Si no hacemos 'drop(tx_lotes)', el bucle de abajo esperaría para siempre.
    drop(tx_lotes);

    // =========================================================================
    // 3. EL CONSUMIDOR: ENTRENANDO A MEDIDA QUE LLEGAN LOS LOTES
    // =========================================================================
    println!("\n[Hilo de Entrenamiento] Esperando batches para entrenar...");

    // Al usar 'for lote in rx_lotes', el bucle recibe datos uno por uno
    // y se cierra automáticamente cuando todos los Workers terminan:
    for (worker_id, archivo, datos) in rx_lotes {
        println!("  🏋️ [GPU] Entrenando con {archivo} (del Worker #{worker_id}) -> {:?}", datos);
    }

    println!("\n✅ Todos los batches fueron consumidos. Fin de la Clase 02.");
}
