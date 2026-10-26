//! ==============================================================================
//! CLASE 04: Errores Comunes y Antipatrones con Memoria Mapeada (`mmap`)
//! ==============================================================================
//!
//! Trabajar con `mmap` interactúa directamente con la Memoria Virtual del kernel
//! de Linux. Los errores aquí no son simples bugs de software: pueden provocar
//! Out-Of-Memory (OOM) o señales fatales del sistema operativo (`SIGBUS`).

use memmap2::Mmap;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Antipatrones de Memoria Mapeada (mmap)");
    println!("============================================================\n");

    error_1_oom_por_lectura_completa_en_heap()?;
    error_2_por_que_mmap_es_unsafe_y_sigbus()?;
    error_3_ciclos_de_vida_y_dangling_pointers()?;

    println!("\n✅ Todos los antipatrones de mmap analizados correctamente.");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 1: Leer modelos gigantes con `std::fs::read`
/// ----------------------------------------------------------------------------
/// `std::fs::read` carga TODOS los bytes a un `Vec<u8>` en la memoria física.
/// Si un servidor tiene 16 GB de RAM e intentas cargar un modelo de 15 GB,
/// el sistema operativo colapsa por OOM Killer.
fn error_1_oom_por_lectura_completa_en_heap() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Error 1: Copiar a RAM (std::fs::read) vs Mapear (mmap) ---");
    println!("❌ Antipatrón: `let bytes = std::fs::read(\"llama_15GB.safetensors\")?;`");
    println!("   Duplica 15 GB de golpe en el Heap. Si entran 2 peticiones concurrentes,");
    println!("   la máquina explota.");
    println!("👉 Regla: Siempre usa `Mmap::map(&file)?` para cualquier archivo mayor a 500 MB.\n");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 2: Por qué `Mmap::map` es `unsafe` (El riesgo de SIGBUS en Linux)
/// ----------------------------------------------------------------------------
/// En Rust, llamar a `Mmap::map(&file)` requiere un bloque `unsafe { }`.
/// ¿Por qué? Porque otro proceso en tu sistema Linux podría borrar o truncar
/// el archivo en disco mientras tu modelo está leyendo sus pesos.
/// Si eso pasa, el procesador dispara la señal `SIGBUS` y el programa se apaga.
fn error_2_por_que_mmap_es_unsafe_y_sigbus() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Error 2: Por qué Mmap requiere 'unsafe' (SIGBUS) ---");
    println!("⚠️ Riesgo del Sistema Operativo:");
    println!("   Si el archivo en disco es modificado o recortado por otro proceso");
    println!("   mientras tu red neuronal lee sus pesos, la CPU genera un fallo de hardware (SIGBUS).");
    println!("🛡️ Solución en Producción: Abrir siempre los archivos con permisos");
    println!("   estrictos de solo lectura (`File::open`) y garantizar que nadie modifique los pesos.\n");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 3: Ciclo de vida: No soltar el `Mmap` mientras haya tensores usándolo
/// ----------------------------------------------------------------------------
/// Los tensores que lees de un SafeTensors mapeado NO son copias; son punteros
/// que apuntan directamente al interior del `Mmap`.
/// Si el `Mmap` se destruye, esos punteros quedarían colgando en el aire (Dangling Pointers).
fn error_3_ciclos_de_vida_y_dangling_pointers() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Error 3: Gestión de Ciclos de Vida (Borrow Checker) ---");

    let ruta = std::env::temp_dir().join("test_dummy_mmap.bin");
    {
        let mut f = File::create(&ruta)?;
        f.write_all(&[1, 2, 3, 4, 5, 6, 7, 8])?;
    }

    let file = File::open(&ruta)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // El slice `&[u8]` tiene el tiempo de vida ligado a `mmap`:
    let slice_prestado: &[u8] = &mmap[0..4];
    println!("Bytes prestados de forma segura con tiempo de vida: {:?}", slice_prestado);

    // Si intentáramos destruir `drop(mmap)` aquí, el compilador de Rust
    // PROHÍBE compilar el código porque `slice_prestado` sigue vivo.
    // ¡Garantía de memoria intacta!

    let _ = std::fs::remove_file(&ruta);
    println!("👉 Regla: El struct del modelo debe ser dueño del `Mmap` para mantener vivos los tensores.");
    Ok(())
}
