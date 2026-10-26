//! ==============================================================================
//! CLASE 03: Errores Comunes y Antipatrones de Serialización en IA
//! ==============================================================================
//!
//! En Machine Learning, guardar y cargar modelos incorrectamente causa dos
//! desastres comunes:
//! 1. Brechas críticas de seguridad (RCE con Pickle de Python).
//! 2. Pérdida de precisión numérica y lentitud extrema (usar JSON para pesos).

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct PesosModelo {
    pesos: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Serialización y Formatos de Modelos");
    println!("============================================================\n");

    error_1_inflacion_y_perdida_precision_en_json()?;
    error_2_el_peligro_mortal_de_pickle_en_ia();
    error_3_deserializacion_con_copia_vs_cero_copia();

    println!("\n✅ Todos los antipatrones analizados y resueltos correctamente.");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 1: Guardar tensores y pesos en JSON
/// ----------------------------------------------------------------------------
/// En JSON, un float de 4 bytes (`f32`) como `0.12345678` se convierte en un
/// string ASCII de 10 caracteres (10 bytes).
/// Además, al convertir de texto a float (`parse::<f32>()`), se introducen
/// errores de redondeo que alteran las predicciones del modelo.
fn error_1_inflacion_y_perdida_precision_en_json() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Error 1: JSON vs Formatos Binarios (Inflación y Precisión) ---");

    let original = PesosModelo {
        pesos: vec![0.12345678, -0.9876543, 3.1415927, 0.0000001],
    };

    // 1. JSON (Texto):
    let json_str = serde_json::to_string(&original)?;
    println!("JSON (Texto, {} bytes):\n  {}", json_str.len(), json_str);

    // 2. Bincode (Binario exacto):
    let binario = bincode::serialize(&original)?;
    println!("Bincode (Binario puro, {} bytes):", binario.len());
    println!("  {:02x?}", &binario);

    // Bincode guarda exactamente 4 bytes por float IEEE-754.
    // Cero pérdida de precisión y 3x a 5x más compacto.
    println!("👉 Regla: NUNCA guardes pesos en JSON. Usa Bincode o Safetensors.\n");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 2: El peligro mortal de Python Pickle (Ejecución Remota de Código)
/// ----------------------------------------------------------------------------
/// Durante años, la comunidad de IA guardó modelos en `.bin` o `.pkl` (Pickle).
/// Pickle permite que un atacante inyecte comandos del sistema operativo (bash,
/// malware) que se ejecutan automáticamente al cargar el modelo con `pickle.load()`.
///
/// Por eso Hugging Face creó `safetensors` en Rust:
/// Es un formato PURAMENTE DE DATOS. No tiene máquina virtual ni intérprete.
/// Es matemáticamente imposible ejecutar código malicioso con safetensors.
fn error_2_el_peligro_mortal_de_pickle_en_ia() {
    println!("--- Error 2: El Ataque de Seguridad de Pickle vs SafeTensors ---");
    println!("⚠️ Pickle en Python: Al hacer `torch.load('modelo.pkl')`, se pueden ejecutar");
    println!("   scripts maliciosos ocultos en el archivo que borran tu servidor.");
    println!("🛡️ SafeTensors en Rust: Solo contiene:");
    println!("   1. Un encabezado JSON con formas y tipos de datos.");
    println!("   2. Los bytes crudos de los tensores.");
    println!("   👉 Es 100% seguro para descargar modelos de Hugging Face.\n");
}

/// ----------------------------------------------------------------------------
/// ERROR 3: Deserializar duplicando memoria en RAM
/// ----------------------------------------------------------------------------
/// Cuando cargas un modelo de 10 GB con SafeTensors, los bytes del archivo
/// tienen exactamente la misma disposición que en la memoria RAM.
/// Si haces copias intermedias, saturas la memoria.
fn error_3_deserializacion_con_copia_vs_cero_copia() {
    println!("--- Error 3: Cero-Copias en la Carga de Modelos ---");
    println!("💡 SafeTensors permite mapear el archivo y leer los punteros a los tensores");
    println!("   sin tener que reservar nuevos vectores `Vec<f32>` en el Heap.");
    println!("   La carga de un modelo de 10 GB pasa de tardar 2 minutos a 100 milisegundos.");
}
