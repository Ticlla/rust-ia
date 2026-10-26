//! ============================================================================
//! 🎯 FASE 2 - CLASE 07: Retos Prácticos con Tokio en Servidores de IA
//! ============================================================================
//!
//! Ejecutar el reto:
//!     cargo run --bin c07_ejercicios
//!
//! Ejecutar las pruebas unitarias:
//!     cargo test --bin c07_ejercicios

use tokio::sync::{mpsc, oneshot};

/// Estructura de petición de inferencia con canal de respuesta único (oneshot).
pub struct PeticionInferencia {
    pub id: usize,
    pub entrada: Vec<f32>,
    pub respuesta_tx: oneshot::Sender<f32>,
}

/// Función que ejecuta la predicción tensorial en CPU (cálculo pesado).
pub fn inferencia_cpu(entrada: &[f32]) -> f32 {
    // Simula producto punto y activación ReLU:
    let suma: f32 = entrada.iter().map(|&x| x * 2.0).sum();
    suma.max(0.0)
}

/// Orquesta el servidor de colas de inferencia:
/// Recibe peticiones asíncronas de red y delega el cómputo CPU a 'spawn_blocking'.
pub async fn servidor_cola_inferencia(mut cola_rx: mpsc::Receiver<PeticionInferencia>) -> usize {
    let mut total_procesadas = 0;

    while let Some(peticion) = cola_rx.recv().await {
        total_procesadas += 1;

        // 💡 Separamos el cómputo CPU usando spawn_blocking:
        tokio::task::spawn_blocking(move || {
            let resultado = inferencia_cpu(&peticion.entrada);
            let _ = peticion.respuesta_tx.send(resultado);
        });
    }

    total_procesadas
}

#[tokio::main]
async fn main() {
    println!("============================================================");
    println!("🎯 RETO PRÁCTICO: Servidor de Inferencia Asíncrono con Colas");
    println!("============================================================\n");

    // Canal con capacidad para 10 peticiones concurrentes:
    let (cola_tx, cola_rx) = mpsc::channel(10);

    // Lanzamos el servidor de inferencia en segundo plano:
    let servidor_handle = tokio::spawn(servidor_cola_inferencia(cola_rx));

    // Simulamos 3 clientes concurrentes enviando vectores:
    let mut clientes_handles = Vec::new();

    let vectores = vec![
        vec![1.0, 2.0, 3.0],   // suma*2 = 12.0
        vec![-2.0, -1.0, 0.0],  // suma*2 = -6.0 -> ReLU = 0.0
        vec![0.5, 0.5, 1.0],   // suma*2 = 4.0
    ];

    for (id, entrada) in vectores.into_iter().enumerate() {
        let tx = cola_tx.clone();
        let h = tokio::spawn(async move {
            let (resp_tx, resp_rx) = oneshot::channel();
            let peticion = PeticionInferencia {
                id,
                entrada,
                respuesta_tx: resp_tx,
            };

            // Enviamos la petición por la cola asíncrona:
            tx.send(peticion).await.unwrap();

            // Esperamos la respuesta individual del servidor:
            let prediccion = resp_rx.await.unwrap();
            println!("  [Cliente #{id}] 🎯 Predicción recibida: {:.2}", prediccion);
            prediccion
        });
        clientes_handles.push(h);
    }

    // Cerramos el canal del emisor principal para que el servidor sepa cuándo parar:
    drop(cola_tx);

    for h in clientes_handles {
        h.await.unwrap();
    }

    let procesadas = servidor_handle.await.unwrap();
    println!("\nTotal de peticiones procesadas por el servidor: {procesadas}");
    println!("\n✅ Reto ejecutado exitosamente.");
}

// =============================================================================
// PRUEBAS UNITARIAS AUTOMÁTICAS CON TOKIO
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cola_inferencia_procesa_correctamente() {
        let (tx, rx) = mpsc::channel(5);
        let servidor_handle = tokio::spawn(servidor_cola_inferencia(rx));

        let (resp_tx, resp_rx) = oneshot::channel();
        let peticion = PeticionInferencia {
            id: 1,
            entrada: vec![2.0, 3.0], // (2 + 3) * 2 = 10.0
            respuesta_tx: resp_tx,
        };

        tx.send(peticion).await.unwrap();
        let res = resp_rx.await.unwrap();
        assert_eq!(res, 10.0);

        drop(tx);
        let count = servidor_handle.await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_inferencia_cpu_relu() {
        assert_eq!(inferencia_cpu(&[-5.0, -2.0]), 0.0);
        assert_eq!(inferencia_cpu(&[1.0, 2.0]), 6.0);
    }
}
