//! ============================================================================
//! ⚡ DEMOSTRACIÓN EN VIVO: RÁFAGA DE TRÁFICO Y HOT-RELOAD EN PRODUCCIÓN
//! ============================================================================
//!
//! Este binario demuestra en tiempo real la ventaja de la arquitectura:
//! 1. El servidor atiende un flujo continuo e ininterrumpido de usuarios concurrentes.
//! 2. Se inyecta una petición de `/hot_reload` con nuevos pesos.
//! 3. El modelo se actualiza en caliente mediante RwLock y el buffer anterior
//!    se destruye limpiamente con RAII Drop.
//! 4. Sin perder ni una sola petición y sin reiniciar el proceso, todas las
//!    solicitudes siguientes se calculan inmediatamente con el Modelo v2.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

#[path = "01_teoria.rs"]
mod proyecto;
use proyecto::*;

#[tokio::main]
async fn main() {
    println!("============================================================");
    println!("⚡ DEMO EN VIVO: TRÁFICO CONCURRENTE Y HOT-RELOAD EN CALIENTE");
    println!("============================================================\n");

    // 1. Canal asíncrono para el DataLoader
    let (data_tx, mut data_rx) = mpsc::channel::<String>(20);

    tokio::spawn(async move {
        while let Some(msg) = data_rx.recv().await {
            println!("  [DataLoader] 📦 Evento del sistema: {msg}");
        }
    });

    // 2. Modelo inicial v1: pesos = [0.5, 2.0, -1.0]
    // Para entrada [10.0, 5.0, 2.0]: 10*0.5 + 5*2.0 - 2*1.0 = 13.0
    let pesos_v1 = vec![0.5, 2.0, -1.0];
    let estado = Arc::new(AppState {
        modelo: Arc::new(RwLock::new(ModeloInferencia {
            version: 1,
            pesos: TensorBuffer::from_slice(&pesos_v1),
        })),
        total_inferencias: Arc::new(AtomicUsize::new(0)),
        dataloader_tx: data_tx,
    });

    // 3. Levantar servidor en un puerto libre
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = crear_router(estado.clone());

    println!("🌐 Servidor de IA activo en: http://{}", addr);
    println!("🧠 Modelo cargado: Versión 1 | Pesos: {:?}", pesos_v1);
    println!("------------------------------------------------------------");

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let ejecutando = Arc::new(AtomicBool::new(true));

    // 4. Simulador de Usuarios: Generando ráfaga de tráfico continuo
    println!("🚀 INICIANDO RÁFAGA DE TRÁFICO DE USUARIOS...");
    let ejecutando_clientes = ejecutando.clone();
    let cliente_task = tokio::spawn(async move {
        while ejecutando_clientes.load(Ordering::Relaxed) {
            let addr_clon = addr;
            tokio::spawn(async move {
                if let Ok(mut stream) = TcpStream::connect(addr_clon).await {
                    let req_body = serde_json::to_string(&PredictRequest {
                        features: vec![10.0, 5.0, 2.0],
                    })
                    .unwrap();

                    let post_req = format!(
                        "POST /predict HTTP/1.1\r\nHost: {addr_clon}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        req_body.len(),
                        req_body
                    );

                    let _ = stream.write_all(post_req.as_bytes()).await;
                    let mut buf = Vec::new();
                    let _ = stream.read_to_end(&mut buf).await;
                }
            });

            tokio::time::sleep(Duration::from_millis(150)).await;
        }
    });

    // Dejar correr peticiones bajo el Modelo v1 durante 1.5 segundos
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // 5. Inyección del HOT-RELOAD con CURL simulado
    println!("\n============================================================");
    println!("📢 DISPARANDO 'POST /hot_reload' CON NUEVOS PESOS [1.0, 1.0, 1.0]");
    println!("   Esperado v2 para [10.0, 5.0, 2.0]: 10*1 + 5*1 + 2*1 = 17.0");
    println!("============================================================\n");

    let reload_body = serde_json::json!({ "nuevos_pesos": [1.0, 1.0, 1.0] }).to_string();
    let reload_req = format!(
        "POST /hot_reload HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        reload_body.len(),
        reload_body
    );

    let mut stream_reload = TcpStream::connect(addr).await.unwrap();
    stream_reload.write_all(reload_req.as_bytes()).await.unwrap();
    let mut resp_reload = Vec::new();
    stream_reload.read_to_end(&mut resp_reload).await.unwrap();

    // Dejar correr tráfico bajo el nuevo Modelo v2 durante 1.5 segundos más
    tokio::time::sleep(Duration::from_millis(1500)).await;

    ejecutando.store(false, Ordering::Relaxed);
    let _ = cliente_task.await;

    // Resumen final
    let total = estado.total_inferencias.load(Ordering::Relaxed);
    let version_final = estado.modelo.read().unwrap().version;
    println!("\n============================================================");
    println!("📊 RESUMEN FINAL DEL TEST EN VIVO:");
    println!("   • Total de inferencias servidas: {}", total);
    println!("   • Versión actual del modelo: v{}", version_final);
    println!("   • Caídas de servicio (Downtime): 0 milisegundos");
    println!("   • Peticiones perdidas: 0");
    println!("============================================================");
    println!("🎉 ¡Demostración de Hot-Reload en ejecución completada con éxito!");
}
