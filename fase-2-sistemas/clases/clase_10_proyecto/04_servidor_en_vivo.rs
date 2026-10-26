//! ============================================================================
//! 🚀 PROYECTO FINAL EN VIVO: Servidor de Inferencia y DataLoader en Producción
//! ============================================================================
//!
//! Este binario arranca el servidor completo de producción de la Fase 2 en:
//!     http://127.0.0.1:3000
//!
//! Para ejecutarlo:
//!     cargo run --bin c10_servidor

use axum::response::Html;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

// Reutilizamos la lógica del proyecto:
#[path = "01_teoria.rs"]
mod proyecto;
use proyecto::*;

async fn root_dashboard() -> Html<&'static str> {
    Html(r#"
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <title>Motor de Inferencia Rust - Fase 2</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 850px; margin: 40px auto; line-height: 1.6; color: #333; padding: 0 20px; }
            h1 { color: #d35400; border-bottom: 2px solid #e67e22; padding-bottom: 10px; }
            .badge { background: #27ae60; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.9em; }
            code { background: #f4f6f7; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.95em; }
            pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 6px; overflow-x: auto; }
            .card { background: #fdfefe; border: 1px solid #d5dbdb; border-radius: 8px; padding: 15px 20px; margin: 15px 0; }
        </style>
    </head>
    <body>
        <h1>🦀 Motor de Inferencia en Rust <span class="badge">ACTIVO EN VIVO</span></h1>
        <p>Servidor de alto rendimiento orquestado con <strong>Tokio + Axum + Rayon + RwLock + RAII</strong>.</p>
        
        <div class="card">
            <h3>📊 Diagnóstico del Sistema</h3>
            <p>Consulta las métricas en tiempo real: <a href="/health"><code>GET /health</code></a></p>
        </div>

        <div class="card">
            <h3>⚡ Enviar Petición de Inferencia (POST /predict)</h3>
            <pre>curl -X POST http://127.0.0.1:3000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [10.0, 5.0, 2.0]}'</pre>
        </div>

        <div class="card">
            <h3>🔄 Recargar Pesos del Modelo en Caliente (POST /hot_reload)</h3>
            <pre>curl -X POST http://127.0.0.1:3000/hot_reload \
     -H "Content-Type: application/json" \
     -d '{"nuevos_pesos": [1.0, 1.0, 1.0]}'</pre>
        </div>
        <p><em>Presiona Ctrl+C en la terminal para apagar el servidor.</em></p>
    </body>
    </html>
    "#)
}

#[tokio::main]
async fn main() {
    println!("============================================================");
    println!("🏆 SERVIDOR EN VIVO - PROYECTO FINAL INTEGRADOR (FASE 2)");
    println!("============================================================\n");

    // 1. Canal asíncrono para el DataLoader:
    let (data_tx, mut data_rx) = mpsc::channel::<String>(20);

    // 2. DataLoader continuo en segundo plano:
    tokio::spawn(async move {
        let mut lote_id = 0;
        loop {
            tokio::select! {
                Some(msg) = data_rx.recv() => {
                    println!("  [DataLoader] 🔔 Notificación de Control: {msg}");
                }
                _ = tokio::time::sleep(Duration::from_secs(10)) => {
                    lote_id += 1;
                    println!("  [DataLoader] 📥 Lote sintético #{lote_id} ingerido y pre-procesado en búfer.");
                }
            }
        }
    });

    // 3. Inicializamos los pesos con TensorBuffer de bajo nivel (Clase 09):
    let pesos_iniciales = vec![0.5, 2.0, -1.0];
    let estado = Arc::new(AppState {
        modelo: Arc::new(RwLock::new(ModeloInferencia {
            version: 1,
            pesos: TensorBuffer::from_slice(&pesos_iniciales),
        })),
        total_inferencias: Arc::new(AtomicUsize::new(0)),
        dataloader_tx: data_tx,
    });

    // 4. Router con el dashboard y los endpoints REST:
    let app = crear_router(estado.clone())
        .route("/", axum::routing::get(root_dashboard));

    let host = "127.0.0.1:3000";
    let listener = TcpListener::bind(host).await.unwrap();

    println!("✅ Servidor escuchando en: http://{}", host);
    println!("💻 Núcleos de CPU disponibles para Rayon: {}", rayon::current_num_threads());
    println!("\n💡 Prueba los comandos desde otra terminal o abre tu navegador en:");
    println!("   👉 http://127.0.0.1:3000");
    println!("\nPresiona [Ctrl + C] para detener el servidor.\n");

    axum::serve(listener, app).await.unwrap();
}
