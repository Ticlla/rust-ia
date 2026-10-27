//! Consola interactiva (REPL) con streaming continuo y telemetría de tokens.

use std::io::{self, Write};
use crate::client::NemotronClient;
use crate::config::BackendMode;
use crate::metrics::StreamTelemetry;
use crate::types::ChatMessage;

pub struct NemotronRepl {
    client: NemotronClient,
    history: Vec<ChatMessage>,
}

impl NemotronRepl {
    pub fn new(client: NemotronClient) -> Self {
        let system_prompt = client.config.system_prompt.clone();
        let history = vec![ChatMessage::system(system_prompt)];
        Self { client, history }
    }

    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.print_welcome();

        loop {
            print!("\n🦀 Usuario > ");
            io::stdout().flush()?;

            let mut input = String::new();
            if io::stdin().read_line(&mut input)? == 0 {
                // EOF (Ctrl+D)
                println!("\n¡Hasta pronto!");
                break;
            }

            let prompt = input.trim();
            if prompt.is_empty() {
                continue;
            }

            // Comandos especiales de la consola
            if prompt.starts_with('/') {
                if self.handle_command(prompt) {
                    break;
                }
                continue;
            }

            // Agregar mensaje del usuario al historial
            self.history.push(ChatMessage::user(prompt));

            // Imprimir respuesta en streaming
            print!("🤖 Nemotron > ");
            io::stdout().flush()?;

            let mut telemetry = StreamTelemetry::new();
            let mut full_response = String::new();

            match self.client.stream_chat(&self.history).await {
                Ok(mut rx) => {
                    while let Some(item) = rx.recv().await {
                        match item {
                            Ok(token) => {
                                print!("{token}");
                                io::stdout().flush()?;
                                telemetry.record_chunk(&token);
                                full_response.push_str(&token);
                            }
                            Err(e) => {
                                eprintln!("\n❌ Error en stream: {e}");
                                break;
                            }
                        }
                    }
                    telemetry.finish();

                    // Imprimir barra de telemetría de inferencia
                    println!();
                    self.print_telemetry_badge(&telemetry);

                    // Guardar respuesta del asistente en el historial
                    self.history.push(ChatMessage::assistant(full_response));
                }
                Err(e) => {
                    eprintln!("\n❌ Error conectando con Nemotron: {e}");
                }
            }
        }

        Ok(())
    }

    fn print_welcome(&self) {
        println!("================================================================================");
        println!("  🧠 NVIDIA NEMOTRON INTERACTIVE CLIENT (RUST AI ENGINE)");
        println!("================================================================================");
        println!("  • Modelo activo:  {}", self.client.config.model);
        println!("  • Backend:        {:?}", self.client.config.backend_mode);
        println!("  • Endpoint:       {}", self.client.config.base_url);
        println!("  • Comandos útiles: /help, /clear, /model <nombre>, /system <prompt>, /exit");
        if self.client.config.backend_mode == BackendMode::MockSimulator {
            println!("  ℹ️  MODO SIMULADOR ACTIVO: Para usar NVIDIA NIM Cloud real, define la variable:");
            println!("     export NVIDIA_API_KEY=\"nvapi-...\"");
        }
        println!("================================================================================");
    }

    fn print_telemetry_badge(&self, telemetry: &StreamTelemetry) {
        let ttft = telemetry
            .ttft_ms()
            .map(|t| format!("{:.1} ms", t))
            .unwrap_or_else(|| "N/A".to_string());
        let tps = telemetry.tokens_per_second();
        let total_time = telemetry.total_duration().as_secs_f64();

        println!(
            "  └─ ⏱️ [TTFT: {} | ⚡ {:.1} tok/s | Total: {} tokens en {:.2}s]",
            ttft, tps, telemetry.token_count, total_time
        );
    }

    fn handle_command(&mut self, cmd: &str) -> bool {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        match parts[0] {
            "/exit" | "/quit" | "/q" => {
                println!("¡Hasta luego!");
                return true;
            }
            "/clear" => {
                let system_prompt = self.client.config.system_prompt.clone();
                self.history = vec![ChatMessage::system(system_prompt)];
                println!("🧹 Historial de conversación reiniciado.");
            }
            "/model" => {
                if parts.len() > 1 {
                    self.client.config.model = parts[1].to_string();
                    println!("✓ Modelo actualizado a: {}", self.client.config.model);
                } else {
                    println!("ℹ️ Modelo actual: {}", self.client.config.model);
                }
            }
            "/system" => {
                if parts.len() > 1 {
                    let new_prompt = parts[1..].join(" ");
                    self.client.config.system_prompt = new_prompt.clone();
                    if let Some(first) = self.history.first_mut() {
                        first.content = new_prompt;
                    }
                    println!("✓ Personalidad / System Prompt actualizado.");
                } else {
                    println!("ℹ️ System Prompt actual:\n{}", self.client.config.system_prompt);
                }
            }
            "/config" => {
                println!("📋 Configuración actual:");
                println!("   • Modelo:       {}", self.client.config.model);
                println!("   • Modo Backend: {:?}", self.client.config.backend_mode);
                println!("   • URL Base:     {}", self.client.config.base_url);
                println!("   • Temperatura:  {}", self.client.config.temperature);
                println!("   • Top-P:        {}", self.client.config.top_p);
                println!("   • Historial:    {} mensajes", self.history.len());
            }
            "/help" => {
                println!("📖 Comandos disponibles:");
                println!("   /clear            - Borra la memoria y reinicia el contexto");
                println!("   /model <nombre>   - Cambia el modelo (ej: nvidia/nemotron-4-340b-instruct)");
                println!("   /system <texto>   - Define las instrucciones de sistema del modelo");
                println!("   /config           - Muestra los parámetros de conexión actuales");
                println!("   /exit             - Cierra la consola interactiva");
            }
            _ => {
                println!("❌ Comando desconocido '{}'. Escribe /help para ver la lista.", parts[0]);
            }
        }
        false
    }
}
