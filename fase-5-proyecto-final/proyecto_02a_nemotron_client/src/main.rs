use proyecto_02a_nemotron_client::{NemotronClient, NemotronConfig, NemotronRepl};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = NemotronConfig::default();

    // Permitir pasar API Key por argumento de línea de comandos: `cargo run -- <API_KEY>`
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && !args[1].starts_with('-') {
        let key = args[1].clone();
        config.api_key = Some(key.clone());
        config.backend_mode = proyecto_02a_nemotron_client::BackendMode::NvidiaNim;
    }
    if args.len() > 2 && !args[2].starts_with('-') {
        config.model = args[2].clone();
    }

    let client = NemotronClient::new(config)?;
    let mut repl = NemotronRepl::new(client);

    repl.run().await?;
    Ok(())
}
