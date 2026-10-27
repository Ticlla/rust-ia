//! Telemetría y métricas de inferencia en tiempo real (TTFT, TPS, Duración).

use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct StreamTelemetry {
    pub start_time: Instant,
    pub first_token_time: Option<Instant>,
    pub end_time: Option<Instant>,
    pub token_count: usize,
    pub character_count: usize,
}

impl Default for StreamTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamTelemetry {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            first_token_time: None,
            end_time: None,
            token_count: 0,
            character_count: 0,
        }
    }

    /// Registra la llegada de un nuevo fragmento de texto
    pub fn record_chunk(&mut self, text: &str) {
        if self.first_token_time.is_none() && !text.is_empty() {
            self.first_token_time = Some(Instant::now());
        }
        self.character_count += text.chars().count();
        // Aproximación estándar: 1 token ~ 4 caracteres (o 1 por llamada de chunk en SSE)
        self.token_count += 1;
    }

    /// Finaliza la medición de la generación
    pub fn finish(&mut self) {
        self.end_time = Some(Instant::now());
    }

    /// Tiempo hasta el primer token (Time To First Token) en milisegundos
    pub fn ttft_ms(&self) -> Option<f64> {
        self.first_token_time.map(|t| t.duration_since(self.start_time).as_secs_f64() * 1000.0)
    }

    /// Duración total de la generación en segundos
    pub fn total_duration(&self) -> Duration {
        let end = self.end_time.unwrap_or_else(Instant::now);
        end.duration_since(self.start_time)
    }

    /// Velocidad de generación en Tokens Por Segundo (TPS)
    pub fn tokens_per_second(&self) -> f64 {
        let dur = self.total_duration().as_secs_f64();
        if dur > 0.0 {
            self.token_count as f64 / dur
        } else {
            0.0
        }
    }
}
