//! Módulo de Resiliencia y Control de Tráfico para el Serving Gateway
//! Implementa el algoritmo Token Bucket para Rate-Limiting por cliente/IP.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// Representa una cubeta de tokens (Token Bucket) individual para un cliente.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    capacity: f64,
    refill_rate_per_sec: f64,
    tokens: f64,
    last_refill: Instant,
}

impl TokenBucket {
    /// Crea una nueva cubeta con capacidad máxima y tasa de recarga por segundo.
    pub fn new(capacity: f64, refill_rate_per_sec: f64) -> Self {
        Self {
            capacity,
            refill_rate_per_sec,
            tokens: capacity,
            last_refill: Instant::now(),
        }
    }

    /// Intenta adquirir `cost` tokens. Devuelve `true` si se autorizó, o `false` si se excedió el límite.
    pub fn try_acquire(&mut self, cost: f64) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        
        // Recargar tokens en función del tiempo transcurrido, acotado a la capacidad máxima
        self.tokens = (self.tokens + elapsed * self.refill_rate_per_sec).min(self.capacity);
        self.last_refill = now;

        if self.tokens >= cost {
            self.tokens -= cost;
            true
        } else {
            false
        }
    }

    /// Retorna la cantidad de tokens disponibles actualmente.
    pub fn available_tokens(&self) -> f64 {
        self.tokens
    }

    /// Retorna los segundos estimados antes de que haya al menos 1 token disponible.
    pub fn seconds_until_available(&self) -> f64 {
        if self.tokens >= 1.0 {
            0.0
        } else {
            (1.0 - self.tokens) / self.refill_rate_per_sec
        }
    }
}

/// Gestor de Rate-Limiting concurrente en memoria para múltiples clientes.
pub struct TokenBucketRateLimiter {
    capacity: f64,
    refill_rate_per_sec: f64,
    buckets: Mutex<HashMap<String, TokenBucket>>,
}

impl TokenBucketRateLimiter {
    /// Inicializa el limitador con capacidad de ráfaga (burst) y tasa de recarga continua.
    pub fn new(capacity: f64, refill_rate_per_sec: f64) -> Self {
        Self {
            capacity,
            refill_rate_per_sec,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Verifica si la petición de un cliente es admitida o rechazada por cuota excedida.
    /// Devuelve `Ok(())` si es admitida, o `Err(retry_after_secs)` si es rechazada.
    pub fn check_admission(&self, client_id: &str) -> Result<(), f64> {
        let mut buckets = self.buckets.lock().unwrap();
        let bucket = buckets
            .entry(client_id.to_string())
            .or_insert_with(|| TokenBucket::new(self.capacity, self.refill_rate_per_sec));

        if bucket.try_acquire(1.0) {
            Ok(())
        } else {
            Err(bucket.seconds_until_available().max(1.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_token_bucket_burst_and_depletion() {
        // Capacidad 3 tokens, recarga 1 token por segundo
        let mut bucket = TokenBucket::new(3.0, 1.0);

        // 3 peticiones consecutivas deben ser aceptadas (burst)
        assert!(bucket.try_acquire(1.0));
        assert!(bucket.try_acquire(1.0));
        assert!(bucket.try_acquire(1.0));

        // La 4ta petición inmediata debe ser rechazada (cubeta vacía)
        assert!(!bucket.try_acquire(1.0));

        // Esperar 1.05 segundos para que se recargue al menos 1 token
        sleep(Duration::from_millis(1050));
        assert!(bucket.try_acquire(1.0));
        assert!(!bucket.try_acquire(1.0));
    }

    #[test]
    fn test_concurrent_rate_limiter_multi_client() {
        let limiter = TokenBucketRateLimiter::new(2.0, 5.0);

        // Cliente A agota su cuota
        assert!(limiter.check_admission("client_a").is_ok());
        assert!(limiter.check_admission("client_a").is_ok());
        assert!(limiter.check_admission("client_a").is_err());

        // Cliente B tiene su propia cuota independiente y debe ser admitido
        assert!(limiter.check_admission("client_b").is_ok());
        assert!(limiter.check_admission("client_b").is_ok());
        assert!(limiter.check_admission("client_b").is_err());
    }
}
