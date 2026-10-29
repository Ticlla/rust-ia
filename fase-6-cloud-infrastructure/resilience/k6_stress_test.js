import http from 'k6/http';
import { check, sleep } from 'k6';

// Configuración de la prueba de carga
export const options = {
  stages: [
    { duration: '3s', target: 10 },  // Rampa de subida: de 0 a 10 usuarios en 3s
    { duration: '10s', target: 30 }, // Carga sostenida: 30 usuarios simultáneos durante 10s
    { duration: '2s', target: 0 },   // Rampa de bajada: enfriamiento a 0 usuarios
  ],
  thresholds: {
    // El 95% de las peticiones debe responder en menos de 20ms
    http_req_duration: ['p(95)<20', 'p(99)<50'],
  },
};

const RAG_URL = 'http://localhost:3000/api/v1/search';
const GATEWAY_HEALTH_URL = 'http://localhost:8081/health';

const QUERIES = [
  'como funciona el kv cache en transformers',
  'optimizacion SIMD AVX2 para embeddings',
  'token bucket rate limiting en axum',
  'continuous batching en vllm rust',
  'algoritmo bm25 para busqueda lexica',
];

export default function () {
  // 1. Prueba de Búsqueda Neuronal Híbrida en el RAG Engine
  const randomQuery = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const ragPayload = JSON.stringify({
    query: randomQuery,
    top_k: 2,
  });

  const ragHeaders = { 'Content-Type': 'application/json' };
  const ragRes = http.post(RAG_URL, ragPayload, { headers: ragHeaders });

  check(ragRes, {
    'RAG status es 200': (r) => r.status === 200,
    'RAG tiene resultados': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.hits && body.hits.length > 0;
      } catch (e) {
        return false;
      }
    },
  });

  // 2. Prueba de Inspección de Salud del LLM Gateway
  const gwRes = http.get(GATEWAY_HEALTH_URL);
  check(gwRes, {
    'Gateway health es 200': (r) => r.status === 200,
    'Gateway está healthy': (r) => {
      try {
        return JSON.parse(r.body).status === 'healthy';
      } catch (e) {
        return false;
      }
    },
  });

  // Pequeña pausa entre iteraciones para simular comportamiento humano/cliente
  sleep(0.05);
}
