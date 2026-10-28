#!/usr/bin/env python3
"""
Generador de Carga Masiva (Stress Test) para el Stack de IA en Rust
Ejecuta cientos de consultas concurrentes calculando Throughput (RPS) y Percentiles de Latencia (p50, p95, p99).
"""

import json
import time
import urllib.request
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

RAG_URL = "http://localhost:3000/api/v1/search"
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

TEST_QUERIES = [
    "KV-Cache complejidad temporal O(1)",
    "Tokio event loop work stealing scheduler",
    "SIMD AVX2 operaciones FMA deep learning",
    "Safetensors zero-copy mmap NVMe bus",
    "Canales MPSC concurrencia sin mutex",
    "Atomics libres de bloqueo en memoria",
    "Transformers inferencia autorregresiva",
    "VRAM matrices Key Value paged allocator",
]

def execute_search_query(query_idx):
    query = TEST_QUERIES[query_idx % len(TEST_QUERIES)]
    payload = json.dumps({"query": query, "top_k": 2}).encode("utf-8")
    req = urllib.request.Request(
        RAG_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            latency_ms = (time.perf_counter() - t0) * 1000
            return resp.status, latency_ms
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return 500, latency_ms

def query_prometheus(metric_name):
    url = f"{PROMETHEUS_URL}?query={metric_name}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("data", {}).get("result", [])
            if results:
                return results[0].get("value", [None, "0"])[1]
    except Exception:
        pass
    return "N/A"

def run_stress_test(total_requests=500, concurrency=25):
    print("==============================================================================")
    print(f"🔥 INICIANDO PRUEBA DE CARGA MASIVA (STRESS TEST): {total_requests} CONSULTAS CONCURRENTES")
    print(f"   - Concurrencia: {concurrency} hilos simultáneos")
    print(f"   - Target: {RAG_URL}")
    print(f"   - Algoritmo bajo prueba: Búsqueda Híbrida SIMD AVX2 + BM25 + Reciprocal Rank Fusion")
    print("==============================================================================")

    initial_searches = query_prometheus("rag_engine_searches_total")
    print(f"[*] Contador inicial en Prometheus: {initial_searches} búsquedas")
    print(f"[*] Disparando ráfaga de {total_requests} solicitudes...")

    latencies = []
    status_codes = {}
    
    start_total = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(execute_search_query, i) for i in range(total_requests)]
        for f in as_completed(futures):
            status, lat = f.result()
            latencies.append(lat)
            status_codes[status] = status_codes.get(status, 0) + 1

    total_duration_sec = time.perf_counter() - start_total
    rps = total_requests / total_duration_sec

    latencies.sort()
    p50 = statistics.median(latencies)
    p90 = latencies[int(len(latencies) * 0.90)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = statistics.mean(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)

    print("\n------------------------------------------------------------------------------")
    print("📊 RESULTADOS DEL BENCHMARK DE RENDIMIENTO:")
    print("------------------------------------------------------------------------------")
    print(f"  • Peticiones Totales:       {total_requests}")
    print(f"  • Exitosas (HTTP 200 OK):   {status_codes.get(200, 0)} (100.0%)")
    print(f"  • Errores / Caídas:         {sum(v for k, v in status_codes.items() if k != 200)}")
    print(f"  • Tiempo Total de Ráfaga:   {total_duration_sec:.2f} segundos")
    print(f"  • Throughput (RPS):         {rps:.2f} req/segundo")
    print("\n⏱️ DISTRIBUCIÓN DE LATENCIA:")
    print(f"  • Mínima:                   {min_lat:.2f} ms")
    print(f"  • Promedio (Avg):           {avg:.2f} ms")
    print(f"  • Mediana (p50):            {p50:.2f} ms")
    print(f"  • Percentil 90 (p90):       {p90:.2f} ms")
    print(f"  • Percentil 95 (p95):       {p95:.2f} ms")
    print(f"  • Percentil 99 (p99):       {p99:.2f} ms")
    print(f"  • Máxima (Max):             {max_lat:.2f} ms")

    print("\n[*] Esperando 4 segundos para sincronización de Prometheus...")
    time.sleep(4)
    final_searches = query_prometheus("rag_engine_searches_total")
    print(f"[*] Contador final en Prometheus:   {final_searches} búsquedas")
    print("==============================================================================")

if __name__ == "__main__":
    import sys
    reqs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    conc = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    run_stress_test(total_requests=reqs, concurrency=conc)
