#!/usr/bin/env python3
"""
Simulador de Usuarios Concurrentes para la Malla de IA en Rust
Demuestra tráfico multi-usuario, recuperación RAG y métricas en Prometheus/Grafana.
"""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

GATEWAY_URL = "http://localhost:8081/v1/chat/completions"
RAG_URL = "http://localhost:3000/api/v1/search"
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

USERS_LLM_QUERIES = [
    ("Usuario 1 (Ana - AI Researcher)", "¿Que beneficio temporal tiene KV-Cache segun los documentos?"),
    ("Usuario 2 (Carlos - Systems Eng)", "¿Como garantiza Rust seguridad en concurrencia y sin data races?"),
    ("Usuario 3 (Beatriz - HPC Lead)", "¿Que ventajas de rendimiento ofrece SIMD AVX2 y FMA en deep learning?"),
    ("Usuario 4 (David - MLOps Lead)", "¿Por que el formato Safetensors previene ejecucion remota de codigo?"),
    ("Usuario 5 (Elena - Cloud Arch)", "¿Como funciona el robo de trabajo en el runtime asincrono de Tokio?"),
]

RAG_SEARCH_QUERIES = [
    "KV-Cache complejidad temporal",
    "Tokio work stealing event loop",
    "SIMD AVX2 FMA producto punto",
    "Safetensors zero-copy mmap",
    "Canales MPSC concurrencia",
    "Inferencia autorregresiva transformers",
    "Memoria VRAM matrices Key Value",
    "Pickle vulnerabilidad seguridad",
]

def send_chat_completion(user_name, question):
    start = time.time()
    payload = {
        "model": "nvidia/nemotron-3.5-bolt-e4b",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 80,
        "temperature": 0.1,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GATEWAY_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed_ms = (time.time() - start) * 1000
            res = json.loads(resp.read().decode("utf-8"))
            content = res["choices"][0]["message"]["content"].replace("\n", " ")[:90]
            return f"[{user_name}] ({elapsed_ms:.1f}ms) -> HTTP 200: \"{content}...\""
    except Exception as e:
        return f"[{user_name}] ERROR: {e}"

def send_rag_search(query_id, query):
    start = time.time()
    payload = {"query": query, "top_k": 2}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        RAG_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            elapsed_ms = (time.time() - start) * 1000
            res = json.loads(resp.read().decode("utf-8"))
            hits = len(res.get("hits", []))
            return f"[Search #{query_id:02d}: \"{query[:22]}\"] ({elapsed_ms:.2f}ms) -> {hits} hits"
    except Exception as e:
        return f"[Search #{query_id:02d}] ERROR: {e}"

def query_prometheus_metric(metric_name):
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

def main():
    print("==============================================================================")
    print("🚀 INICIANDO SIMULACIÓN DE TRÁFICO MULTI-USUARIO CONCURRENTE")
    print("==============================================================================")
    
    print("\n[FASE 1] Ráfaga de Búsquedas Concurrentes en el RAG Engine (20 peticiones):")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(20):
            q = RAG_SEARCH_QUERIES[i % len(RAG_SEARCH_QUERIES)]
            futures.append(executor.submit(send_rag_search, i + 1, q))
        for f in as_completed(futures):
            print("  ", f.result())
    rag_total_time = (time.time() - t0) * 1000
    print(f"  -> 20 búsquedas RAG híbridas procesadas en {rag_total_time:.1f}ms total (promedio {rag_total_time/20:.1f}ms/req)")

    print("\n[FASE 2] 5 Usuarios Concurrentes realizando Consultas RAG + LLM Gateway:")
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(send_chat_completion, user, q)
            for user, q in USERS_LLM_QUERIES
        ]
        for f in as_completed(futures):
            print("  ", f.result())
    llm_total_time = time.time() - t1
    print(f"  -> 5 inferencias concurrentes completadas en {llm_total_time:.2f}s")

    print("\n[FASE 3] Esperando 4 segundos a que Prometheus termine el ciclo de scraping...")
    time.sleep(4)

    print("\n[FASE 4] Métricas Actualizadas en Tiempo Real desde Prometheus:")
    print("  - Total Peticiones LLM Gateway (llm_gateway_requests_total):", query_prometheus_metric("llm_gateway_requests_total"))
    print("  - Recuperaciones RAG Automáticas (llm_gateway_rag_retrievals_total):", query_prometheus_metric("llm_gateway_rag_retrievals_total"))
    print("  - Total Búsquedas en RAG Engine (rag_engine_searches_total):", query_prometheus_metric("rag_engine_searches_total"))
    print("  - Chunks Indexados en Memoria (rag_engine_chunks_total):", query_prometheus_metric("rag_engine_chunks_total"))
    print("  - Uptime del Servicio RAG en Segundos (rag_engine_uptime_seconds):", query_prometheus_metric("rag_engine_uptime_seconds"))
    print("==============================================================================")
    print("💡 Abre tu navegador en http://localhost:3001 para ver los picos en Grafana.")
    print("==============================================================================")

if __name__ == "__main__":
    main()
