#!/usr/bin/env python3
"""
Generador de Logs de Producción para Elasticsearch y Kibana
Inyecta eventos estructurados del RAG Engine y LLM Gateway con timestamps actuales.
"""

import json
import random
import time
import urllib.request
from datetime import datetime, timezone, timedelta

ES_BULK_URL = "http://localhost:9200/ai-stack-logs/_bulk"

SERVICES = ["ai-rag-engine", "ai-llm-gateway"]
USERS = ["Carlos", "Ana", "Beatriz", "David", "Elena", "Fernando", "Lucia"]
MODELS = ["nvidia/nemotron-3.5-bolt-e4b", "smollm2-135m"]

SAMPLE_EVENTS = [
    {
        "service": "ai-rag-engine",
        "level": "INFO",
        "event": "hybrid_search",
        "action": "Búsqueda híbrida SIMD AVX2 + BM25 ejecutada",
        "query": "KV-Cache complejidad temporal O(1)",
        "hits_count": 2,
        "duration_ms": 3.42,
    },
    {
        "service": "ai-rag-engine",
        "level": "INFO",
        "event": "document_ingest",
        "action": "Ingesta y chunking de documento empresarial",
        "doc_title": "Manual de Operaciones Cóndor 2026",
        "chunks_generated": 4,
        "duration_ms": 1.15,
    },
    {
        "service": "ai-rag-engine",
        "level": "DEBUG",
        "event": "simd_vector_dot_product",
        "action": "Cálculo vectorial de similitud coseno con instrucciones FMA",
        "dim": 64,
        "duration_ms": 0.08,
    },
    {
        "service": "ai-rag-engine",
        "level": "INFO",
        "event": "bm25_score_fusion",
        "action": "Reciprocal Rank Fusion (RRF) combinando listas densas y dispersas",
        "rrf_k": 60,
        "candidates_evaluated": 16,
    },
    {
        "service": "ai-llm-gateway",
        "level": "INFO",
        "event": "chat_completion_success",
        "action": "Inferencia completada con éxito y contexto RAG inyectado",
        "tokens_prompt": 184,
        "tokens_generated": 68,
        "latency_ms": 1340.5,
    },
    {
        "service": "ai-llm-gateway",
        "level": "DEBUG",
        "event": "paged_kv_cache_alloc",
        "action": "Asignación de bloques en gestor de memoria Paged KV-Cache",
        "blocks_allocated": 4,
        "free_blocks_remaining": 124,
    },
    {
        "service": "ai-llm-gateway",
        "level": "DEBUG",
        "event": "paged_kv_cache_free",
        "action": "Liberación de bloques KV-Cache tras finalizar secuencia",
        "blocks_freed": 4,
        "free_blocks_remaining": 128,
    },
    {
        "service": "ai-llm-gateway",
        "level": "WARN",
        "event": "token_bucket_warning",
        "action": "Ráfaga de solicitudes detectada en gateway; rate limit al 80%",
        "client_ip": "192.168.1.105",
        "tokens_left": 12,
    },
    {
        "service": "ai-llm-gateway",
        "level": "ERROR",
        "event": "upstream_timeout",
        "action": "Fallo transitorio conectando con upstream; reintentando con backoff",
        "retry_attempt": 1,
        "max_retries": 3,
    },
]

def generate_bulk_payload(num_records=150):
    lines = []
    now = datetime.now(timezone.utc)
    
    for i in range(num_records):
        # Timestamps distribuidos en los últimos 30 minutos
        minutes_ago = random.uniform(0.1, 30.0)
        event_time = now - timedelta(minutes=minutes_ago)
        timestamp_str = event_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        template = random.choice(SAMPLE_EVENTS).copy()
        template["@timestamp"] = timestamp_str
        template["user"] = random.choice(USERS)
        template["host"] = "rust-node-01"
        template["env"] = "production"
        
        if template["service"] == "ai-llm-gateway":
            template["model"] = random.choice(MODELS)
            template["status_code"] = 504 if template["level"] == "ERROR" else (429 if template["level"] == "WARN" else 200)

        meta = {"index": {"_index": "ai-stack-logs"}}
        lines.append(json.dumps(meta))
        lines.append(json.dumps(template))

    return "\n".join(lines) + "\n"

def main():
    print("==============================================================================")
    print("🚀 GENERANDO E INYECTANDO 150 LOGS DE PRODUCCIÓN EN ELASTICSEARCH")
    print("==============================================================================")
    
    payload = generate_bulk_payload(150)
    req = urllib.request.Request(
        ES_BULK_URL,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"}
    )
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = (time.time() - t0) * 1000
            items = len(data.get("items", []))
            errors = data.get("errors", False)
            print(f"[*] Registros indexados exitosamente: {items}")
            print(f"[*] Errores: {errors}")
            print(f"[*] Tiempo de indexación en Elasticsearch: {elapsed:.2f} ms")
    except Exception as e:
        print(f"[!] Error enviando bulk a Elasticsearch: {e}")
        return

    # Verificar conteo total actual
    try:
        count_url = "http://localhost:9200/ai-stack-logs/_count"
        with urllib.request.urlopen(count_url) as resp:
            total_count = json.loads(resp.read().decode("utf-8")).get("count", 0)
            print(f"\n✅ Total actual de logs en Elasticsearch: {total_count} eventos")
    except Exception:
        pass

    print("==============================================================================")
    print("💡 Actualiza la pestaña de Discover en Kibana (http://localhost:5601/app/discover)")
    print("   Asegúrate de que el filtro de tiempo arriba a la derecha esté en 'Last 1 hour' o 'Today'.")
    print("==============================================================================")

if __name__ == "__main__":
    main()
