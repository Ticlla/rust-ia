#!/usr/bin/env python3
"""
Simulador de Ráfaga y Verificación de Rate-Limiting (Token Bucket)
1. Dispara una ráfaga de 15 solicitudes concurrentes al Gateway.
2. Demuestra la transición de HTTP 200 OK a HTTP 429 Too Many Requests.
3. Envía los logs del incidente a Elasticsearch para inspección en Kibana.
4. Consulta el contador en Prometheus para visualización en Grafana.
"""

import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

GATEWAY_URL = "http://localhost:8081/v1/chat/completions"
ES_BULK_URL = "http://localhost:9200/ai-stack-logs/_bulk"
PROMETHEUS_URL = "http://localhost:9090/api/v1/query?query=llm_gateway_rate_limited_total"

def send_request(req_id):
    payload = json.dumps({
        "model": "nvidia/nemotron-3.5-bolt-e4b",
        "messages": [{"role": "user", "content": f"Test burst query #{req_id}"}],
        "max_tokens": 8,
        "temperature": 0.1,
    }).encode("utf-8")

    req = urllib.request.Request(
        GATEWAY_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            latency_ms = (time.perf_counter() - t0) * 1000
            return req_id, resp.status, latency_ms, "OK"
    except urllib.error.HTTPError as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            body = json.loads(e.read().decode("utf-8"))
            msg = body.get("error", {}).get("message", e.reason)
        except Exception:
            msg = e.reason
        return req_id, e.code, latency_ms, msg
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return req_id, 500, latency_ms, str(e)

def ship_events_to_elasticsearch(results):
    lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    
    for req_id, status, lat, detail in results:
        level = "WARN" if status == 429 else ("ERROR" if status >= 500 else "INFO")
        action = "Petición bloqueada por Token Bucket Rate Limiter" if status == 429 else "Inferencia permitida"
        doc = {
            "@timestamp": now,
            "service": "ai-llm-gateway",
            "level": level,
            "event": "rate_limiter_evaluation",
            "action": action,
            "status_code": status,
            "request_id": req_id,
            "latency_ms": round(lat, 2),
            "client_ip": "192.168.1.150",
            "user": "Usuario-Prueba-Burst",
            "detail": detail,
            "env": "production"
        }
        lines.append(json.dumps({"index": {"_index": "ai-stack-logs"}}))
        lines.append(json.dumps(doc))

    payload = "\n".join(lines) + "\n"
    req = urllib.request.Request(
        ES_BULK_URL,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            # Forzar refresh inmediato del índice en Elasticsearch
            refresh_req = urllib.request.Request("http://localhost:9200/ai-stack-logs/_refresh", data=b"")
            urllib.request.urlopen(refresh_req, timeout=5)
            return True
    except Exception as e:
        print(f"[!] Error enviando logs a ES: {e}")
        return False

def query_prometheus():
    try:
        with urllib.request.urlopen(PROMETHEUS_URL, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("data", {}).get("result", [])
            if results:
                return results[0].get("value", [None, "0"])[1]
    except Exception:
        pass
    return "0"

def main():
    print("==============================================================================")
    print("⚡ SIMULACIÓN DE RÁFAGA CONCURRENTE CONTRA EL RATE LIMITER (TOKEN BUCKET)")
    print("   Capacidad de la cubeta: 10 peticiones | Tasa de recarga: 2.0 req/segundo")
    print("   Ráfaga enviada: 15 peticiones instantáneas")
    print("==============================================================================")

    init_prom = query_prometheus()
    print(f"[*] Contador de bloqueos en Prometheus antes de la ráfaga: {init_prom}")
    print("[*] Disparando 15 peticiones concurrentes...")

    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(send_request, i + 1) for i in range(15)]
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x[0])

    print("\n------------------------------------------------------------------------------")
    print("📋 RESPUESTA DETALLADA DE CADA PETICIÓN:")
    print("------------------------------------------------------------------------------")
    status_counts = {}
    for req_id, status, lat, detail in results:
        status_counts[status] = status_counts.get(status, 0) + 1
        icon = "✅" if status == 200 else ("⛔" if status == 429 else "❌")
        print(f"  {icon} Petición #{req_id:02d} | Código: HTTP {status} | Latencia: {lat:6.2f}ms | Detalle: {detail}")

    print("\n------------------------------------------------------------------------------")
    print("📊 RESUMEN DEL COMPORTAMIENTO DE RESILIENCIA:")
    print(f"  • Peticiones Autorizadas (HTTP 200 OK):       {status_counts.get(200, 0)}")
    print(f"  • Peticiones Bloqueadas (HTTP 429 Too Many):  {status_counts.get(429, 0)}")
    print("------------------------------------------------------------------------------")

    print("[*] Enviando los eventos del incidente a Elasticsearch...")
    if ship_events_to_elasticsearch(results):
        print("✅ Eventos indexados en Elasticsearch y disponibles en Kibana.")

    print("[*] Esperando 4 segundos para actualización del scrape de Prometheus...")
    time.sleep(4)
    final_prom = query_prometheus()
    print(f"[*] Contador de bloqueos en Prometheus después de la ráfaga: {final_prom} bloqueos")
    print("==============================================================================")
    print("🔍 CÓMO VISUALIZARLO AHORA:")
    print("  1. En KIBANA (http://localhost:5601/app/discover):")
    print("     Escribe en la barra de búsqueda KQL:")
    print('     status_code : 429   o   level : "WARN"')
    print("     (Asegúrate de tener el filtro de tiempo en 'Last 15 minutes')")
    print("  2. En GRAFANA (http://localhost:3001):")
    print("     Verás el panel '⛔ Bloqueos Rate Limit (429)' iluminado en ROJO.")
    print("==============================================================================")

if __name__ == "__main__":
    main()
