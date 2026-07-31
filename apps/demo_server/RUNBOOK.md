# Hermes Live Avatar — Operator Runbook

This runbook documents the **live `/api/health` surface** as of `c985d3f` (the
resilience trio commit) plus the consolidated `util.audit` event stream as of
this commit. When something in the demo is broken, the first thing you should
do is `curl /api/health` — every degraded/unreachable subsystem reports there
alongside the operator data you need to fix it.

> The `/api/health` endpoint is **never-raising**: it returns 200 even when
> one or more probes have failed. Each top-level `components` entry has
> `status` (`"ok" | "degraded" | "error"`) and `detail` with the actual data.
> The endpoint-level `status` is the worst of the per-component statuses.

---

## 0 — First move when something looks wrong

```bash
# 1. Probe the health endpoint. Identify the COMPNENT(S) that are not "ok".
curl -s http://localhost:8001/api/health | jq

# 2. Tail the audit-log stream. Every breaker transition emits one canonical
#    structured event via logger.info / logger.warning with:
#      extra={"audit": {"event": "<name>.<kind>", "name": "<name>",
#                       "kind": "<kind>", ...fields}}
#    Look for events with kind="trip" (circuit breaker tripped OPEN) or
#    kind="cost_cap_exceeded" (per-minute subprocess budget exhausted).
#    --trace-id correlation: each request gets an id stamped on the
#    X-Trace-Id response header; grep the logs for trace_id=<id> to see
#    every tick + log line that happened during that request.
curl -s http://localhost:8001/api/health | jq '.components.audit_log.detail.by_breaker'
```

The 7 components you can probe today:

| Key in `/api/health.components`   | Adapter                                       | Underlying primitive       |
| --------------------------- | --------------------------------------------- | -------------------------- |
| `config`                    | `DemoOrchestrator.config`                     | schema + hardware profile  |
| `renderer`                  | `LiveTalkingAdapter.capabilities()`           | HTTP probe + breaker + retry |
| `voice_backend`             | `LuxTTSAdapter.capability_status()`           | subprocess + breaker + cost-cap |
| `protocol_agent`            | `AgentBridge.capability_status()` + adapter  | HTTP + breaker + retry counters |
| `character_catalog`         | `DemoOrchestrator.character_catalog`          | filesystem walk            |
| `faceswap`                  | renderer `health()` if available              | FaceFusion/Deep-Live-Cam   |
| `livetalking_reachability`  | renderer `base_url` reachability              | HTTP probe                 |
| `audit_log` (NEW this commit) | `util.audit` counter cache across all 3 breakers + each subsystem | structured event log      |

---

## 1 — Symptom → component → cause → fix tables

### 1.1 — `voice_backend.status == "degraded"`

**Most likely cause:** the LuxTTS vendor subprocess breaker has tripped or the
per-minute subprocess budget has been exhausted.

> Read `/api/health.components.voice_backend.detail.circuit_breaker.state`:
> - `"closed"` → breaker healthy; the degradation is somewhere else.
> - `"open"` → breaker has tripped. Two consecutive failures in the last 60s.
> - `"half-open"` → open window elapsed, awaiting a single recovery probe.
>
> Read `/api/health.components.voice_backend.detail.cost_cap.used_seconds`:
> - if it exceeds `cost_cap.remaining_seconds` and `calls_blocked > 0`, the
>   cost-cap (this commit, default 240s/min) is throttling new calls.

**Fix paths:**

| Symptom                                                                           | Likely cause                                     | Action                                                                          |
| --------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------- |
| breaker.state=="open" + circuit_breaker.last_failure_time recent                   | upstream LuxTTS CLI is broken or hung            | Restart vendor / check `LUXTTS_COMMAND` template; wait 60s for breaker to half-open |
| cost_cap.used_seconds near cap_seconds + calls_blocked > 0                         | thundering herd of /api/speak in fallback mode   | Raise `LUXTTS_COST_CAP_SECONDS_PER_MINUTE` (env) — or wait 60s for window to roll |
| last_error contains "timed out"                                                    | vendored CLI genuinely slow                       | Bump `LUXTTS_*` timeout (current 120s) — or replace CLI                          |
| last_error contains "completed without producing a WAV"                            | vendored CLI succeeded but output missing        | Re-run the vendor command manually; check filesystem permissions                 |

Parametric fallback audio is the intended graceful-degradation: users still hear
something during a vendor outage. Do NOT disable the breaker to "fix" this.

### 1.2 — `renderer.status == "degraded"`

**Most likely cause:** LiveTalking HTTP backend is unreachable, or the breaker
has tripped.

> Read `/api/health.components.renderer.detail.online` (boolean) and
> `circuit_breaker.state`. Note: `renderer.circuit_breaker` uses `name="renderer"`
> under the audit counter (`by_breaker.renderer.last_event_kind=='trip'`).

**Fix paths:**

| Symptom                                                                           | Likely cause                                     | Action                                                                          |
| --------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------- |
| online==false                                                                      | LiveTalking container offline                    | `docker ps -a \` grep livetalking`; check LIVETALKING_URL env var on the demo server |
| breaker.state=="open" + last_latency_ms > request_timeout_seconds (1.5)            | LiveTalking container overloaded                 | Check the LiveTalking server logs; consider scaling worker count                  |
| endpoint_status["last_error"] contains "Connection refused"                        | Wrong port or container stopped                  | Verify `LIVETALKING_URL` (default `http://127.0.0.1:8001/video`)                |
| breaker cycling open→half-open→open every 60s                                       | LiveTalking genuinely broken                     | Inspect the upstream service; grace-degrade is the avatar passthrough -- that is expected |

### 1.3 — `protocol_agent.status == "degraded"`

**Most likely cause:** openai-compatible endpoint breaker tripped, OR retry
exhausted on a transient failure, OR no API key configured.

> Read `protocol_agent.detail.circuit_breaker.state` (similar shape to
> renderer). Read `protocol_agent.detail.retry.last_retry_count` and
> `total_retries`. Read `url_configured` (false means no
> `OPENAI_COMPATIBLE_API_KEY`).

**Fix paths:**

| Symptom                                                                           | Likely cause                                     | Action                                                                          |
| --------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------- |
| url_configured==false                                                              | No API key in env                                | Put `OPENAI_COMPATIBLE_API_KEY` in the Freebuff API Keys panel                   |
| breaker.state=="open"                                                              | 3+ consecutive failures in last 30s              | Check the upstream LLM provider status page; breaker auto-recovers in 30s        |
| retry.last_retry_count > 0 for one specific user request                           | Transient 5xx/429/timeout on that turn           | Nothing -- retries caught the recovery. If persistent, check upstream rate-limit |
| retry.total_retries very high relative to uptime                                   | Persistent upstream instability                  | Investigate the LLM provider; consider raising max_retries (currently 2) in adapter config |
| last_error contains "json" or "schema"                                              | Model returning parse-failed JSON                | Try a different model via `OPENAI_COMPATIBLE_MODEL` env var                      |

### 1.4 — `faceswap.status == "degraded"` or `"error"`

**Most likely cause:** FaceFusion or Deep-Live-Cam dependencies missing (model
weights, ffmpeg, insightface). The sidecar container (`sidecar/Dockerfile.facefusion`)
holds the FaceFusion runtime — it lives separately from the main demo.

> Read `faceswap.detail.vendor_dir_exists`, `models_present`, `gpu_available`.

**Fix paths:**

| Symptom                                                                           | Likely cause                                     | Action                                                                          |
| --------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------- |
| vendor_dir_exists==false                                                           | missing `vendor/FaceFusion` checkout             | `git clone vendor/FaceFusion` (or rerun `make setup`)                            |
| models_present==false (inswapper_128_fp16.onnx)                                    | ONNX model weights missing                       | Hash-pinned download via the FaceFusion setup script                             |
| gpu_available==false                                                              | CPU-only host                                    | Expected on the demo sandbox -- this is the warning, not the error               |
| last_error contains "libcudart"                                                   | CUDA driver/library mismatch                     | Re-run on a CUDA host; check `nvidia-smi`                                         |

### 1.5 — `config.status == "degraded"`

Read `config.detail.affect_update_hz`, `renderer_url`, `hardware_profile`. The
config load is a synchronous YAML parse at startup; a degraded status here means
something prevented it from parsing.

**Fix paths:** Validate `packages/hermes_avatar/config/defaults.yaml` syntax
(`yamllint -d relaxed`). Check for missing required keys in any env overlays.

### 1.6 — `character_catalog.status == "degraded"`

Means the character folder scan failed. Read `count` and `ids`. If empty, ensure
`character_path` exists and contains image files matching `*.png/*.jpg/*.jpeg/*.webp`.

### 1.7 — `livetalking_reachability.status == "degraded"`

A simplified external reachability check (reuses renderer probe). Useful for
splitting "renderer module is loaded" from "renderer is actually answering HTTP".

---

## 2 — API quick reference

```bash
# Full health dump
curl -s http://localhost:8001/api/health | jq

# Just the breakers (the resilience trio)
curl -s http://localhost:8001/api/health | jq '.components | {
  voice_backend:  .voice_backend.detail.circuit_breaker,
  renderer:       .renderer.detail.circuit_breaker,
  protocol_agent: .protocol_agent.detail.circuit_breaker,
}' | jq

# Audit counter snapshot across all subsystems
curl -s http://localhost:8001/api/health | jq '.components.audit_log.detail'

# Prometheus metrics (existing endpoint -- runs alongside /api/health)
curl -s http://localhost:8001/metrics

# Per-request trace id is stamped on every response
curl -i http://localhost:8001/api/health | grep X-Trace-Id

# WebSocket-driven live state -- /ws is the avatar stream; for a snapshot:
curl -s http://localhost:8001/api/status | jq

# Trigger a vendor face-swap attempt (synthetic test surface for /api/v1/swap)
curl -X POST http://localhost:8001/api/v1/swap \
    -F 'source_face=@/path/to/source.jpg' \
    -F 'target_frame=@/path/to/frame.jpg' \
    -i 2>&1 | grep -E 'X-Swap-Mode|X-Latency-Ms|X-Trace-Id'
```

**Response headers worth knowing:**
- `X-Trace-Id` — per-request correlation; grep the structured logs.
- `X-Swap-Mode` — `passthrough` (no real swap ran) or `swap` (real inference fired).
- `X-Latency-Ms` — wall-clock milliseconds from the last `/api/v1/swap` request.

---

## 3 — Postmortem checklist

When something failed and you need to write a postmortem:

1. **Capture `/api/health` at the failure moment.** Every component status is the
   snapshot.
2. **Grep the JSON-aware log stream for `extra={"audit": ...}`.** Each breaker
   trip is one structured record; the `name` field tells you which subsystem
   (`breaker.luxtts`, `breaker.renderer`, `breaker.openai`); `kind` tells you
   the verb (`trip`/`recover`/`half_open`/`cost_cap_exceeded`/`vendor_fallback`).
3. **Walk the breaker sequence.** `half_open` → `trip` → `half_open` shows a
   rapid oscillation (provider barely recovered). A long gap with no events
   then a `trip` shows a quiet period followed by a hard outage. Cost-cap
   `cost_cap_exceeded` events in the same window mean resource-pressure was
   part of the failure mode.
4. **Cross-reference traces.** Take an `X-Trace-Id` from a failing user
   request and grep the log stream for that trace id; every downstream log
   line + every affect tick + every audit event in that request shares it.
5. **Check the audit counter cache under `/api/health.components.audit_log`.**
   `events_total` per subsystem tells you how often the same failure has fired
   over the process lifetime (or since `audit.reset()` was last called).

---

## 4 — When to escalate

1. **Three subsystems in "error" at once** — possible cascading outage from a
   shared dependency (e.g., the GPU host is down for both LiveTalking and
   FaceFusion). Pull `/api/health.components.config.detail.hardware_profile` to
   confirm the host is the common factor.
2. **`breakers cycle open→half-open→open` across multiple subsystems within a
   5-minute window** — likely an upstream provider outage that the breakers
   are correctly catching; the user impact is "graceful passthrough fallback
   instead of real inference" — this is the intended behavior.
3. **Persistent `cost_cap_exceeded` for >10 minutes** — the cost-cap default
   is a generous 240s/min. If this fires regularly, raise the env var.

When in doubt: `curl http://localhost:8001/api/health | jq` is the answer.
