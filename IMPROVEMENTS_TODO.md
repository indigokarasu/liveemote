# Hermes Live Avatar System - Improvement Recommendations

Based on code review, here are prioritized improvements for the Hermes Live Avatar system.

## Priority 1: Configuration Management

### 1.1 Add Environment Variable Support
**File:** `packages/hermes_avatar/config/schema.py`
**Description:** Add support for loading configuration from environment variables with nested prefix parsing (e.g., `AFFECT__UPDATE_HZ=60`).
**Implementation:** Extend `load_config()` function to check environment variables before falling back to YAML defaults.

### 1.2 Implement Configuration Validation
**File:** `packages/hermes_avatar/config/schema.py`
**Description:** Add custom validators for ranges and cross-field dependencies (e.g., ensuring mirror_min < mirror_max).
**Implementation:** Use Pydantic's `field_validator` decorator for custom validation logic.

### 1.3 Add Runtime Configuration Reloading
**File:** `packages/hermes_avatar/config/schema.py`
**Description:** Implement a mechanism to reload configuration at runtime without restarting the application.
**Implementation:** Add a `reload_config()` method and expose it via an API endpoint.

### 1.4 Hardware-Aware Defaults
**File:** `packages/hermes_avatar/config/schema.py`
**Description:** Make configuration defaults aware of available hardware (CPU cores, memory, GPU availability).
**Implementation:** Add hardware detection logic that adjusts defaults like thread counts, buffer sizes, etc.

## Priority 2: Error Handling and Resilience

### 2.1 Circuit Breaker Pattern
**File:** `packages/hermes_avatar/voice/luxtts_adapter.py` and similar adapter files
**Description:** Implement circuit breaker pattern for external service calls (TTS APIs, etc.) to prevent cascading failures.
**Implementation:** Add circuit breaker state tracking with timeout-based recovery.

### 2.2 Exponential Backoff Retry
**File:** All adapter files making external calls
**Description:** Add exponential backoff with jitter for failed external service calls.
**Implementation:** Create a retry decorator that can be applied to external API calls.

### 2.3 Improved Timeout Handling
**File:** `packages/hermes_avatar/voice/luxtts_adapter.py` and renderer adapters
**Status:** Completed (rolling 60s subprocess-second budget on LuxTTSAdapter;
  see `test_luxtts_cost_cap.py`).  Configurable via env var
  `LUXTTS_COST_CAP_SECONDS_PER_MINUTE` (default 240s/min).

### 2.4 Comprehensive Health Check Endpoints
**File:** `apps/demo_server/routes.py`
**Description:** Add detailed health check endpoints that report status of all subsystems.
**Implementation:** Create `/health` endpoint that checks database connections, external services, internal state, etc.

## Priority 3: Testing Coverage

### 3.1 Integration Tests for Full Pipeline
**File:** `tests/test_integration.py` (new)
**Description:** Add tests that cover the full perception → affect → behavior → rendering pipeline.
**Implementation:** Use mocked audio/video input to verify end-to-end behavior.

### 3.2 Property-Based Testing for Affect Algorithms
**File:** `tests/test_affect_properties.py` (new)
**Description:** Use property-based testing (hypothesis) to validate affect computation properties.
**Implementation:** Test properties like monotonicity, bounds, and continuity of affect functions.

### 3.3 Configuration Edge Case Tests
**File:** `tests/test_config.py` (new)
**Description:** Test configuration loading with various edge cases (missing values, invalid types, etc.).
**Implementation:** Parameterized tests for different config scenarios.

### 3.4 Load/Stress Tests
**File:** `tests/test_performance.py` (new)
**Description:** Add load testing for concurrent users and sustained operation.
**Implementation:** Use locust or similar to simulate multiple WebSocket connections.

### 3.5 WebSocket API Tests
**File:** `tests/test_websocket.py` (new)
**Description:** Test WebSocket connection handling, message parsing, and error scenarios.
**Implementation:** Use pytest-asyncio or similar to test async WebSocket behavior.

## Priority 4: Performance Optimizations

### 4.1 Profile and Optimize AffectRuntime.tick()
**File:** `packages/hermes_avatar/affect/policy.py`
**Description:** Profile the tick() method to identify bottlenecks and optimize.
**Implementation:** Use cProfile or similar to find hotspots, then optimize critical paths.

### 4.2 Vector Operations with NumPy
**File:** `packages/hermes_avatar/affect/smoothing.py`
**Description:** Consider using NumPy arrays for vector operations in smoothing functions.
**Implementation:** Refactor EMA and other math operations to use NumPy where beneficial.

### 4.3 Caching Strategies
**File:** Various modules
**Description:** Implement caching for expensive computations that don't change frequently.
**Implementation:** Add LRU caches for character asset lookups, configuration access, etc.

### 4.4 Object Pooling
**File:** `packages/hermes_avatar/affect/state.py`
**Description:** Consider object pooling for high-frequency allocations like AvatarBehaviorState.
**Implementation:** Add object pools for frequently allocated short-lived objects.

## Priority 5: Code Quality and Maintainability

### 5.1 Comprehensive Type Hints
**File:** All Python files
**Description:** Add missing type hints and improve existing ones.
**Implementation:** Run mypy to identify missing annotations and add them.

### 5.2 Extract Complex Methods
**File:** `packages/hermes_avatar/affect/policy.py`
**Description:** Break down large methods like `_update_face` and `_update_audio` into smaller functions.
**Implementation:** Extract facial expression processing, audio processing, etc. into helper methods.

### 5.3 Detailed Docstrings
**File:** Complex algorithm files
**Description:** Add detailed docstrings explaining the psychological models and mathematical formulas.
**Implementation:** Document the rationale behind affect computation algorithms.

### 5.4 Consistent Dataclass Usage
**File:** Throughout codebase
**Description:** Use dataclasses more consistently for data transfer objects.
**Implementation:** Convert appropriate classes to dataclasses with proper field specifications.

### 5.5 Structured Logging
**File:** Throughout codebase
**Status:** Completed (`hermes_avatar.util.audit` central helper + canonical
  `KIND_*` taxonomy + thread-safe per-name counter cache).  All three
  breakers and all three adapters now emit through the helper.

### 5.6 Comprehensive Inline Documentation
**File:** Throughout codebase
**Description:** Add more explanatory comments for non-obvious code sections.
**Implementation:** Focus on complex algorithms and configuration sections.

## Priority 6: Observability and Monitoring

### 6.1 Prometheus Metrics Endpoint
**File:** `apps/demo_server/routes.py` (new endpoint)
**Description:** Add Prometheus-compatible metrics endpoint.
**Implementation:** Expose counters, gauges, and histograms for key metrics (FPS, latency, error rates, etc.).

### 6.2 Distributed Tracing
**File:** Throughout codebase
**Description:** Add tracing capabilities for cross-service request tracing.
**Implementation:** Integrate with OpenTelemetry or similar for trace context propagation.

### 6.3 Enhanced Health Checks
**File:** `apps/demo_server/routes.py`
**Description:** Expand health checks to include dependency checks and performance metrics.
**Implementation:** Add checks for external service connectivity, processing lag, resource usage, etc.

### 6.4 Debugging Tools for Affect State
**File:** Add debug endpoints or WebSocket events
**Description:** Provide better visualization tools for affect state during development.
**Implementation:** Add debug endpoints that expose internal affect state for visualization.

### 6.5 Audit Logging for Configuration Changes
**File:** Configuration loading code + adapter lifecycle
**Status:** Completed (audit_event now fires on startup.renderer_degraded,
  http.request, breaker.trip/recover/half_open, retry.scheduled,
  retry.exhausted, vendor_fallback, cost_cap_exceeded).

## Operator Runbook

**Status:** Completed (this commit) -- see `apps/demo_server/RUNBOOK.md`.
Symptom -> component -> cause -> fix tables mapping every `/api/health`
component state to the operator action.  Driven off the live `/api/health`
shape from commit `c985d3f` + the audit counter cache added in this commit.

## Implementation Priority Order

1. **Immediate (Week 1):** Configuration improvements (1.1-1.4) and basic error handling (2.1-2.2)
2. **Short-term (Weeks 2-3):** Testing improvements (3.1-3.3) and performance profiling (4.1)
3. **Medium-term (Weeks 4-5):** Remaining performance optimizations (4.2-4.4) and code quality (5.1-5.3)
4. **Long-term (Week 6+):** Observability (6.1-6.5) and remaining code quality improvements

Each task should be implemented as a separate pull request with appropriate tests.


## Repository Evaluation Delegation

**Status:** Completed via KODA EM (Engineering Specialist)
**Task:** Evaluate face swapping/repo integration for LiveEmote improvements
**Kanban Task:** t_bc855b2e
**Completion Time:** Sun Jul 12 17:42:13 PDT 2026

### Results Summary
Koda evaluated 18+ face swapping and deepfake repositories for potential LiveEmote improvements. Key findings:

**Current State:** LiveEmote uses Deep-Live-Cam via DeepLiveCamAdapter, but the adapter is non-functional for actual face swapping.

**Primary Recommendation:** Evaluate FaceFusion as a replacement/alternative to Deep-Live-Cam due to:
- More recent active development (updated July 12, 2026)
- Industry-leading face manipulation platform
- Modular, job-based architecture
- Built-in benchmarking tools
- Includes lip-sync functionality
- Potentially more favorable licensing

**Secondary Recommendations:**
1. Create abstraction layer for hot-swapping between face swap backends
2. Implement performance tiers (preview vs. render modes)
3. Integrate benchmarking capabilities similar to FaceFusion
4. Implement fallback strategies for graceful degradation

**Next Steps:**
1. Proof-of-concept integration of FaceFusion core functionality
2. Side-by-side comparison with current Deep-Live-Cam implementation
3. License compliance verification
4. Community feedback gathering

**Related Files:** This evaluation complements the general improvement recommendations in this document, particularly those related to error handling, performance optimization, and observability.

## Implementation Status (2026-07-12)

ALL 28 improvement tasks PLUS the face-swap integration are COMPLETE and merged into `main` (HEAD `42a6c85`).

- **Improvement tasks** (config, error handling, testing, performance, code quality, observability) were implemented across branches `wt/config-1-env-vars`, `koda/obs`, `koda/perf`, `koda/test5` and consolidated into `koda/quality` (which also added a test-regression-fix commit). `koda/quality` was merged into `main`.
- **Face-swap integration** implemented in `koda/faceswap` (merged into `main`): new `packages/hermes_avatar/renderer/facefusion_adapter.py` (`FaceSwapAdapter` + `BackendManager` + frame pipeline), `DeepLiveCamAdapter` rewritten as a backward-compatible subclass of it, `FaceSwapConfig` added to `schema.py` with `FACESWAP__*` env overrides, wired into `demo_orchestrator.py` / `apps/demo_server/main.py` / `apps/demo_server/routes.py` health.
- **Verification:** full-repo `py_compile` clean; **67 unit/integration tests pass** (config, integration, observability, performance, affect-property, caching/pooling, face-swap).
- **Caveat — not runtime-tested here:** real GPU face-swap could not be executed in this environment (no GPU, no models, `cv2` absent). The `swap()` call sites are structured but degrade to passthrough in this env; they invoke the real FaceFusion / Deep-Live-Cam backend on a GPU host with models installed.
- **Not pushed to origin** (local branches only): `koda/obs`, `koda/perf`, `koda/quality`, `koda/test5`, `koda/faceswap`, `wt/config-1-env-vars`, `add-improvement-todos`.
