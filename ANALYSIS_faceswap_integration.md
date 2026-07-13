# LiveEmote Face-Swap Integration Analysis

Date: 2026-07-12
Analyzer: Koda

## Executive Summary

LiveEmote's `DeepLiveCamAdapter` is a **non-functional stub**. `_activate_face_replacement()` sets in-memory flags and runs validation checks, but never invokes any actual face swapping — no model inference, no vendor code execution, no face compositing. Meanwhile, the vendored `Deep-Live-Cam` code in `vendor/Deep-Live-Cam/` contains a sophisticated, GPU-optimized face swap pipeline that is already present in the repo but completely unused.

**Primary recommendation: Rewrite DeepLiveCamAdapter to actually call the vendored Deep-Live-Cam pipeline.**

---

## 1. Current State: DeepLiveCamAdapter

```
deeplivecam_adapter.py
├── load_character()     → sets source_image_path, calls _activate_face_replacement()
├── set_theme()          → same pattern
├── set_behavior()       → sets self.behavior, calls _activate_face_replacement()
├── speak()              → same
├── interrupt()          → clears replacement_active flag
└── _activate_face_replacement()
    └── validates: enabled flag, character loaded, source image exists
    └── sets: self.replacement_active = True    ← NO INFERENCE
```

The adapter never:
- Loads ONNX models (inswapper, GFPGAN)
- Spawns a subprocess
- Calls insightface for face detection/swapping
- Produces any output frames
- Interacts with the vendor directory at all beyond checking `self.vendor_dir.exists()`

---

## 2. What's Already in vendor/Deep-Live-Cam

The vendored code already has a full production pipeline:

### Core Pipeline Components

| Module | Purpose | Lines |
|--------|---------|-------|
| `modules/processors/frame/face_swapper.py` | Inswapper ONNX model inference + Poisson blending | 1571 |
| `modules/face_analyser.py` | RetinaFace detection + ArcFace recognition | 371 |
| `modules/processors/frame/core.py` | FFmpeg pipe-based in-memory video processing | 407 |
| `modules/processors/frame/face_enhancer.py` | GFPGAN face enhancement | ~200 |
| `modules/processors/frame/face_masking.py` | Landmark-based face/mouth mask generation | 576 |
| `modules/processors/frame/face_enhancer_gpen256.py` | GPEN face enhancement (256) | ~150 |
| `modules/processors/frame/face_enhancer_gpen512.py` | GPEN face enhancement (512) | ~150 |
| `modules/video_capture.py` | Webcam capture with real FPS measurement | 159 |
| `modules/gpu_processing.py` | OpenCV CUDA wrappers (fallback to CPU) | 285 |
| `modules/onnx_optimize.py` | CoreML model rewriting (Pad→Slice+Concat, Shape folding) | 550 |
| `modules/capturer.py` | Video frame extraction | 33 |
| `modules/utilities.py` | FFmpeg helpers, temp dirs, frame I/O | ~500 |
| `modules/predicter.py` | NSFW content filter | 37 |
| `modules/cluster_analysis.py` | Face clustering for multi-face mapping | ~100 |

### Already-Integrated Optimizations

- **CUDA Graph replay**: Inswapper runs via recorded CUDA graphs (~0 CPU overhead per frame)
- **Pipelined detection**: Overlaps face detection with next frame's swap inference
- **CoreML optimization**: ONNX model rewriting for Apple Silicon ANE
- **Apple Silicon focus**: Detection on GPU, swap on ANE — concurrent execution
- **Hardware encoder fallback**: h264_nvenc → libx264 graceful degradation
- **Poisson blending**: seamlessClone with affine-locked mask (zero jitter)
- **FP16 model support**: Half-precision on Tensor Core GPUs
- **DML lock**: Thread-safe DirectML on Windows
- **Adaptive detection interval**: ~30 FPS detection rate in live mode

---

## 3. Repository Analysis

### Tier 1: High Impact — Directly Integrable

#### [facefusion/facefusion](https://github.com/facefusion/facefusion)
- **Stars**: Industry leading face manipulation platform
- **License**: OpenRAIL-AS (modified MIT, allows commercial use with restrictions)
- **Language**: Python
- **Key Extractable Components**:
  - Headless CLI mode with job management (create/submit/run/retry pipeline)
  - Batch processing architecture
  - Modular processor chain design
  - Benchmarking infrastructure
- **LiveEmote Relevance**: Could serve as an alternative backend. Its `headless-run` mode is more server-friendly than Deep-Live-Cam's tkinter GUI. The job-queue pattern matches LiveEmote's async architecture.
- **Extraction Effort**: Medium — the architecture is well-separated from its UI

#### [neuralchen/SimSwap](https://github.com/neuralchen/SimSwap)
- **Stars**: 5,178
- **License**: CC-BY-NC 4.0 (non-commercial only — not usable in LiveEmote without licensing change)
- **Language**: Python/PyTorch
- **Key Extractable Components**:
  - ID injection via ArcFace (alternative to inswapper's approach)
  - Generator + discriminator architecture with AAD ResBlocks
  - Multi-specific face swapping in video (face index → target mapping)
  - High-resolution (512×512) face swap via SimSwap-HQ
  - Training code with VGGFace2-HQ dataset pipeline
- **LiveEmote Relevance**: Research reference only. The ID injection + AAD ResBlock pattern is well-documented in the paper. The arcface-based identity encoding is the same family as insightface. **License blocks integration.**
- **Extraction Effort**: Low (read-only) — paper and architecture reference

#### [ai-forever/ghost (Sber GHOST)](https://github.com/ai-forever/ghost)
- **Stars**: 1,581
- **License**: Not explicitly stated (Apache-2.0-adjacent per Sber's typical pattern)
- **Language**: Python/PyTorch
- **Key Extractable Components**:
  - One-shot image-to-image and image-to-video face swap pipeline
  - ONNX-based inference support
  - Training code with configurable backbones
  - Colab inference notebook
  - IEEE-published methodology (IEEE Access 2022)
- **LiveEmote Relevance**: Good research reference. The ONNX inference path is compatible with LiveEmote's execution model. The one-shot approach is the same class as inswapper.
- **Extraction Effort**: Low (read-only reference)

### Tier 2: Medium Impact — Feature Additions

#### [numz/sd-wav2lip-uhq](https://github.com/numz/sd-wav2lip-uhq)
- **Stars**: 1,420
- **License**: Not explicitly stated (derived from Wav2Lip, which is MIT-like)
- **Language**: Python
- **Key Extractable Components**:
  - Face swapping integration with Wav2Lip lip-sync
  - GFPGAN/CodeFormer post-processing (face enhancement + upscaling)
  - Face index selection for multi-face videos
  - Mask dilation/erosion/blur controls
  - No-smooth mode (retains original mouth shape)
  - Padding adjustment for mouth alignment
- **LiveEmote Relevance**: Very relevant — bridges face swap + lip-sync. The mask controls and face enhancement pipeline directly apply to LiveEmote's rendering pipeline. The standalone version includes voice cloning and project management features.
- **Extraction Effort**: Medium — integrated with Stable Diffusion WebUI, needs decoupling

#### [visomaster/VisoMaster](https://github.com/visomaster/VisoMaster)
- **Stars**: 1,970
- **License**: Not explicitly stated ("hobby project")
- **Language**: Python
- **Key Extractable Components**:
  - Face Editor (LivePortrait models) — expression/pose adjustment
  - Expression Restorer — transfers original expressions to swapped face
  - Face embeddings from multiple source images for better accuracy
  - Live webcam → virtual camera streaming
  - OBS/StreamLabs/YouTube/Zoom integration pattern
  - TensorRT support
  - Video markers for per-frame adjustments
- **LiveEmote Relevance**: Expression restorer is the most valuable component — LiveEmote needs to preserve the avatar's expressions while replacing the face. The webcam streaming pattern is a reference for LiveEmote's meeting/streaming feature.
- **Extraction Effort**: Medium-High — Windows-focused installer, needs Linux adaptation

#### [jeeliz/jeelizFaceFilter](https://github.com/jeeliz/jeelizFaceFilter)
- **Stars**: 2,913
- **License**: Apache-2.0
- **Language**: JavaScript/WebGL
- **Key Extractable Components**:
  - Lightweight WebGL face tracking (no Python dependency)
  - Multi-face detection/tracking
  - 3D pose estimation (position, scale, Euler angles)
  - WebRTC camera integration pattern
  - Framework-agnostic API
- **LiveEmote Relevance**: Client-side face tracking for web-based avatar UI. Not directly applicable to Python server-side pipeline, but the lightweight approach is a reference for any browser-based preview feature.
- **Extraction Effort**: N/A (different platform)

### Tier 3: Low Impact — Minor Enhancements

#### [iVideoGameBoss/iRoopDeepFaceCam](https://github.com/iVideoGameBoss/iRoopDeepFaceCam)
- **Stars**: 250
- **License**: Not stated
- **Description**: roop fork with webcam support + OBS virtual camera integration
- **LiveEmote Relevance**: Minimal — already superseded by vendored Deep-Live-Cam (which was also a roop evolution)

#### [MaxMassi/Face-Swapper](https://github.com/MaxMassi/Face-Swapper)
- **Stars**: 19
- **License**: Not stated
- **Description**: Simple inswapper-based dataset creation tool
- **LiveEmote Relevance**: Too small. Uses the same inswapper model already vendored.

#### [ENGINEER-MUHAMMAD-SHAHZAIB/Deep-Stream-Cam](https://github.com/ENGINEER-MUHAMMAD-SHAHZAIB/Deep-Stream-Cam)
- **Stars**: 21
- **Description**: Deep-Live-Cam fork
- **LiveEmote Relevance**: None — lower quality fork of already-vendored code.

#### [mrezaakbari/DeepFake](https://github.com/mrezaakbari/DeepFake)
- **Stars**: 13
- **Description**: Another roop fork
- **LiveEmote Relevance**: None.

#### [codingdudecom/faceswap](https://github.com/codingdudecom/faceswap)
- **Stars**: 2
- **Description**: Basic face feature transfer script
- **LiveEmote Relevance**: None. 404'd.

#### [deepmancer/fake-webcam-streamer](https://github.com/deepmancer/fake-webcam-streamer)
- **Stars**: 6
- **Description**: Bash script for fake webcam via FFmpeg
- **LiveEmote Relevance**: Low — the FFmpeg v4l2loopback pattern is useful. FFmpeg virtual camera setup is a reference.

#### [tdv/fuel](https://github.com/tdv/fuel)
- **Stars**: 1
- **Description**: Intel GPU background removal for fake webcam
- **LiveEmote Relevance**: None. Very narrow (Intel GPU OpenVINO background removal only).

#### [facefusion/facefusion-pinokio](https://github.com/facefusion/facefusion-pinokio)
- **Stars**: 396
- **Description**: Facefusion installer wrapper
- **LiveEmote Relevance**: None — distribution mechanism only.

### Tier 4: Resource Lists

#### [aerophile/awesome-deepfakes](https://github.com/aerophile/awesome-deepfakes)
- **Stars**: 1,722
- **Description**: Curated list of deepfake resources
- **LiveEmote Relevance**: Research index only. Good for discovering new repos.

#### [MitchellX/deepfake-models](https://github.com/MitchellX/deepfake-models)
- **Stars**: 197
- **Description**: List of popular deepfake models
- **LiveEmote Relevance**: Directory list only.

---

## 4. Licensing Compatibility

| Repository | License | Compatible with AGPL-3.0? | Notes |
|------------|---------|--------------------------|-------|
| Deep-Live-Cam | AGPL-3.0 | ✓ Yes | Already vendored |
| facefusion | OpenRAIL-AS | ⚠️ Conditional | Research + redistribution may need checks |
| deepfakes/faceswap | GPL-3.0 | ✓ Yes | Compatible (GPL-3.0 → AGPL-3.0) |
| SimSwap | CC-BY-NC 4.0 | ✗ No | Non-commercial only — cannot integrate |
| jeelizFaceFilter | Apache-2.0 | ✓ Yes | Compatible |
| GHOST (Sber) | Not stated | ⚠️ Unknown | Google cache: likely Apache 2.0 |
| sd-wav2lip-uhq | Not stated | ⚠️ Unknown | Derived from Wav2Lip (MIT) |

**The Deep-Live-Cam AGPL-3.0 license is already compatible** since LiveEmote is AGPL-3.0. No new licensing issues arise from integrating the vendored code.

---

## 5. Prioritized Recommendations

### P0: Make DeepLiveCamAdapter Actually Work (2-3 days)

**What**: Rewrite `deeplivecam_adapter.py` to call the vendored Deep-Live-Cam pipeline.

**How**:
1. Import and call `modules.face_analyser.get_face_analyser()` to initialize insightface
2. Load the inswapper model via `modules.processors.frame.face_swapper.get_face_swapper()`
3. Create a subprocess-based webcam pipeline using the existing `VideoCapturer` class
4. Generate output frames and feed them to a virtual camera (v4l2loopback on Linux)
5. Use the existing `process_video_in_memory()` for pre-recorded video processing

**Key module dependencies**:
```
deeplivecam_adapter.py → face_analyser.py → insightface (RetinaFace)
                      → face_swapper.py   → inswapper ONNX model
                      → core.py           → FFmpeg pipe processing
                      → face_enhancer.py  → GFPGAN ONNX model
                      → face_masking.py   → Landmark-based compositing
                      → video_capture.py  → Webcam capture
```

**Files to modify**:
- `packages/hermes_avatar/renderer/deeplivecam_adapter.py` — full rewrite of `_activate_face_replacement()`
- `scripts/setup_deeplivecam_models.py` — may need additional model entries

**Required dependencies** (already listed in Deep-Live-Cam's requirements):
- `insightface` (face detection + recognition)
- `onnxruntime` / `onnxruntime-gpu`
- `opencv-python`
- `opennsfw2` (NSFW filter)
- `numpy`

### P1: Headless Frame Processing Server (3-5 days)

**What**: Port facefusion's headless job-queue architecture to run as a sidecar service that LiveEmote's HTTP adapter can talk to.

**Why**: The vendored Deep-Live-Cam has a tkinter GUI at its core — not server-friendly. facefusion's headless mode is designed for this.

**Implementation**: New module `packages/hermes_avatar/renderer/deeplivecam_server.py` that wraps the vendored pipeline behind an HTTP API (same pattern as LiveTalkingAdapter).

### P2: Expression Preservation (2-3 days)

**What**: Implement expression restorer (pattern from VisoMaster) that preserves the avatar's original facial expressions after face swapping.

**Why**: Current face swap replaces everything — you lose the avatar's expression. For a talking avatar, the expression (eyebrow raise, smile, surprise) is part of the performance.

**Approach**: Expression transfer via landmark warping or blend shapes, applied post-swap.

### P3: Anti-Jitter Compositing (1-2 days)

**What**: The vendored code already has excellent Poisson blending with affine-locked masks and cache-friendly elliptical masks. Ensure these are configured as defaults.

**Already implemented in vendored code**:
- `_create_elliptical_mask()` — cached per-size, zero jitter
- `_poisson_cached_mask` — reused when face is still
- `_fast_paste_back()` — O(crop_area) cost regardless of face size

### P4: Multi-Face Mapping (1-2 days)

**What**: Enable face mapping (which face maps to which target face) for multi-person meeting scenarios. The vendored code has cluster-based face mapping (`cluster_analysis.py`, `source_target_map`, `simple_map`) but it's unused in adapter mode.

### P5: Wav2Lip Integration (5-7 days)

**What**: Port the lip-sync enhancement pipeline from sd-wav2lip-uhq. This combines face swapping with Wav2Lip's accurate lip-sync and adds GFPGAN/CodeFormer post-processing.

**Note**: LiveEmote already has a LiveTalkingAdapter for an external runtime. This could be an alternative to the vendored approach.

---

## 6. Architecture Decision Matrix

| Approach | Effort | Performance | Maintainability | Static Face Quality | Lip-Sync Quality |
|----------|--------|-------------|-----------------|-------------------|-----------------|
| Fix DeepLiveCamAdapter (vendored DLC) | 2-3 days | Excellent (CUDA graphs, ANE) | Good (already vendored) | Good (inswapper+GFPGAN) | Requires Wav2Lip add-on |
| LiveTalkingAdapter (external runtime) | 0 (already done) | Depends on runtime | Best (HTTP decoupling) | Depends on runtime | Depends on runtime |
| facefusion headless server | 3-5 days | Good (similar to DLC) | Good (modular) | Good (same models) | Requires add-on |
| sd-wav2lip-uhq server | 5-7 days | Medium (Wav2Lip overhead) | Medium (needs decoupling from SD WebUI) | Best (CodeFormer post) | Best (native Wav2Lip) |

**Recommendation**: Fix DeepLiveCamAdapter first (it's the shortest path to a working face-swapped avatar), then add sd-wav2lip-uhq's Wav2Lip pipeline for lip-sync quality.

---

## 7. Extractable Code Patterns (Vendored)

### Pattern 1: GPU-optimized ONNX Inference Pipeline
Location: `vendor/Deep-Live-Cam/modules/processors/frame/face_swapper.py`
- CUDA graph session setup (`_init_cuda_graph_session`)
- CUDA graph replay (`_cuda_graph_swap_inference`)
- CoreML provider config with MLComputeUnits=ALL
- FP16 vs FP32 model selection based on GPU capability

### Pattern 2: FFmpeg Pipe Processing
Location: `vendor/Deep-Live-Cam/modules/processors/frame/core.py`
- Raw video frames via FFmpeg stdout pipe → process → pipe to encoder
- Hardware encoder + software encoder fallback chain
- Pipelined detection via ThreadPoolExecutor (detect frame N while processing frame N)
- Frame size validation + graceful pipe failure handling

### Pattern 3: Stable Poisson Compositing
Location: `vendor/Deep-Live-Cam/modules/processors/frame/face_swapper.py`
- Elastic cache for elliptical masks (keyed by size, never recomputed per frame)
- Dual path: preferred (affine-locked mask from swap transform) vs fallback (bbox ellipse)
- ROI-bounded compositing (preserves other faces)
- Mask erosion to keep Poisson seam on swapped-only pixels

### Pattern 4: Webcam Capture with Real FPS Measurement
Location: `vendor/Deep-Live-Cam/modules/video_capture.py`
- Empirical FPS measurement (warmup + timed read burst)
- DirectShow/MJPG construction param negotiation (Windows)
- Backend fallback chain (DSHOW → MSMF → ANY)
- Frame callback pattern for async processing

### Pattern 5: Thread-Safe Model Initialization
Location: `vendor/Deep-Live-Cam/modules/face_analyser.py`
- Double-checked locking pattern
- Lazy initialization with threading.Lock
- Conditional model loading (skip landmark model when only swapper active)
- Thread-safe face analysis with DML-specific lock

---

## 8. Files to Modify (Priority Order)

### Phase 1: Make It Work

1. **`packages/hermes_avatar/renderer/deeplivecam_adapter.py`**
   - Add model initialization (inswapper + GFPGAN)
   - Implement actual frame processing in `_activate_face_replacement()`
   - Add webcam capture via `VideoCapturer`
   - Add virtual camera output (v4l2loopback on Linux)
   - Replace stub state management with real pipeline orchestration

2. **`scripts/setup_deeplivecam_models.py`**
   - Add GPEN enhancer models if needed
   - Verify existing model download paths

### Phase 2: Production Hardening

3. **`packages/hermes_avatar/renderer/deeplivecam_adapter.py`**
   - Add health check + capability endpoint (matching LiveTalkingAdapter pattern)
   - Add async processing queue
   - Add graceful shutdown and resource cleanup

### Phase 3: Lip-Sync Enhancement

4. **New: `packages/hermes_avatar/renderer/wav2lip_adapter.py`**
   - Port Wav2Lip inference from sd-wav2lip-uhq
   - Integrate with existing voice pipeline (audio_path → lip-sync face)
   - Add CodeFormer or GFPGAN post-processing

---

## 9. Conclusion

The shortest path to a functional face-swapped avatar is to make `DeepLiveCamAdapter` actually call the vendored Deep-Live-Cam pipeline that's already sitting in `vendor/Deep-Live-Cam/modules/`. The code exists, is well-optimized (CUDA graphs, CoreML, pipelined detection, Poisson blending), and just needs to be wired up.

Of the 17 repositories evaluated, only **facefusion** (headless architecture) and **sd-wav2lip-uhq** (lip-sync integration) offer significant value beyond what's already vendored. The rest are either roop forks, research references, or resource lists.