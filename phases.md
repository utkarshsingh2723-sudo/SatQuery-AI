# SatQuery AI — Master Prompt & Phases for Claude Opus (Antigravity)

How to use this: paste the **Master Context Block** once at the start of your Antigravity session (or keep it pinned/in a project rules file). Then feed the phases to Opus **one at a time, in order** — don't dump all 7 phases in one prompt, agentic coding tools do much better with one bounded phase per turn. Each phase block below is ready to paste as-is. Wait for a phase to be working before moving to the next.

---

## Master Context Block (paste first)

```
We are building SatQuery AI for SIH 2026, ISRO Problem Statement 26167:
"SatQuery AI - An Interactive Vision-Language Assistant for Multimodal
Remote Sensing Image Analysis through Text Queries."

Requirements we must satisfy:
- Support single optical/multispectral or SAR images, co-registered
  optical-SAR pairs, and bi-temporal pairs, in GeoTIFF/TIFF format.
- Perform: single-image VQA, one additional single-image task (scene
  classification), change analysis on bi-temporal pairs, optical-SAR
  joint analysis, and agentic model/tool selection — all through an
  interactive GUI/web app.
- Will be evaluated on public benchmark test subsets (RSVQA, VRSBench)
  plus a held-out ISRO/SAC dataset with Cartosat-2S optical and RISAT
  SAR pairs.

Architecture (do not deviate without discussion): an agentic orchestrator,
not one fine-tuned monolith model.
- A router agent (vision-language model) reads the query + image(s) and
  decides which specialist tool to call.
- Specialist tools are separate modules: VQA, scene classification,
  change detection, optical-SAR fusion.
- The VLM's job is reasoning and language only. Pixel-level math
  (differencing, thresholding, co-registration, overlays) is done with
  classical CV / GDAL / rasterio, not asked of the VLM.
- Router formats tool output into a final natural-language answer,
  returned to the GUI along with any relevant image overlay.

Hardware and model constraints (hard limits, do not suggest anything
that violates these without flagging it explicitly):
- Local machine: Windows PC, RTX 5050 GPU, 8GB VRAM only.
- Primary VLM: qwen2.5vl:7b served locally via Ollama (already pulled).
- Fallback VLM if OOM or too slow: moondream or llava:7b-v1.6-q4 via
  Ollama.
- No fine-tuning of the VLM — zero-shot + prompting only. If a task
  needs a trained model (e.g. scene classifier), use a small pretrained
  model (ResNet18/EfficientNet class) that comfortably fits in 8GB, not
  the VLM.
- Never run more than one VLM inference in parallel — sequential calls
  only, to avoid VRAM OOM.

Timeline: 7 days total, one phase per day roughly. Prioritize a working
end-to-end pipeline over any single polished component. A demo that
covers all 4 task types badly beats one perfect task type and 3 missing
ones.

Project structure on disk (Windows, C:\FILES\SIH\Demo):
/router          - agentic dispatcher logic
/tools/vqa
/tools/classify
/tools/change
/tools/sar_fusion
/gui             - Streamlit or Flask app
/data            - sample datasets (RSVQA-LR, VRSBench, BigEarthNet subset)
/tests           - benchmark eval scripts

For every phase: write working code, not pseudocode. Explain any
assumption you make before writing code, in one or two lines, then
implement it. If something in this context conflicts with what you're
about to do, stop and flag it instead of silently working around it.
```

---

## Phase 1 — Environment + data scaffolding

```
Phase 1 of SatQuery AI. Set up the project scaffolding described in the
master context.

Do:
1. Create the folder structure listed in the master context.
2. Write a `requirements.txt` / environment setup (ollama client,
   rasterio, opencv-python, streamlit, pillow, numpy).
3. Write a small script `tools/ollama_client.py` that sends an image +
   text prompt to qwen2.5vl:7b via the local Ollama API and returns the
   text response. Include a basic retry/timeout and a fallback to
   moondream if the primary model call fails or times out.
4. Write `tools/geotiff_utils.py`: load a GeoTIFF/TIFF file, return it
   as a normal RGB array/PIL image the VLM can consume, and preserve
   georeferencing metadata for later use.
5. Do NOT build the GUI or router yet — this phase is only the
   foundation. Confirm each piece works with a tiny standalone test
   script before moving on.

Acceptance: I can run one script that loads a sample GeoTIFF and gets a
text answer back from qwen2.5vl for a hardcoded question, end to end.
```

## Phase 2 — VQA module

```
Phase 2 of SatQuery AI. Build the VQA specialist tool.

Do:
1. `tools/vqa/vqa_tool.py`: a function that takes an image + a natural
   language question, calls the Ollama client from Phase 1, and returns
   a clean text answer.
2. Handle failure modes: empty/garbled model output, timeout, unsupported
   image format — return a clear error rather than crashing.
3. Write a small eval script `tests/eval_vqa.py` that runs this tool
   against 15-20 sample RSVQA-LR question/image pairs and logs the
   answers next to the ground truth (informal accuracy check, not a
   full benchmark run yet).

Acceptance: running the eval script produces a log of predicted vs
ground-truth answers for the sample set, with no crashes.
```

## Phase 3 — Scene classification tool

```
Phase 3 of SatQuery AI. Build the scene classification specialist tool
(the required "additional single-image task").

Do:
1. `tools/classify/classify_tool.py`: load a small pretrained image
   classifier (ResNet18 or EfficientNet-B0 class) with weights suited to
   land-cover/scene classification (BigEarthNet or EuroSAT style classes).
   If suitable pretrained weights aren't easily available, fall back to
   prompting qwen2.5vl to classify the scene into a fixed label set —
   note explicitly which approach you used and why.
2. Function takes an image, returns a predicted class + confidence.
3. Wire this as a second callable tool with the same input/output shape
   as the VQA tool from Phase 2, so the router can treat them uniformly.

Acceptance: given a sample image, the tool returns a scene class label
without crashing, and the code is structured so this tool and the VQA
tool are interchangeable from the caller's perspective.
```

## Phase 4 — Change detection tool

```
Phase 4 of SatQuery AI. Build the change detection specialist tool for
bi-temporal image pairs.

Do:
1. `tools/change/change_tool.py`: takes two images of the same area at
   different times.
2. Do simple co-registration (doesn't need to be perfect — approximate
   alignment is fine for this timeline).
3. Compute a difference/change map using classical CV (image differencing
   + thresholding via OpenCV) — do not ask the VLM to do this math.
4. Pass the original pair + the change map to qwen2.5vl and prompt it to
   describe, in plain language, what changed between the two images.
5. Return both the text description and the change-map image (for the
   GUI to display later).

Acceptance: given a sample bi-temporal pair, the tool returns a natural-
language change summary plus a visual diff image, without crashing.
```

## Phase 5 — Optical-SAR fusion tool + router

```
Phase 5 of SatQuery AI. Build the optical-SAR joint analysis tool and the
router that ties all four tools together.

Do:
1. `tools/sar_fusion/sar_tool.py`: takes a co-registered optical + SAR
   pair, computes a basic overlay/comparison (simple correlation or edge
   comparison via classical CV, not the VLM), then prompts qwen2.5vl to
   narrate the comparison in plain language.
2. `router/router.py`: the agentic dispatcher. Given a user's text query
   (and the images provided), decide which of the four tools (VQA,
   classify, change, sar_fusion) to call. Use a simple few-shot prompt to
   qwen2.5vl for this classification, with a keyword-based fallback if
   the model's answer is ambiguous. Keep this simple — it does not need
   to be a complex planning agent.
3. Router should call the chosen tool, then format the tool's raw output
   into a final answer for the user.

Acceptance: I can call the router with a query + appropriate image(s) for
each of the 4 task types and get a correctly-dispatched, sensible
response for all 4.
```

## Phase 6 — GUI

```
Phase 6 of SatQuery AI. Build the interactive GUI required by the problem
statement.

Do:
1. Build a Streamlit app in /gui: file upload for GeoTIFF/TIFF (single
   image, optical-SAR pair, or bi-temporal pair), a text query box, and a
   "submit" flow that calls the router from Phase 5.
2. Display the router's answer, and any relevant overlay image (change
   map, SAR-optical comparison) it returns.
3. Add basic loading states and error handling so a bad upload or a
   model failure shows a clear message instead of crashing the app.
4. Do an end-to-end smoke test: upload one example for each of the 4
   task types and confirm each produces a sensible answer through the
   full GUI flow.

Acceptance: the app runs locally, accepts all 3 input types (single
image, optical-SAR pair, bi-temporal pair), and produces a correct-
looking answer end to end for each task type.
```

## Phase 7 — Benchmark, polish, submission prep

```
Phase 7 of SatQuery AI, final phase.

Do:
1. `tests/eval_benchmarks.py`: run the VQA and classification tools
   against the RSVQA/VRSBench public test subsets we have locally, log
   results in a simple report (this mirrors how ISRO will actually score
   us, so treat it as the real check, not a formality).
2. Fix any obvious failure modes surfaced by the benchmark run or the
   Phase 6 smoke test.
3. Add small UX polish to the GUI (clear labels, sensible defaults,
   loading indicators) — no new features.
4. Do NOT start anything new that isn't already scoped in Phases 1-6.
   This phase is stabilization only.

Acceptance: benchmark script produces a results log, the GUI runs without
known crashes across all 4 task types, and the project is in a state
that can be recorded as a demo video.
```
