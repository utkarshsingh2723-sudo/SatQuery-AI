# SatQuery AI — 7-Day Build Plan
**SIH 2026 · Problem Statement 26167 (ISRO)**
Approach: agentic orchestrator + specialist tools, running mostly on Ollama (RTX 5050, 8GB VRAM)

---

## Model picks for your GPU

| Role | Model | Why |
|---|---|---|
| Router / VQA / captioning | `qwen2.5vl:7b` (Ollama, q4 quant) | Vision-language, easy to run via Ollama, strong zero-shot VQA/captioning on general imagery |
| Backup / lighter | `moondream` or `llava:7b-v1.6-q4` | Falls back to these if 7B Qwen is too slow/OOM on 8GB — smaller footprint |
| Scene classification | Small pretrained CNN (ResNet18/EfficientNet on EuroSAT or BigEarthNet weights, via `transformers`/`timm`) | Runs comfortably on 8GB, doesn't need the VLM at all |
| Change detection | Classical CV first (image differencing + thresholding via OpenCV), VLM only writes the natural-language summary | Zero training needed, reliable baseline |
| Optical-SAR fusion | Co-registration (rasterio/GDAL) + simple overlay/statistics, VLM narrates the comparison | Same idea — don't make the model do the math, make it describe the result |

**Rule for the week:** the VLM's job is *reasoning and language*, not pixel math. Classical CV/GDAL does the pixel math. This is what keeps it buildable on 8GB VRAM in 7 days.

---

## Day 1 — Setup + data
- [ ] Repo + folder structure (`/router`, `/tools/vqa`, `/tools/change`, `/tools/sar_fusion`, `/tools/classify`, `/gui`)
- [ ] Pull `ollama pull qwen2.5vl:7b`, test a single image + text query from the CLI
- [ ] Download sample sets: RSVQA-LR (VQA), VRSBench (VQA + captioning), a BigEarthNet subset (classification), one bi-temporal pair for change detection
- [ ] Confirm you can read GeoTIFF with `rasterio` and display it as a normal image (this trips people up — do it day 1, not day 5)

## Day 2 — VQA module
- [ ] Build `tools/vqa`: send image + question to `qwen2.5vl` via Ollama's API, return answer
- [ ] Test on 15–20 RSVQA-LR questions, log accuracy informally
- [ ] Handle basic failure cases (blank answer, model refusal, timeout)

## Day 3 — Scene classification / captioning (the "extra single-image task")
- [ ] Pick one: lightweight classifier (fast, safe) or VLM-prompted captioning (reuses Day 2 pipeline, less code)
- [ ] If classifier: fine-tune/load a small pretrained model on BigEarthNet classes, wrap it as a tool
- [ ] Wire it in as a second callable tool alongside VQA

## Day 4 — Change detection
- [ ] Co-register the bi-temporal pair (simple alignment, doesn't need to be perfect)
- [ ] Image differencing + thresholding → highlight changed regions with OpenCV
- [ ] Feed the diff map + both images to the VLM, prompt it to describe what changed in plain language

## Day 5 — Optical-SAR joint analysis + router logic
- [ ] Basic SAR-optical overlay/statistics (co-registration, simple correlation or edge comparison)
- [ ] VLM narrates the comparison
- [ ] Build the router: a prompt-based classifier ("is this query about VQA / change / SAR-optical / classification?") that dispatches to the right tool — this is your "agentic tool selection" requirement, keep it simple (few-shot prompt or keyword + LLM fallback is fine)

## Day 6 — GUI + integration
- [ ] Streamlit (fastest for a solo/small team) or a minimal Flask+HTML front end
- [ ] File upload for GeoTIFF/TIFF + optical/SAR pairs + bi-temporal pairs
- [ ] Query box → router → tool → response, shown with the relevant image overlay
- [ ] End-to-end smoke test on all 4 task types

## Day 7 — Benchmark, polish, submit
- [ ] Run against RSVQA/VRSBench test subsets — this is literally how ISRO scores you, so test against the real thing, not just eyeballing
- [ ] Fix obvious failure modes, add loading states / error handling in the GUI
- [ ] Record a short demo video, write up the approach (mention the agentic router explicitly — it directly answers a named requirement)
- [ ] Buffer time for last-minute bugs — don't schedule new features on Day 7

---

## Things that will eat your week if you don't watch them
- **GeoTIFF handling** — always harder than expected, do it Day 1.
- **Trying to fine-tune the VLM** — don't. Zero-shot + good prompts is the entire point of this approach.
- **Perfect co-registration** — approximate alignment is fine for a hackathon demo; don't sink a day into it.
- **VRAM OOM** — if `qwen2.5vl:7b` chokes on your 8GB card under load, drop to `moondream` or run the VLM calls sequentially, never in parallel.
