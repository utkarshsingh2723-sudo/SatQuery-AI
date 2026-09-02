<p align="center">
  <img src="https://img.shields.io/badge/SIH-2026-14b8a6?style=flat-square" alt="SIH 2026">
  <img src="https://img.shields.io/badge/Problem%20Statement-26167-818cf8?style=flat-square" alt="Problem Statement 26167">
  <img src="https://img.shields.io/badge/status-in%20development-eab308?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/license-MIT-64748b?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/python-3.10%2B-14b8a6?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Ollama-local%20inference-1e293b?style=flat-square" alt="Ollama">
  <img src="https://img.shields.io/badge/Streamlit-GUI-ff4b4b?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

<p align="center">
  An interactive vision-language assistant that answers natural-language questions about satellite imagery —
  built for <b>ISRO Problem Statement 26167</b>, Smart India Hackathon 2026.
</p>

---

## 📡 Overview

Satellite images are hard to read without training — SatQuery AI lets anyone ask a plain-English question about one, and get a plain-English answer back. Upload a single optical or SAR image, a co-registered optical–SAR pair, or two images of the same place taken at different times, and the assistant figures out what you're asking and routes it to the right analysis tool automatically.

| Capability | Description |
|---|---|
| 🖼️ Visual Question Answering | Ask anything about a single image — object counts, land features, scene description |
| 🏷️ Scene Classification | Automatic land-cover/scene labeling (urban, farmland, forest, water…) |
| 🔁 Change Detection | Compare two images of the same location across time and describe what changed |
| 📶 Optical–SAR Fusion | Joint analysis across optical and radar imagery of the same area |
| 🤖 Agentic Routing | No manual mode-switching — the system decides which tool your question needs |

---

## 🏗️ Architecture

A router agent (vision-language model) reads the query and image(s), decides which specialist tool applies, and hands off the pixel-level work to classical CV/GDAL rather than asking the model to do math it isn't built for. The router then formats the tool's output into a natural-language answer.

---

## ⚙️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Qwen2.5--VL-7B-14b8a6?style=for-the-badge" alt="Qwen2.5-VL">
  <img src="https://img.shields.io/badge/Ollama-local%20LLM-1e293b?style=for-the-badge" alt="Ollama">
  <img src="https://img.shields.io/badge/OpenCV-image%20processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/rasterio-GeoTIFF-4c9a2a?style=for-the-badge" alt="rasterio">
  <img src="https://img.shields.io/badge/Streamlit-GUI-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

---

## 📂 Project Structure

```
SatQuery-AI/
├── router/            # agentic dispatcher — decides which tool to call
├── tools/
│   ├── vqa/            # visual question answering
│   ├── classify/       # scene classification
│   ├── change/          # bi-temporal change detection
│   └── sar_fusion/      # optical–SAR joint analysis
├── gui/                # Streamlit app
├── data/                # sample datasets (RSVQA, VRSBench, BigEarthNet)
├── tests/               # benchmark evaluation scripts
└── assets/              # README images
```

---

## 🚀 Getting Started

```bash
# 1. Clone
git clone https://github.com/<your-username>/satquery-ai.git
cd satquery-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull the local models
ollama pull qwen2.5vl:7b
ollama pull moondream

# 4. Run the app
streamlit run gui/app.py
```

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) installed locally, a GPU with 8GB+ VRAM recommended.

---

## 🗺️ Roadmap

- [x] Problem scoping & architecture design
- [ ] Phase 1 — environment + data scaffolding
- [ ] Phase 2 — VQA module
- [ ] Phase 3 — scene classification module
- [ ] Phase 4 — change detection module
- [ ] Phase 5 — optical–SAR fusion + router
- [ ] Phase 6 — GUI
- [ ] Phase 7 — benchmarking & polish

---

## 📊 Datasets & Benchmarks

- [RSVQA](http://rsvqa.sylvainlobry.com/) — visual question answering on satellite imagery
- [VRSBench](https://huggingface.co/datasets/xiang709/VRSBench) — VQA, captioning, and SAR benchmark set
- [BigEarthNet](https://bigearth.net/) — land-cover scene classification

---

## 📄 License

This project is released under the MIT License.

---

<p align="center"><sub>Built for Smart India Hackathon 2026 · Indian Space Research Organisation (ISRO)</sub></p>
