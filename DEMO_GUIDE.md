# 🛰️ SatQuery AI — Demo & Live Run Cheat Sheet

Quick copy-paste guide to run **SatQuery AI** locally and generate a live mobile-friendly link for judges/evaluators.

---

## ⚡ Quick 3-Step Demo Launch (Windows CMD)

Open **3 separate Command Prompt (`cmd.exe`)** windows:

### 1️⃣ CMD Window 1: Start Ollama VLM Server
```cmd
ollama serve
```
*(Make sure `qwen2.5vl:7b` model is installed: `ollama pull qwen2.5vl:7b`)*

---

### 2️⃣ CMD Window 2: Start Streamlit GUI
```cmd
cd /d C:\FILES\SIH\Demo
python -m streamlit run gui\app.py
```
*App will open locally at `http://localhost:8501`.*

---

### 3️⃣ CMD Window 3: Generate Live Public HTTPS URL (for Judges / Phones)
```cmd
cloudflared tunnel --url http://localhost:8501
```
*Copy the generated `https://xxxx.trycloudflare.com` URL and share it with judges or open on your phone!*

---

## 🛑 How to Stop & Free Ports (Kill Processes)

If you need to stop all running instances and free up ports (`8501` & `11434`), run in CMD:

```cmd
taskkill /F /IM streamlit.exe
taskkill /F /IM python.exe
taskkill /F /IM ollama.exe /T
```

---

## 🧪 Quick Test Commands (Optional)

Run automated offline/integration tests in CMD:

```cmd
cd /d C:\FILES\SIH\Demo

:: Test Router
python tests/test_router.py

:: Test Scene Classifier (EuroSAT ResNet-18)
python tests/test_classify.py

:: Test Change Detection (CV differencing)
python tests/test_change.py
```
