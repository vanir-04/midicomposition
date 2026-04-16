# MIDI Music Generator

AI-powered music continuation tool built with **Gradio + OpenMuseNet 3**, capable of generating MIDI continuations from both **audio and MIDI inputs**, with built-in **visualization and coherence analysis**.

---

##  Features

###  Music Generation
- Generate MIDI continuations using **OpenMuseNet 3**
- Supports:
  - 🎹 MIDI input (`.mid`, `.midi`)
  - 🎙️ Audio input (`.wav`, `.mp3`, `.flac`, `.ogg`, `.aiff`, `.m4a`)
- Iterative generation (extend music multiple times)
- Genre-conditioned output:
  - `classical`, `rock`, `metal`, `pop`, `jazz`, `romantic`

---

###  Audio → MIDI Transcription
- Uses **Spotify Basic Pitch** for automatic transcription
- Converts audio into MIDI before generation

---

###  Coherence Metrics
-  Pitch similarity
-  Interval similarity
-  KL divergence
-  Rhythmic regularity
-  Density ratio
-  Polyphony
-  Overall coherence score (0–100)

---

###  Visualizations
-  Piano roll comparison
-  Radar chart + bar chart
-  Detailed metric table

---

##  Model Details
- Model: `kobimusic/maestro-4-genre`
- Framework: PyTorch
- Auto device detection (CUDA / MPS / CPU)

---

##  Installation

```bash
git clone https://github.com/vanir-04/midicomposition.git
cd midicomposition
pip install -r requirements.txt
python app.py
```

---

##  Usage

1. Load model
2. Upload audio/MIDI
3. Configure parameters
4. Generate music
5. Analyze results

---

##  Parameters

| Parameter     | Description |
|--------------|------------|
| Iterations   | Number of generation steps |
| Genre        | Style conditioning |
| Temperature  | Creativity control |

---

## 📁 Project Structure

```
.
├── app.py
├── model_cache/
├── OpenMusenet2/
├── README.md
```

---

##  Notes
- GPU recommended
- CPU supported but slower
