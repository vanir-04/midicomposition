import gradio as gr
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import traceback
import io
import json
import math
import numpy as np


# ── Global state ───────────────────────────────────────────────────────────────
setup_complete = False
ai_model = None
MIDI_class = None
DEVICE = None
MODEL_CACHE_DIR = "./model_cache"


# ── Piano roll renderer ────────────────────────────────────────────────────────


def midi_to_notes(midi_path):
    from mido import MidiFile
    mid = MidiFile(midi_path)
    tempo = 500000
    ticks_per_beat = mid.ticks_per_beat
    notes = []
    for track in mid.tracks:
        active = {}
        current_tempo = tempo
        abs_time_sec = 0.0
        for msg in track:
            delta_sec = (msg.time / ticks_per_beat) * (current_tempo / 1_000_000)
            abs_time_sec += delta_sec
            if msg.type == "set_tempo":
                current_tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = abs_time_sec
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    notes.append((msg.note, active.pop(msg.note), abs_time_sec))
    return notes


# ── Coherence Metrics ──────────────────────────────────────────────────────────


def pitch_histogram_similarity(orig_notes, gen_notes):
    from scipy.stats import entropy as scipy_entropy
    orig_hist = np.zeros(128)
    gen_hist = np.zeros(128)
    for pitch, _, _ in orig_notes:
        orig_hist[pitch] += 1
    for pitch, _, _ in gen_notes:
        gen_hist[pitch] += 1
    orig_hist = orig_hist / (orig_hist.sum() + 1e-9)
    gen_hist = gen_hist / (gen_hist.sum() + 1e-9)
    cos_sim = float(np.dot(orig_hist, gen_hist) / (
        np.linalg.norm(orig_hist) * np.linalg.norm(gen_hist) + 1e-9
    ))
    kl_div = float(scipy_entropy(orig_hist + 1e-9, gen_hist + 1e-9))
    return cos_sim, kl_div



def rhythmic_regularity(notes, bpm=120, subdivisions=16):
    beat_len = 60 / bpm
    grid_unit = beat_len / (subdivisions / 4)
    scores = []
    for _, start, _ in notes:
        offset = start % grid_unit
        deviation = min(offset, grid_unit - offset) / (grid_unit / 2)
        scores.append(1.0 - deviation)
    return float(np.mean(scores)) if scores else 0.0



def note_density(notes):
    if len(notes) < 2:
        return 0.0
    duration = max(e for _, _, e in notes) - min(s for _, s, _ in notes)
    return len(notes) / (duration + 1e-9)



def density_ratio(orig_notes, gen_notes):
    return note_density(gen_notes) / (note_density(orig_notes) + 1e-9)



def interval_distribution(notes):
    pitches = [p for p, _, _ in sorted(notes, key=lambda x: x[1])]
    intervals = np.abs(np.diff(pitches))
    hist = np.zeros(25)
    for i in intervals:
        if i <= 24:
            hist[int(i)] += 1
    return hist / (hist.sum() + 1e-9)



def interval_similarity(orig_notes, gen_notes):
    oh = interval_distribution(orig_notes)
    gh = interval_distribution(gen_notes)
    denom = np.linalg.norm(oh) * np.linalg.norm(gh) + 1e-9
    return float(np.dot(oh, gh) / denom)



def avg_polyphony(notes, time_resolution=0.05):
    if not notes:
        return 0.0
    end_time = max(e for _, _, e in notes)
    times = np.arange(0, end_time, time_resolution)
    counts = [sum(1 for _, s, e in notes if s <= t < e) for t in times]
    return float(np.mean(counts))



def coherence_score(orig_notes, gen_notes):
    pitch_sim, _ = pitch_histogram_similarity(orig_notes, gen_notes)
    int_sim = interval_similarity(orig_notes, gen_notes)
    dr = min(density_ratio(orig_notes, gen_notes), 2.0)
    density_score = max(0.0, 1.0 - abs(1.0 - dr))
    score = 0.40 * pitch_sim + 0.40 * int_sim + 0.20 * density_score
    return round(score * 100, 1)



def compute_all_metrics(orig_notes, gen_notes):
    pitch_sim, kl_div = pitch_histogram_similarity(orig_notes, gen_notes)
    int_sim = interval_similarity(orig_notes, gen_notes)
    dr = density_ratio(orig_notes, gen_notes)
    rhythm_orig = rhythmic_regularity(orig_notes)
    rhythm_gen = rhythmic_regularity(gen_notes)
    poly_orig = avg_polyphony(orig_notes)
    poly_gen = avg_polyphony(gen_notes)
    overall = coherence_score(orig_notes, gen_notes)

    return {
        "overall": overall,
        "pitch_sim": round(pitch_sim * 100, 1),
        "kl_div": round(kl_div, 3),
        "interval_sim": round(int_sim * 100, 1),
        "density_ratio": round(dr, 2),
        "rhythm_orig": round(rhythm_orig * 100, 1),
        "rhythm_gen": round(rhythm_gen * 100, 1),
        "poly_orig": round(poly_orig, 2),
        "poly_gen": round(poly_gen, 2),
        "note_count_orig": len(orig_notes),
        "note_count_gen": len(gen_notes),
        "density_orig": round(note_density(orig_notes), 2),
        "density_gen": round(note_density(gen_notes), 2),
    }



def _radar_svg(labels, orig_vals, gen_vals, size=320):
    cx = cy = size / 2
    radius = size * 0.32
    levels = [25, 50, 75, 100]

    def pt(i, value):
        angle = -math.pi / 2 + (2 * math.pi * i / len(labels))
        r = radius * (value / 100.0)
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        return x, y

    def poly(vals):
        return " ".join(f"{x:.1f},{y:.1f}" for i, v in enumerate(vals) for x, y in [pt(i, v)])

    rings = []
    for lvl in levels:
        rings.append(f'<polygon points="{poly([lvl]*len(labels))}" fill="none" stroke="#1c2030" stroke-width="1" />')
    spokes = []
    for i in range(len(labels)):
        x, y = pt(i, 100)
        spokes.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#2a2f42" stroke-width="1" />')

    text = []
    for i, label in enumerate(labels):
        x, y = pt(i, 112)
        anchor = "middle"
        if x < cx - 8:
            anchor = "end"
        elif x > cx + 8:
            anchor = "start"
        text.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="#94a3b8" font-size="11" text-anchor="{anchor}" dominant-baseline="middle">{label}</text>')

    return f'''
<svg viewBox="0 0 {size} {size}" width="100%" height="220" role="img" aria-label="Radar chart comparing original and generated music metrics">
  {''.join(rings)}
  {''.join(spokes)}
  <polygon points="{poly(orig_vals)}" fill="rgba(91,140,255,0.14)" stroke="rgba(91,140,255,0.8)" stroke-width="2" />
  <polygon points="{poly(gen_vals)}" fill="rgba(167,139,250,0.18)" stroke="#a78bfa" stroke-width="2.5" />
  {''.join(text)}
</svg>
'''



def _bar_svg(labels, orig_vals, gen_vals, width=420, height=240):
    left = 44
    right = 16
    top = 16
    bottom = 36
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max(max(orig_vals), max(gen_vals), 1)
    max_val *= 1.15
    group_w = plot_w / len(labels)
    bar_w = min(28, group_w * 0.24)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="220" role="img" aria-label="Bar chart comparing original and generated metrics">']
    for i in range(5):
        y = top + plot_h * i / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#1c2030" stroke-width="1" />')
        val = max_val * (1 - i / 4)
        parts.append(f'<text x="{left-6}" y="{y+4:.1f}" fill="#64748b" font-size="10" text-anchor="end">{val:.0f}</text>')

    for i, label in enumerate(labels):
        gx = left + group_w * i + group_w / 2
        for j, val in enumerate([orig_vals[i], gen_vals[i]]):
            x = gx + (-bar_w*0.7 if j == 0 else bar_w*0.7) - bar_w/2
            h = (val / max_val) * plot_h
            y = top + plot_h - h
            color = 'rgba(91,140,255,0.7)' if j == 0 else 'rgba(167,139,250,0.78)'
            stroke = 'rgba(91,140,255,0.95)' if j == 0 else '#a78bfa'
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="4" fill="{color}" stroke="{stroke}" stroke-width="1" />')
        parts.append(f'<text x="{gx:.1f}" y="{height-12}" fill="#94a3b8" font-size="11" text-anchor="middle">{label}</text>')

    parts.append(f'<line x1="{left}" y1="{top+plot_h:.1f}" x2="{width-right}" y2="{top+plot_h:.1f}" stroke="#2a2f42" stroke-width="1" />')
    parts.append('</svg>')
    return ''.join(parts)



def build_metrics_html(metrics: dict) -> str:
    m = metrics
    score_color = (
        "#34d399" if m["overall"] >= 70
        else "#f59e0b" if m["overall"] >= 40
        else "#f87171"
    )

    radar_labels = ["Pitch", "Interval", "Rhythm", "Density", "Polyphony"]
    density_score_orig = 100.0
    density_score_gen = round(max(0.0, 1.0 - abs(1.0 - m["density_ratio"])) * 100, 1)
    poly_max = max(m["poly_orig"], m["poly_gen"], 1)
    poly_norm_orig = round(m["poly_orig"] / poly_max * 100, 1)
    poly_norm_gen = round(m["poly_gen"] / poly_max * 100, 1)

    orig_radar = [100.0, 100.0, m["rhythm_orig"], density_score_orig, poly_norm_orig]
    gen_radar = [m["pitch_sim"], m["interval_sim"], m["rhythm_gen"], density_score_gen, poly_norm_gen]

    radar_svg = _radar_svg(radar_labels, orig_radar, gen_radar)
    bar_svg = _bar_svg(
        ["Notes/sec", "Avg voices", "Rhythm"],
        [m["density_orig"], m["poly_orig"], m["rhythm_orig"]],
        [m["density_gen"], m["poly_gen"], m["rhythm_gen"]],
    )

    html = f"""
<div style="font-family:'DM Mono',monospace;padding:0 4px;">
  <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:20px;">
    <div style="background:#14171f;border:1px solid #2a2f42;border-radius:10px;padding:16px 12px;text-align:center;">
      <div style="font-size:0.62rem;letter-spacing:.1em;text-transform:uppercase;color:#64748b;margin-bottom:6px;">Overall Score</div>
      <div style="font-size:2rem;font-weight:700;color:{score_color};">{m['overall']}</div>
      <div style="font-size:0.65rem;color:#475569;">/ 100</div>
    </div>
    <div style="background:#14171f;border:1px solid #2a2f42;border-radius:10px;padding:16px 12px;text-align:center;">
      <div style="font-size:0.62rem;letter-spacing:.1em;text-transform:uppercase;color:#64748b;margin-bottom:6px;">Pitch Match</div>
      <div style="font-size:2rem;font-weight:700;color:#5b8cff;">{m['pitch_sim']}</div>
      <div style="font-size:0.65rem;color:#475569;">/ 100</div>
    </div>
    <div style="background:#14171f;border:1px solid #2a2f42;border-radius:10px;padding:16px 12px;text-align:center;">
      <div style="font-size:0.62rem;letter-spacing:.1em;text-transform:uppercase;color:#64748b;margin-bottom:6px;">Interval Match</div>
      <div style="font-size:2rem;font-weight:700;color:#a78bfa;">{m['interval_sim']}</div>
      <div style="font-size:0.65rem;color:#475569;">/ 100</div>
    </div>
    <div style="background:#14171f;border:1px solid #2a2f42;border-radius:10px;padding:16px 12px;text-align:center;">
      <div style="font-size:0.62rem;letter-spacing:.1em;text-transform:uppercase;color:#64748b;margin-bottom:6px;">Density Ratio</div>
      <div style="font-size:2rem;font-weight:700;color:#f59e0b;">{m['density_ratio']}×</div>
      <div style="font-size:0.65rem;color:#475569;">gen / orig</div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;">
    <div style="background:#14171f;border:1px solid #2a2f42;border-radius:10px;padding:16px;">
      <div style="font-size:0.62rem;letter-spacing:.1em;text-transform:uppercase;color:#5b8cff;margin-bottom:12px;">Similarity Profile</div>
      <div style="position:relative;height:220px;">{radar_svg}</div>
      <div style="display:flex;gap:16px;margin-top:8px;font-size:11px;color:#64748b;">
        <span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:2px;background:#5b8cff;opacity:.6;display:inline-block;"></span>Original</span>
        <span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:2px;background:#a78bfa;display:inline-block;"></span>Generated</span>
      </div>
    </div>

    <div style="background:#14171f;border:1px solid #2a2f42;border-radius:10px;padding:16px;">
      <div style="font-size:0.62rem;letter-spacing:.1em;text-transform:uppercase;color:#5b8cff;margin-bottom:12px;">Texture Comparison</div>
      <div style="position:relative;height:220px;">{bar_svg}</div>
      <div style="display:flex;gap:16px;margin-top:8px;font-size:11px;color:#64748b;">
        <span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:2px;background:#5b8cff;opacity:.6;display:inline-block;"></span>Original</span>
        <span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:2px;background:#a78bfa;display:inline-block;"></span>Generated</span>
      </div>
    </div>
  </div>

  <div style="background:#14171f;border:1px solid #2a2f42;border-radius:10px;padding:16px;overflow-x:auto;">
    <div style="font-size:0.62rem;letter-spacing:.1em;text-transform:uppercase;color:#5b8cff;margin-bottom:12px;">Full Breakdown</div>
    <table style="width:100%;border-collapse:collapse;font-size:0.78rem;color:#cbd5e1;min-width:640px;">
      <thead>
        <tr style="border-bottom:1px solid #2a2f42;">
          <th style="text-align:left;padding:6px 8px;color:#64748b;font-weight:500;">Metric</th>
          <th style="text-align:right;padding:6px 8px;color:#64748b;font-weight:500;">Original</th>
          <th style="text-align:right;padding:6px 8px;color:#64748b;font-weight:500;">Generated</th>
          <th style="text-align:right;padding:6px 8px;color:#64748b;font-weight:500;">Score</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom:1px solid #1c2030;"><td style="padding:7px 8px;">Pitch similarity</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">—</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">—</td><td style="text-align:right;padding:7px 8px;color:#5b8cff;">{m['pitch_sim']} / 100</td></tr>
        <tr style="border-bottom:1px solid #1c2030;"><td style="padding:7px 8px;">KL divergence <span style="color:#475569;font-size:0.7rem;">(lower=better)</span></td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">—</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">—</td><td style="text-align:right;padding:7px 8px;color:#f59e0b;">{m['kl_div']}</td></tr>
        <tr style="border-bottom:1px solid #1c2030;"><td style="padding:7px 8px;">Interval similarity</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">—</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">—</td><td style="text-align:right;padding:7px 8px;color:#a78bfa;">{m['interval_sim']} / 100</td></tr>
        <tr style="border-bottom:1px solid #1c2030;"><td style="padding:7px 8px;">Density ratio</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">{m['density_orig']}</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">{m['density_gen']}</td><td style="text-align:right;padding:7px 8px;color:#f59e0b;">{m['density_ratio']}×</td></tr>
        <tr style="border-bottom:1px solid #1c2030;"><td style="padding:7px 8px;">Rhythmic regularity</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">{m['rhythm_orig']}</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">{m['rhythm_gen']}</td><td style="text-align:right;padding:7px 8px;color:#34d399;">Δ {round(m['rhythm_gen'] - m['rhythm_orig'], 1):+.1f}</td></tr>
        <tr style="border-bottom:1px solid #1c2030;"><td style="padding:7px 8px;">Avg polyphony</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">{m['poly_orig']}</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">{m['poly_gen']}</td><td style="text-align:right;padding:7px 8px;color:#34d399;">Δ {round(m['poly_gen'] - m['poly_orig'], 2):+.2f}</td></tr>
        <tr><td style="padding:7px 8px;">Note count</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">{m['note_count_orig']}</td><td style="text-align:right;padding:7px 8px;color:#94a3b8;">{m['note_count_gen']}</td><td style="text-align:right;padding:7px 8px;color:#34d399;">Δ {m['note_count_gen'] - m['note_count_orig']:+d}</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""
    return html


# ── Backend logic ──────────────────────────────────────────────────────────────


def setup_openmusenet():
    global setup_complete
    try:
        if not os.path.exists("OpenMusenet2"):
            yield "⏳ Cloning OpenMuseNet repository…"
            result = subprocess.run(
                ["git", "clone", "https://github.com/hidude562/OpenMusenet2.git"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                yield f"❌ Clone failed: {result.stderr}"
                return

        openmusenet_path = os.path.abspath("OpenMusenet2/OpenMusenet3/src/OpenMusenet3")
        if openmusenet_path not in sys.path:
            sys.path.insert(0, openmusenet_path)

        setup_complete = True
        yield "✅ Repository ready."
    except Exception as e:
        setup_complete = False
        yield f"❌ Setup failed: {str(e)}"



def load_model_and_setup():
    global ai_model, MIDI_class, setup_complete, DEVICE

    if not setup_complete:
        for msg in setup_openmusenet():
            yield msg
        if not setup_complete:
            return

    try:
        import torch

        def log(msg):
            print(msg)
            return msg

        yield log("🔍 Detecting compute device…")
        if torch.cuda.is_available():
            DEVICE = "cuda"
            yield log("🚀 GPU detected — using CUDA.")
        elif torch.backends.mps.is_available():
            DEVICE = "mps"
            yield log("🍎 Apple Silicon detected — using MPS.")
        else:
            DEVICE = "cpu"
            yield log("⚠️  No GPU found — using CPU.")

        yield log("📦 Importing OpenMuseNet modules…")
        from ai import AI
        from midi import MIDI
        MIDI_class = MIDI

        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        cached_model_path = os.path.join(MODEL_CACHE_DIR, "maestro-4-genre")

        os.environ["HF_HOME"] = os.path.abspath(MODEL_CACHE_DIR)
        os.environ["TRANSFORMERS_CACHE"] = os.path.abspath(MODEL_CACHE_DIR)

        if os.path.exists(cached_model_path):
            yield log("💾 Found local cache — loading from disk…")
            ai_model = AI(cached_model_path)
            source = "local cache"
        else:
            yield log("⬇️  Downloading model from Hugging Face (first time only)…")
            ai_model = AI("kobimusic/maestro-4-genre")
            yield log("💾 Saving model to local cache for next time…")
            try:
                ai_model.model.save_pretrained(cached_model_path)
                if hasattr(ai_model, "tokenizer"):
                    ai_model.tokenizer.save_pretrained(cached_model_path)
            except Exception as e:
                yield log(f"⚠️  Cache save failed (will re-download next time): {e}")
            source = "Hugging Face"

        yield log(f"📲 Moving model to {DEVICE.upper()}…")
        ai_model.model = ai_model.model.to(DEVICE)

        if DEVICE == "cuda":
            yield log("⚡ Converting model to float16 for faster inference…")
            ai_model.model = ai_model.model.half()

        actual_device = next(ai_model.model.parameters()).device
        msg = f"✅ Model ready!  Source: {source}  •  Device: {actual_device}"
        if DEVICE == "cuda":
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            msg += f"  •  VRAM: {vram_used:.1f} GB / {vram_total:.1f} GB"
        yield log(msg)

    except Exception as e:
        ai_model = None
        msg = f"❌ Failed: {str(e)}\n{traceback.format_exc()}"
        print(msg)
        yield msg



def audio_to_midi_conversion(audio_path):
    try:
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        script = """
import os, sys
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
audio_file, output_dir = sys.argv[1], sys.argv[2]
os.makedirs(output_dir, exist_ok=True)
predict_and_save([audio_file], output_dir,
    sonify_midi=False, save_midi=True,
    save_model_outputs=False, save_notes=False,
    model_or_model_path=ICASSP_2022_MODEL_PATH)
"""
        script_path = os.path.join(temp_dir, "convert.py")
        with open(script_path, "w") as f:
            f.write(script)

        result = subprocess.run(
            ["python", script_path, audio_path, output_dir],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return None, f"❌ Basic Pitch failed: {result.stderr}"

        midi_files = list(Path(output_dir).glob("*.mid"))
        if not midi_files:
            return None, "❌ No MIDI file was produced."

        out = os.path.join(tempfile.gettempdir(), "converted_audio.mid")
        shutil.copy(str(midi_files[0]), out)
        return out, "✅ Audio → MIDI conversion complete."
    except subprocess.TimeoutExpired:
        return None, "❌ Conversion timed out."
    except Exception as e:
        return None, f"❌ Conversion error: {e}"



def generate_music(input_file, num_iterations, genre, temperature, progress=gr.Progress(track_tqdm=False)):
    global ai_model, MIDI_class

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)
        return "\n".join(log_lines)

    def tick(msg):
        return None, log(msg), None, ""

    if ai_model is None:
        yield None, log("❌ Model not loaded — click 'Load Model' first."), None, ""
        return

    if input_file is None:
        yield None, log("❌ No file uploaded."), None, ""
        return

    file_path = input_file.name if hasattr(input_file, "name") else str(input_file)
    ext = Path(file_path).suffix.lower()
    MIDI_EXTS = {".mid", ".midi"}
    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a"}

    progress(0, desc="Starting…")

    if ext in MIDI_EXTS:
        original_midi = file_path
        yield *tick("🎵 Detected MIDI file — skipping transcription."),
    elif ext in AUDIO_EXTS:
        progress(0, desc="Transcribing audio to MIDI…")
        yield *tick(f"🎙️  Detected audio file ({ext}) — transcribing to MIDI via Basic Pitch…"),
        original_midi, status = audio_to_midi_conversion(file_path)
        yield *tick(status),
        if original_midi is None:
            return
    else:
        supported = ", ".join(sorted(MIDI_EXTS | AUDIO_EXTS))
        yield None, log(
            f"❌ Unsupported file type '{ext}'.\n   Supported formats: {supported}"
        ), None, ""
        return

    last_file = original_midi

    for iteration in range(num_iterations):
        pct = iteration / num_iterations
        progress(pct, desc=f"Iteration {iteration + 1} of {num_iterations}…")
        yield *tick(f"🎹 Iteration {iteration + 1} / {num_iterations} — generating…"),

        try:
            midi_obj = MIDI_class(last_file)
            prompt = f"{genre} | . start"

            import torch
            if DEVICE == "cuda" and ai_model.model.dtype == torch.float16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    generations = ai_model.continueMusic(prompt, midi_obj, temperature=temperature)
            else:
                generations = ai_model.continueMusic(prompt, midi_obj, temperature=temperature)

            for gen_idx, generation in enumerate(generations):
                for track in generation.tracks:
                    for msg in track:
                        if msg.time < 0:
                            msg.time = 0

                from mido import MidiFile, MidiTrack, merge_tracks
                mf = MidiFile(type=0)
                merged = MidiTrack()
                merged.extend(merge_tracks(generation.tracks))
                mf.tracks.append(merged)

                output_path = os.path.join(
                    tempfile.gettempdir(),
                    f"generated_iter{iteration}_{gen_idx}.mid",
                )
                mf.save(output_path)
                last_file = output_path

            yield *tick(f"   ✔ Iteration {iteration + 1} done."),

        except Exception as e:
            yield None, log(f"❌ Iteration {iteration + 1} failed: {e}"), None, ""
            return

    progress(0.90, desc="Rendering piano roll…")
    yield *tick("🎨 Rendering piano roll…"),

    piano_img = None
    try:
        piano_img = render_piano_roll(original_midi, last_file)
        yield *tick("   ✔ Piano roll ready."),
    except Exception as e:
        yield *tick(f"⚠️  Piano roll failed (MIDI still saved): {e}"),

    progress(0.95, desc="Computing coherence metrics…")
    yield *tick("📊 Computing coherence metrics…"),

    metrics_html = ""
    try:
        orig_notes = midi_to_notes(original_midi)
        gen_notes = midi_to_notes(last_file)
        metrics = compute_all_metrics(orig_notes, gen_notes)
        metrics_html = build_metrics_html(metrics)

        summary = (
            f"\n─── Coherence Metrics ──────────────────\n"
            f"  Overall Score       : {metrics['overall']} / 100\n"
            f"  Pitch Similarity    : {metrics['pitch_sim']} / 100\n"
            f"  Interval Similarity : {metrics['interval_sim']} / 100\n"
            f"  KL Divergence       : {metrics['kl_div']}  (lower=better)\n"
            f"  Density Ratio       : {metrics['density_ratio']}×\n"
            f"  Rhythm (orig/gen)   : {metrics['rhythm_orig']} / {metrics['rhythm_gen']}\n"
            f"  Polyphony (orig/gen): {metrics['poly_orig']} / {metrics['poly_gen']} voices\n"
            f"  Note count (orig/gen): {metrics['note_count_orig']} / {metrics['note_count_gen']}\n"
            f"────────────────────────────────────────"
        )
        log_lines.append(summary)
        print(summary)
        yield *tick("   ✔ Metrics ready."),
    except Exception as e:
        yield *tick(f"⚠️  Metrics failed: {e}"),

    progress(1.0, desc="Done!")
    yield last_file, "\n".join(log_lines) + f"\n\n✅ All done! {num_iterations} iteration(s) complete.", piano_img, metrics_html



def render_piano_roll(original_midi_path, generated_midi_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image

    orig_notes = midi_to_notes(original_midi_path)
    gen_notes = midi_to_notes(generated_midi_path)

    orig_duration = max((end for _, _, end in orig_notes), default=0)
    gen_duration = max((end for _, _, end in gen_notes), default=0)
    shifted_gen = [(p, s + orig_duration, e + orig_duration) for p, s, e in gen_notes]

    all_notes = orig_notes + shifted_gen
    if not all_notes:
        return None

    pitches = [p for p, _, _ in all_notes]
    pitch_min = max(0, min(pitches) - 2)
    pitch_max = min(127, max(pitches) + 2)
    total_dur = orig_duration + gen_duration

    fig_w = max(14, total_dur * 0.6)
    fig_h = max(5, (pitch_max - pitch_min) * 0.18)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bg_color = "#0d0f14"
    orig_color = "#5b8cff"
    gen_color = "#a78bfa"
    divider_color = "#34d399"
    grid_color = "#1c2030"
    text_color = "#94a3b8"

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    for octave in range(pitch_min // 12, pitch_max // 12 + 1):
        y = octave * 12
        if pitch_min <= y <= pitch_max:
            ax.axhline(y, color=grid_color, linewidth=0.6, zorder=1)

    note_h = 0.7
    for pitch, start, end in orig_notes:
        rect = patches.FancyBboxPatch(
            (start, pitch - note_h / 2), max(end - start, 0.02), note_h,
            boxstyle="round,pad=0.01",
            facecolor=orig_color, edgecolor="none", alpha=0.85, zorder=2,
        )
        ax.add_patch(rect)

    for pitch, start, end in shifted_gen:
        rect = patches.FancyBboxPatch(
            (start, pitch - note_h / 2), max(end - start, 0.02), note_h,
            boxstyle="round,pad=0.01",
            facecolor=gen_color, edgecolor="none", alpha=0.85, zorder=2,
        )
        ax.add_patch(rect)

    ax.axvline(orig_duration, color=divider_color, linewidth=1.8, linestyle="--", alpha=0.9, zorder=3)
    ax.text(orig_duration + total_dur * 0.005, pitch_max - 0.5,
            "generated →", color=divider_color, fontsize=8, va="top", ha="left",
            style="italic", fontfamily="monospace")
    ax.text(orig_duration - total_dur * 0.005, pitch_max - 0.5,
            "← original", color=orig_color, fontsize=8, va="top", ha="right",
            style="italic", fontfamily="monospace")

    ax.set_xlim(0, total_dur)
    ax.set_ylim(pitch_min, pitch_max)
    ax.set_xlabel("Time (seconds)", color=text_color, fontsize=9)
    ax.set_ylabel("MIDI Note", color=text_color, fontsize=9)
    ax.tick_params(colors=text_color, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2f42")

    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    yticks = [n for n in range(pitch_min, pitch_max + 1) if note_names[n % 12] == "C"]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"C{n // 12 - 1}" for n in yticks], color=text_color, fontsize=8)

    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #14171f;
    --surface2: #1c2030;
    --border: #2a2f42;
    --accent: #5b8cff;
    --accent2: #a78bfa;
    --green: #34d399;
    --text: #e2e8f0;
    --muted: #64748b;
    --radius: 12px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

.contain, .gap, .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#header {
    text-align: center;
    padding: 48px 0 32px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
#header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
}
#header p { color: var(--muted); font-size: 0.95rem; margin: 0; }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin-bottom: 20px;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 16px;
}

button.primary, .gr-button-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    color: white !important;
    padding: 12px 24px !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
button.primary:hover, .gr-button-primary:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

textarea, input[type="text"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
}

label span, .gr-block label {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

#metrics-panel { min-height: 60px; }
#footer {
    text-align: center;
    padding: 24px 0 40px;
    color: var(--muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 32px;
}
"""


with gr.Blocks(title="MIDI Music Generator", css=CSS, theme=gr.themes.Base()) as app:
    gr.HTML("""
    <div id="header">
        <h1>🎵 MIDI Music Generator</h1>
        <p>AI-powered music continuation using OpenMuseNet 3 &nbsp;·&nbsp; <code>kobimusic/maestro-4-genre</code></p>
    </div>
    """)

    gr.HTML('<div class="card"><div class="card-title">① Load Model</div>')
    gr.Markdown(
        "Downloads once, then loads instantly from `./model_cache/`. "
        "GPU / Apple Silicon is used automatically if available."
    )
    with gr.Row():
        setup_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
        setup_log = gr.Textbox(
            label="Model Log", value="", interactive=False, lines=5,
            placeholder="Step-by-step progress will appear here…", scale=3,
        )
    gr.HTML("</div>")

    gr.HTML('<div class="card"><div class="card-title">② Generate Music</div>')
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.Markdown("*Accepts audio (WAV, MP3, FLAC, OGG, AIFF, M4A) or MIDI — type is detected automatically.*")
            input_file = gr.File(
                label="Upload File",
                file_types=[".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a", ".mid", ".midi"],
            )
            num_iterations = gr.Slider(
                minimum=1, maximum=10, value=1, step=1,
                label="Iterations",
                info="Each iteration appends one segment of generated music.",
            )
            genre = gr.Dropdown(
                choices=["classical", "nan", "rock", "metal", "pop", "jazz", "romantic"],
                value="pop", label="Genre", info="Steers the model's style.",
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.82, step=0.01,
                label="Temperature",
                info="Low = conservative  ·  High = experimental",
            )
            generate_btn = gr.Button("🎹 Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            gen_log = gr.Textbox(
                label="Generation Log", interactive=False, lines=8,
                placeholder="Live progress will stream here once generation starts…",
            )
            output_midi = gr.File(
                label="⬇ Download Generated MIDI", interactive=False,
            )
    gr.HTML("</div>")

    gr.HTML('<div class="card"><div class="card-title">③ Piano Roll</div>')
    gr.Markdown(
        "Rendered after generation completes. "
        "<span style='color:#5b8cff'>■ Blue</span> = original &nbsp;·&nbsp; "
        "<span style='color:#a78bfa'>■ Purple</span> = generated &nbsp;·&nbsp; "
        "<span style='color:#34d399'>⸺</span> = boundary"
    )
    piano_roll_img = gr.Image(
        label="Piano Roll", interactive=False, show_download_button=True,
    )
    gr.HTML("</div>")

    gr.HTML('<div class="card"><div class="card-title">④ Coherence Metrics</div>')
    gr.Markdown(
        "Automatically computed after generation. "
        "Compares the generated MIDI against your original across pitch, "
        "rhythm, texture, and melodic interval dimensions."
    )
    metrics_display = gr.HTML(
        value="<div style='color:#475569;font-family:DM Mono,monospace;font-size:0.8rem;padding:16px 0;'>Metrics will appear here after generation completes…</div>",
        elem_id="metrics-panel",
    )
    gr.HTML("</div>")

    gr.HTML("""
    <div id="footer">
        Best results with clean, melodic audio &nbsp;·&nbsp;
        Output is a <code>.mid</code> file — open in GarageBand, Ableton, FL Studio, etc.
    </div>
    """)

    setup_btn.click(
        fn=load_model_and_setup,
        inputs=[],
        outputs=[setup_log],
        api_name=False,
    )

    generate_btn.click(
        fn=generate_music,
        inputs=[input_file, num_iterations, genre, temperature],
        outputs=[output_midi, gen_log, piano_roll_img, metrics_display],
        api_name=False,
    )


if __name__ == "__main__":
    app.launch(show_api=False)
