import sys
import os
import io
import gc
import math
import uuid
import shutil
import datetime
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------
# SHIM DE COMPATIBILIDADE: pkg_resources no Python 3.11+
# ---------------------------------------------------------
try:
    import pkg_resources
except ImportError:
    import types
    import importlib.resources
    import importlib.metadata
    pkg_resources = types.ModuleType("pkg_resources")
    def resource_filename(package_or_requirement, resource_name):
        return str(importlib.resources.files(package_or_requirement).joinpath(resource_name))
    def get_distribution(dist):
        class Dist:
            version = importlib.metadata.version(dist)
        return Dist()
    pkg_resources.resource_filename = resource_filename
    pkg_resources.get_distribution = get_distribution
    pkg_resources.DistributionNotFound = importlib.metadata.PackageNotFoundError
    sys.modules["pkg_resources"] = pkg_resources

import streamlit as st

st.set_page_config(page_title="Partitura.ai", page_icon="🎵", layout="centered")

# ---------------------------------------------------------
# TRADUÇÃO DA INTERFACE E COMPONENTE DE UPLOAD (PT-BR)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span {
        display: none;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div::before {
        content: "Arraste e solte o arquivo de áudio aqui";
        font-weight: 500;
        font-size: 1rem;
        color: #1e293b;
        display: block;
        margin-bottom: 4px;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > small {
        font-size: 0 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > small::after {
        content: "Limite de 200MB por arquivo • MP3, WAV, M4A, FLAC, OGG";
        font-size: 0.85rem;
        color: #64748b;
    }
    [data-testid="stFileUploaderDropzone"] button {
        font-size: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "Selecionar Arquivo";
        font-size: 0.875rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import numpy as np
import librosa
import soundfile as sf
import pretty_midi
import music21
import pypdfium2 as pdfium
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH, FilenameSuffix, build_icassp_2022_model_path

UPLOAD_DIR = Path("temp_uploads")
OUTPUT_DIR = Path("temp_outputs")
for d in [UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

APP_VERSION = "A4.0.1-Hierarchical-Inference"

# ---------------------------------------------------------
# 1. SISTEMA DE AUDITORIA E LOG TÉCNICO
# ---------------------------------------------------------
class TechnicalLogger:
    def __init__(self):
        self.lines = []
        self.warnings = []
        self.errors = []

    def reset(self):
        self.lines = []
        self.warnings = []
        self.errors = []

    def add(self, text: str):
        msg = str(text)
        self.lines.append(msg)
        print(msg)

    def add_warning(self, text: str):
        msg = f"[WARNING] {text}"
        self.warnings.append(msg)
        self.lines.append(msg)
        print(msg)

    def add_error(self, text: str):
        msg = f"[ERROR] {text}"
        self.errors.append(msg)
        self.lines.append(msg)
        print(msg)

    def get_text(self) -> str:
        return "\n".join(self.lines)


# ---------------------------------------------------------
# 2. CONFIGURAÇÕES CENTRALIZADAS EM 7 GRUPOS
# ---------------------------------------------------------
CONFIG_SIGNAL = {
    "sample_rate": 22050,
    "hop_length": 512,
    "chunk_duration_sec": 60.0,
    "chunk_overlap_sec": 1.0,
    "min_chunk_sec": 1.5,
    "dedup_time_tol_sec": 0.045,
    "duration_min_abs_sec": 0.08,
}

CONFIG_TACTUS = {
    "ratios": (0.25, 1.0/3.0, 0.375, 0.5, 2.0/3.0, 0.75, 1.0, 4.0/3.0, 1.5, 2.0, 8.0/3.0, 3.0, 4.0),
    "min_bpm": 42.0,
    "max_bpm": 216.0,
    "perceptual_center_bpm": 100.0,
    "perceptual_sigma_bpm": 38.0,
    "merge_tolerance_pct": 0.035,
    "tatum_subdivisions": 6,
    "segment_positions": (0.15, 0.38, 0.62, 0.85),
    "segment_window_sec": 18.0,
}

CONFIG_METER = {
    "w_periodicity": 0.24,
    "w_accent": 0.24,
    "w_subdivision": 0.18,
    "w_harmonic_change": 0.12,
    "w_recurrence": 0.12,
    "w_bass": 0.10,
    "beat3_continuation_threshold": 0.52,
    "ternary_phase_bins": 12,
    "viterbi_self_transition": 0.82,
    "viterbi_section_min_windows": 2,
}

CONFIG_CONFIDENCE = {
    "bootstrap_iterations": 40,
    "min_decision_quality": 0.46,
    "min_bootstrap_stability": 0.58,
    "min_feature_agreement": 0.42,
    "base_margin_threshold": 0.060,
    "min_margin_abs": 0.025,
    "max_margin_abs": 0.140,
}

CONFIG_MELODY = {
    "velocity_base": 48.0,
    "velocity_min_floor": 32.0,
    "velocity_max_ceil": 65.0,
    "pitch_base_min": 50,
    "pitch_floor_abs": 43,
    "duration_base_sec": 0.095,
    "simultaneous_tolerance_sec": 0.045,
    "jump_penalty_semitone": 0.50,
    "octave_jump_penalty": 4.2,
    "short_gap_sec": 0.12,
    "short_gap_jump_mult": 1.65,
    "phrase_reset_gap_sec": 1.35,
    "phrase_reset_jump_mult": 0.25,
    "continuity_bonus": 2.8,
    "no_melody_threshold": 2.2,
    "no_melody_transition_cost": 1.4,
}

CONFIG_HARMONY = {
    "velocity_base": 40.0,
    "velocity_min_floor": 26.0,
    "peso_base": 0.18,
    "peso_min_floor": 0.07,
    "bass_ceiling_midi": 57,
    "root_min_confidence": 0.36,
    "quality_min_confidence": 0.40,
    "quality_min_margin": 0.045,
    "third_presence_min_ratio": 0.14,
    "hmm_stay_prob": 0.76,
    "tonal_prior_weight": 0.12,
}

CONFIG_QUANTIZATION = {
    "grid_subdivision_ql": 0.25,
    "w_onset_error": 1.0,
    "w_dur_error": 0.85,
    "w_complexity": 0.12,
    "expressible_single_qls": (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
}


# ---------------------------------------------------------
# 3. CARREGAMENTO DO MODELO ONNX E ESTRUTURAS DE NOTA
# ---------------------------------------------------------
@st.cache_resource
def get_inference_model():
    try:
        onnx_model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
        return Model(onnx_model_path)
    except Exception:
        return Model(ICASSP_2022_MODEL_PATH)

class MockNote:
    def __init__(self, start_sec, end_sec, pitch, velocity, source_chunk=0):
        self.start_sec = float(start_sec)
        self.end_sec = float(end_sec)
        self.pitch = int(pitch)
        self.velocity = int(velocity)
        self.source_chunk = int(source_chunk)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


# ---------------------------------------------------------
# 4. FUNÇÕES MATEMÁTICAS DE RELIABILITY E ENTROPIA
# ---------------------------------------------------------
def binary_reliability(p: float) -> float:
    p_clip = float(np.clip(p, 0.0, 1.0))
    return float(np.clip(2.0 * abs(p_clip - 0.5), 0.0, 1.0))

def multiclass_reliability(prob_dict: dict) -> float:
    vals = np.array([max(0.0, float(v)) for v in prob_dict.values()], dtype=float)
    n = len(vals)
    if n <= 1 or np.sum(vals) <= 1e-9:
        return 0.0
    p = vals / np.sum(vals)
    entropy = -np.sum(p * np.log(p + 1e-12))
    norm_ent = entropy / math.log(n)
    s_sorted = np.sort(p)[::-1]
    margin = float(s_sorted[0] - s_sorted[1])
    rel = 0.55 * (1.0 - norm_ent) + 0.45 * min(1.0, margin * 2.5)
    return float(np.clip(rel, 0.0, 1.0))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


# ---------------------------------------------------------
# 5. ENARMONIA, TONALIDADE E FORMATAÇÃO PT-BR
# ---------------------------------------------------------
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
FLAT_KEYS_TONICS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "d", "g", "c", "f", "bb", "eb"}

def prefer_flats_for_key(detected_key) -> bool:
    if detected_key is None:
        return False
    try:
        if getattr(detected_key, "sharps", 0) < 0:
            return True
        t_name = detected_key.tonic.name.replace("-", "b")
        if detected_key.mode == "minor":
            t_name = t_name.lower()
        return t_name in FLAT_KEYS_TONICS
    except Exception:
        return False

def pc_to_note_name(pc: int, use_flats: bool = False) -> str:
    pc_mod = int(pc) % 12
    return FLAT_NAMES[pc_mod] if use_flats else SHARP_NAMES[pc_mod]

def note_name_to_music21(name_str: str) -> str:
    if not name_str:
        return "C"
    clean = name_str.replace("♭", "b").replace("♯", "#")
    if len(clean) > 1 and clean[1] == "b":
        return clean[0] + "-" + clean[2:]
    return clean

def sanitizar_nome_tom(tonic_name: str) -> str:
    if not tonic_name:
        return "Não determinado"
    clean = tonic_name.replace("-", "b")
    enarm_map = {"Fb": "E", "Cb": "B", "E#": "F", "B#": "C"}
    clean = enarm_map.get(clean, clean)
    mapa_pt = {
        "C": "Dó", "C#": "Dó#", "Db": "Ré♭",
        "D": "Ré", "D#": "Ré#", "Eb": "Mi♭",
        "E": "Mi",
        "F": "Fá", "F#": "Fá#", "Gb": "Sol♭",
        "G": "Sol", "G#": "Sol#", "Ab": "Lá♭",
        "A": "Lá", "A#": "Lá#", "Bb": "Si♭",
        "B": "Si"
    }
    return mapa_pt.get(clean, clean)


def detectar_centro_tonal_global(part, y_harm=None, sr=22050, all_notes=None, log: TechnicalLogger = None):
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_profile = major_profile / np.linalg.norm(major_profile)
    minor_profile = minor_profile / np.linalg.norm(minor_profile)

    pc_global = np.zeros(12, dtype=float)
    seg_winners = []

    if y_harm is not None and len(y_harm) > sr * 2:
        try:
            chroma_full = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=CONFIG_SIGNAL["hop_length"])
            c_mean = np.mean(chroma_full, axis=1)
            if np.max(c_mean) > 0:
                pc_global += 0.45 * (c_mean / np.max(c_mean))

            n_frames = chroma_full.shape[1]
            if n_frames >= 30:
                for s_i in range(3):
                    f0 = int(s_i * n_frames / 3)
                    f1 = int((s_i + 1) * n_frames / 3)
                    c_seg = np.mean(chroma_full[:, f0:f1], axis=1)
                    if np.linalg.norm(c_seg) > 1e-6:
                        c_seg_n = c_seg / np.linalg.norm(c_seg)
                        best_k = max(
                            [(float(np.dot(c_seg_n, np.roll(major_profile, k))), k, "major") for k in range(12)] +
                            [(float(np.dot(c_seg_n, np.roll(minor_profile, k))), k, "minor") for k in range(12)],
                            key=lambda x: x[0]
                        )
                        seg_winners.append((best_k[1], best_k[2]))
        except Exception:
            pass

    if all_notes:
        hist_notes = np.zeros(12, dtype=float)
        for n in all_notes:
            hist_notes[n.pitch % 12] += n.duration_sec * (n.velocity / 127.0)
        if np.max(hist_notes) > 0:
            pc_global += 0.35 * (hist_notes / np.max(hist_notes))

    mel_notes = list(part.recurse().notes) if part is not None else []
    if mel_notes:
        hist_mel = np.zeros(12, dtype=float)
        for el in mel_notes:
            for p in el.pitches:
                hist_mel[p.pitchClass] += float(el.quarterLength)
        if np.max(hist_mel) > 0:
            pc_global += 0.20 * (hist_mel / np.max(hist_mel))

    if np.sum(pc_global) <= 1e-6:
        if log:
            log.add_warning("Material insuficiente para análise tonal. Aplicando fallback C major.")
        return music21.key.Key("C"), False, 0.0, "Não determinado", "indeterminado"

    pc_norm = pc_global / np.linalg.norm(pc_global)
    candidates = []
    for pc in range(12):
        corr_maj = float(np.dot(pc_norm, np.roll(major_profile, pc)))
        corr_min = float(np.dot(pc_norm, np.roll(minor_profile, pc)))
        candidates.append((corr_maj, pc, "major"))
        candidates.append((corr_min, pc, "minor"))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_corr, best_pc, best_mode = candidates[0]
    second_corr, sec_pc, sec_mode = candidates[1]
    margin = best_corr - second_corr

    same_tonic_other = [sc for sc, pc_i, md_i in candidates if pc_i == best_pc and md_i != best_mode]
    mode_diff = best_corr - (same_tonic_other[0] if same_tonic_other else 0.0)
    mode_determined = mode_diff >= 0.025

    seg_stability = 0.75
    if seg_winners:
        matches = sum(1 for k_pc, _ in seg_winners if k_pc == best_pc or (k_pc - best_pc) % 12 in (3, 9))
        seg_stability = matches / float(len(seg_winners))

    use_flats = best_pc in ({1, 3, 5, 8, 10} if best_mode == "major" else {2, 3, 5, 7, 10})
    tonic_str = pc_to_note_name(best_pc, use_flats=use_flats)
    m21_tonic = note_name_to_music21(tonic_str)
    tonic_pt = sanitizar_nome_tom(tonic_str)
    mode_pt = ("Maior" if best_mode == "major" else "Menor") if mode_determined else "modo indeterminado"

    conf = float(np.clip(0.45 * max(0.0, (best_corr - 0.5) * 2.0) + 0.30 * min(1.0, margin * 8.0) + 0.25 * seg_stability, 0.0, 1.0))

    if log:
        log.add("\n==============================\nTONALITY & ENHARMONIC ANALYSIS\n==============================")
        for rank_i, (sc, pc_i, md_i) in enumerate(candidates[:5]):
            nm = pc_to_note_name(pc_i, use_flats=(pc_i in {1, 3, 5, 8, 10}))
            log.add(f"  Top {rank_i+1}: {nm} {md_i:5s} = {sc:.4f}")
        log.add(f"Tônica: {tonic_pt} ({tonic_str}) | Modo: {mode_pt} (diff={mode_diff:.4f})")
        log.add(f"Margem Top1-Top2: {margin:.4f} | Estabilidade por segmentos: {seg_stability:.2f} | Key Confidence: {conf:.3f}")

    try:
        key_obj = music21.key.Key(m21_tonic, best_mode)
        if best_corr < 0.56 or conf < 0.32:
            return music21.key.Key("C"), False, conf, "Não determinado", "indeterminado"
        return key_obj, True, conf, tonic_pt, mode_pt
    except Exception:
        return music21.key.Key("C"), False, 0.0, "Não determinado", "indeterminado"


# ---------------------------------------------------------
# 6. FONTE ÚNICA DE TEMPO, TATUM GRID E FAMÍLIA DE TACTUS
# ---------------------------------------------------------
def analyze_unified_tempo_and_tatums(y_full, sr, log: TechnicalLogger):
    hop = CONFIG_SIGNAL["hop_length"]
    total_dur = len(y_full) / float(sr)

    try:
        y_harm, y_perc = librosa.effects.hpss(y_full)
    except Exception:
        y_harm, y_perc = y_full, y_full

    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop)
    onset_max = float(np.max(onset_env)) if len(onset_env) > 0 else 0.0
    onset_norm = onset_env / (onset_max + 1e-8) if onset_max > 0 else onset_env

    rms = float(np.sqrt(np.mean(y_full ** 2)))
    onset_kurt = float(np.percentile(onset_norm, 95) / (np.mean(onset_norm) + 1e-8)) if len(onset_norm) > 0 else 0.0
    signal_quality = float(np.clip(0.5 * min(1.0, rms * 12.0) + 0.5 * min(1.0, onset_kurt / 4.5), 0.1, 1.0))

    frames_45 = min(len(onset_norm), int(45.0 * sr / hop))
    t_glob, b_glob = librosa.beat.beat_track(onset_envelope=onset_norm[:frames_45], sr=sr, hop_length=hop)
    t_arr = np.asarray(t_glob).reshape(-1)
    global_raw_bpm = float(t_arr[0]) if (t_arr.size > 0 and math.isfinite(float(t_arr[0])) and float(t_arr[0]) > 0) else 120.0

    win_sec = CONFIG_TACTUS["segment_window_sec"]
    win_frames = int(win_sec * sr / hop)
    segments = []

    for pos in CONFIG_TACTUS["segment_positions"]:
        c_frame = int(pos * len(onset_norm))
        f0 = max(0, c_frame - win_frames // 2)
        f1 = min(len(onset_norm), f0 + win_frames)
        if f1 - f0 < int(4.0 * sr / hop):
            continue

        env_seg = onset_norm[f0:f1]
        t_seg, b_seg = librosa.beat.beat_track(onset_envelope=env_seg, sr=sr, hop_length=hop)
        ts_arr = np.asarray(t_seg).reshape(-1)
        seg_bpm = float(ts_arr[0]) if (ts_arr.size > 0 and math.isfinite(float(ts_arr[0])) and float(ts_arr[0]) > 0) else global_raw_bpm
        b_times_seg = librosa.frames_to_time(b_seg + f0, sr=sr, hop_length=hop)

        segments.append({
            "position": pos,
            "pos_label": f"{int(pos*100)}%",
            "frame_start": f0,
            "frame_end": f1,
            "time_start": f0 * hop / float(sr),
            "time_end": f1 * hop / float(sr),
            "bpm": round(seg_bpm, 2),
            "beat_frames_local": b_seg,
            "beat_times": b_times_seg,
            "onset_env": env_seg,
        })

    raw_pool = [global_raw_bpm] + [s["bpm"] for s in segments]
    candidate_pool = []
    for base_b in raw_pool:
        for r in CONFIG_TACTUS["ratios"]:
            cand = base_b * r
            if CONFIG_TACTUS["min_bpm"] <= cand <= CONFIG_TACTUS["max_bpm"]:
                candidate_pool.append(cand)

    if not candidate_pool:
        candidate_pool = [120.0]

    candidate_pool.sort()
    merged_groups = []
    tol_pct = CONFIG_TACTUS["merge_tolerance_pct"]
    for c in candidate_pool:
        if not merged_groups or abs(c - np.mean(merged_groups[-1])) / np.mean(merged_groups[-1]) > tol_pct:
            merged_groups.append([c])
        else:
            merged_groups[-1].append(c)
    unique_cands = [round(float(np.mean(g)), 2) for g in merged_groups]

    tg_full = librosa.feature.tempogram(onset_envelope=onset_norm[:frames_45], sr=sr, hop_length=hop)
    ac_mean = np.mean(tg_full, axis=1)
    ac_mean = ac_mean / (np.max(ac_mean) + 1e-8)

    tactus_family = []
    for cand in unique_cands:
        lag = int(round((60.0 / cand * sr) / hop))
        w0, w1 = max(1, lag - 1), min(len(ac_mean), lag + 2)
        ac_sc = float(np.max(ac_mean[w0:w1])) if w1 > w0 else 0.0

        seg_align = []
        for s in segments:
            errs = [abs(cand - s["bpm"] * r) / cand for r in (0.5, 2.0/3.0, 1.0, 1.5, 2.0)]
            seg_align.append(max(0.0, 1.0 - min(errs) * 4.5))
        align_sc = float(np.mean(seg_align)) if seg_align else 0.5

        perc_prior = math.exp(-0.5 * ((cand - CONFIG_TACTUS["perceptual_center_bpm"]) / CONFIG_TACTUS["perceptual_sigma_bpm"]) ** 2)
        fam_score = 0.46 * ac_sc + 0.32 * align_sc + 0.22 * perc_prior
        tactus_family.append({
            "bpm": cand,
            "score": round(fam_score, 4),
            "ac_score": round(ac_sc, 4),
            "align_score": round(align_sc, 4),
            "perceptual_prior": round(perc_prior, 4),
        })

    tactus_family.sort(key=lambda x: x["score"], reverse=True)
    top_tactus_candidates = tactus_family[:5]

    onset_peaks_frames = librosa.onset.onset_detect(onset_envelope=onset_norm, sr=sr, hop_length=hop, backtrack=False)
    onset_peaks_times = librosa.frames_to_time(onset_peaks_frames, sr=sr, hop_length=hop)

    log.add("\n==============================\nUNIFIED TEMPO & TACTUS FAMILY\n==============================")
    log.add(f"Signal Quality: {signal_quality:.3f} | BPM bruto (45s): {global_raw_bpm:.2f}")
    for s in segments:
        diff_p = ((s["bpm"] - global_raw_bpm) / (global_raw_bpm + 1e-8)) * 100.0
        log.add(f"  Segmento {s['pos_label']} ({s['time_start']:.1f}s-{s['time_end']:.1f}s): BPM={s['bpm']:.2f} ({diff_p:+.1f}%), beats={len(s['beat_times'])}")
    log.add("Top Candidatos da Família de Tactus (avaliados em conjunto com o Compasso):")
    for tf in top_tactus_candidates:
        log.add(f"  {tf['bpm']:6.2f} BPM -> Score={tf['score']:.3f} (AC={tf['ac_score']:.3f}, Align={tf['align_score']:.3f}, Prior={tf['perceptual_prior']:.3f})")

    return {
        "y_harm": y_harm,
        "y_perc": y_perc,
        "onset_norm": onset_norm,
        "onset_peaks_times": onset_peaks_times,
        "global_raw_bpm": round(global_raw_bpm, 2),
        "segments": segments,
        "tactus_family": tactus_family,
        "top_tactus": top_tactus_candidates,
        "signal_quality": signal_quality,
        "total_duration": total_dur,
    }


def build_beat_and_tatum_grid(onset_norm, sr, chosen_bpm, total_duration, log: TechnicalLogger):
    hop = CONFIG_SIGNAL["hop_length"]
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_norm, sr=sr, hop_length=hop, bpm=chosen_bpm)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    sec_per_beat = 60.0 / chosen_bpm

    if len(beat_times) < 4:
        beat_times = np.arange(0.0, total_duration + sec_per_beat, sec_per_beat)
        beat_quality = 0.45
        med_ibi = sec_per_beat
        std_ibi = 0.0
    else:
        ibis = np.diff(beat_times)
        med_ibi = float(np.median(ibis))
        std_ibi = float(np.std(ibis))
        cv = std_ibi / (med_ibi + 1e-8)
        beat_quality = float(np.clip(1.0 - cv * 2.4, 0.1, 1.0))

        first_b = float(beat_times[0])
        pre = []
        c_t = first_b - med_ibi
        while c_t >= 0.0:
            pre.append(c_t)
            c_t -= med_ibi
        pre.reverse()

        last_b = float(beat_times[-1])
        post = []
        c_t = last_b + med_ibi
        while c_t <= total_duration + med_ibi:
            post.append(c_t)
            c_t += med_ibi

        beat_times = np.concatenate([np.array(pre), beat_times, np.array(post)])

    n_sub = CONFIG_TACTUS["tatum_subdivisions"]
    tatum_times = []
    for i in range(len(beat_times) - 1):
        b0, b1 = beat_times[i], beat_times[i + 1]
        for k in range(n_sub):
            tatum_times.append(b0 + (b1 - b0) * (k / float(n_sub)))
    tatum_times.append(float(beat_times[-1]))
    tatum_times = np.array(tatum_times, dtype=float)

    log.add("\n==============================\nBEAT & TATUM GRID AUDIT\n==============================")
    log.add(f"Tactus Efetivo da Grade: {chosen_bpm:.1f} BPM | Beats={len(beat_times)} | Tatums={len(tatum_times)}")
    log.add(f"Mediana IBI: {med_ibi:.4f}s | Desvio IBI: {std_ibi:.4f}s | Beat Quality: {beat_quality:.3f}")
    log.add(f"Primeiros 15 beats (s): {[round(float(x), 3) for x in beat_times[:15]]}")

    return beat_times, tatum_times, beat_quality


def seconds_to_musical_position(time_sec: float, beat_times: np.ndarray, pulse_ql: float = 1.0) -> float:
    if beat_times is None or len(beat_times) < 2:
        return max(0.0, float(time_sec))
    idx_float = float(np.interp(time_sec, beat_times, np.arange(len(beat_times), dtype=float)))
    return max(0.0, idx_float * pulse_ql)


# ---------------------------------------------------------
# 7. MOTOR DE COMPASSO HIERÁRQUICO, VITERBI E BOOTSTRAP
# ---------------------------------------------------------
def _compute_inner_pulse_subdivision(onset_peaks_times, beat_times_seg):
    n_bins = CONFIG_METER["ternary_phase_bins"]
    tpl_bin = np.zeros(n_bins, dtype=float)
    tpl_tern = np.zeros(n_bins, dtype=float)

    for idx_b in [0, n_bins // 2]:
        tpl_bin[idx_b] = 1.0
        tpl_bin[(idx_b - 1) % n_bins] = 0.25
        tpl_bin[(idx_b + 1) % n_bins] = 0.25

    for idx_t in [0, n_bins // 3, (2 * n_bins) // 3]:
        tpl_tern[idx_t] = 1.0
        tpl_tern[(idx_t - 1) % n_bins] = 0.20
        tpl_tern[(idx_t + 1) % n_bins] = 0.20

    if len(beat_times_seg) < 4 or len(onset_peaks_times) < 4:
        return 0.52, 0.48, 0.0, 0.0

    hist_global = np.zeros(n_bins, dtype=float)
    bar_ternary_wins = 0
    valid_pulses = 0

    for i in range(len(beat_times_seg) - 1):
        b0, b1 = beat_times_seg[i], beat_times_seg[i + 1]
        dur = b1 - b0
        if dur <= 0.12:
            continue
        p_in = onset_peaks_times[(onset_peaks_times >= b0) & (onset_peaks_times < b1)]
        if len(p_in) < 2:
            continue
        phases = (p_in - b0) / dur
        h_pulse, _ = np.histogram(phases, bins=n_bins, range=(0.0, 1.0))
        hist_global += h_pulse
        valid_pulses += 1
        if cosine_sim(h_pulse.astype(float), tpl_tern) > cosine_sim(h_pulse.astype(float), tpl_bin) + 0.04:
            bar_ternary_wins += 1

    if np.sum(hist_global) <= 0:
        return 0.52, 0.48, 0.0, 0.0

    sim_bin = cosine_sim(hist_global, tpl_bin)
    sim_tern = cosine_sim(hist_global, tpl_tern)

    inner_tern_support = min(hist_global[n_bins // 3], hist_global[(2 * n_bins) // 3]) / (np.max(hist_global) + 1e-8)
    rep_ratio = bar_ternary_wins / float(max(1, valid_pulses))

    if inner_tern_support < 0.15 or rep_ratio < 0.28:
        sim_tern *= 0.78

    tot = sim_bin + sim_tern + 1e-8
    p_bin = sim_bin / tot
    p_tern = sim_tern / tot
    rel = binary_reliability(p_bin)
    return float(p_bin), float(p_tern), float(rel), float(rep_ratio)


def _evaluate_segment_for_tactus(seg, y_harm_seg, y_perc_seg, sr, cand_bpm, onset_peaks_times, seg_notes):
    hop = CONFIG_SIGNAL["hop_length"]
    env = seg["onset_env"]
    if len(env) < 16:
        return None

    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, bpm=cand_bpm)
    if len(beats) < 6:
        step_f = max(2, int(round((60.0 / cand_bpm * sr) / hop)))
        beats = np.arange(0, len(env), step_f)
    if len(beats) < 6:
        return None

    beat_times_local = librosa.frames_to_time(beats + seg["frame_start"], sr=sr, hop_length=hop)
    frames_per_beat = max(2, int(round((60.0 / cand_bpm * sr) / hop)))

    tg = librosa.feature.tempogram(onset_envelope=env, sr=sr, hop_length=hop)
    ac = np.mean(tg, axis=1)
    ac = ac / (np.max(ac) + 1e-8)

    def _lag_e(mult):
        lg = int(round(frames_per_beat * mult))
        if 1 <= lg < len(ac):
            return float(np.max(ac[max(1, lg - 1):min(len(ac), lg + 2)]))
        return 0.0

    per_raw = {2: _lag_e(2.0), 3: _lag_e(3.0), 4: _lag_e(4.0)}
    per_sum = sum(per_raw.values()) + 1e-8
    per_probs = {k: v / per_sum for k, v in per_raw.items()}
    per_rel = multiclass_reliability(per_probs)

    b_acc = np.array([
        float(np.max(env[max(0, int(b) - 2):min(len(env), int(b) + 3)]))
        for b in beats
    ], dtype=float)
    b_z = (b_acc - np.mean(b_acc)) / (np.std(b_acc) + 1e-8)

    try:
        chroma = librosa.feature.chroma_cqt(y=y_harm_seg, sr=sr, hop_length=hop)
        c_sync = librosa.util.sync(chroma, beats)
        c_norm = c_sync / (np.linalg.norm(c_sync, axis=0, keepdims=True) + 1e-8)
        h_diff = np.sqrt(np.sum(np.diff(c_norm, axis=1) ** 2, axis=0))
        h_diff = np.pad(h_diff, (1, 0), mode="edge")
        h_diff = h_diff / (np.max(h_diff) + 1e-8)
    except Exception:
        c_norm = np.zeros((12, len(beats)))
        h_diff = np.zeros(len(beats))

    bass_beat_env = np.zeros(len(beat_times_local), dtype=float)
    if seg_notes:
        for bi, bt in enumerate(beat_times_local):
            low_n = [n for n in seg_notes if n.pitch <= 55 and abs(n.start_sec - bt) <= 0.14]
            if low_n:
                bass_beat_env[bi] = max(n.velocity / 127.0 for n in low_n)

    templates_acc = {
        2: np.array([1.0, -0.85]),
        3: np.array([1.0, -0.50, -0.50]),
        4: np.array([1.0, -0.60, 0.35, -0.60]),
    }

    acc_probs = {}
    harm_probs = {}
    bass_probs = {}
    best_phases = {}

    for p_n, tpl in templates_acc.items():
        best_f = 0
        best_comb = -1e9
        best_a = 0.0
        best_h = 0.0
        best_b = 0.0
        for f in range(p_n):
            idx_seq = np.arange(len(beats))
            roles = (idx_seq - f) % p_n
            a_val = float(np.mean(b_z * tpl[roles]))
            down_i = idx_seq[roles == 0]
            h_val = float(np.mean(h_diff[down_i])) if len(down_i) > 0 else 0.0
            b_val = float(np.mean(bass_beat_env[down_i])) if len(down_i) > 0 else 0.0
            comb = 0.55 * a_val + 0.25 * h_val + 0.20 * b_val
            if comb > best_comb:
                best_comb = comb
                best_f = f
                best_a = a_val
                best_h = h_val
                best_b = b_val
        acc_probs[p_n] = float(1.0 / (1.0 + math.exp(-2.2 * best_a)))
        harm_probs[p_n] = float(best_h)
        bass_probs[p_n] = float(best_b)
        best_phases[p_n] = int(best_f)

    def _norm_dict(d):
        s = sum(d.values()) + 1e-8
        return {k: v / s for k, v in d.items()}

    acc_probs = _norm_dict(acc_probs)
    harm_probs = _norm_dict(harm_probs)
    bass_probs = _norm_dict(bass_probs)

    acc_rel = multiclass_reliability(acc_probs)
    harm_rel = multiclass_reliability(harm_probs)
    bass_rel = multiclass_reliability(bass_probs)

    rec_raw = {2: 0.33, 3: 0.33, 4: 0.33}
    if c_norm.shape[1] >= 10:
        sim_m = np.dot(c_norm.T, c_norm)
        for lg in (2, 3, 4):
            diags = [sim_m[i, i + lg] for i in range(sim_m.shape[0] - lg)]
            rec_raw[lg] = float(np.clip(np.mean(diags), 0.01, 1.0)) if diags else 0.33
    rec_probs = _norm_dict(rec_raw)
    rec_rel = multiclass_reliability(rec_probs)

    f4 = best_phases[4]
    idx_all = np.arange(len(beats))
    b1_idx = idx_all[(idx_all - f4) % 4 == 0]
    b3_idx = idx_all[(idx_all - f4) % 4 == 2]

    beat3_is_continuation = 0.50
    if len(b1_idx) >= 2 and len(b3_idx) >= 2:
        h_b1 = float(np.mean(h_diff[b1_idx]))
        h_b3 = float(np.mean(h_diff[b3_idx]))
        a_b1 = float(np.mean(b_acc[b1_idx]))
        a_b3 = float(np.mean(b_acc[b3_idx]))
        bs_b1 = float(np.mean(bass_beat_env[b1_idx]))
        bs_b3 = float(np.mean(bass_beat_env[b3_idx]))

        asym_h = (h_b1 - h_b3) / (h_b1 + h_b3 + 1e-8)
        asym_a = (a_b1 - a_b3) / (a_b1 + a_b3 + 1e-8)
        asym_b = (bs_b1 - bs_b3) / (bs_b1 + bs_b3 + 1e-8)
        asym_total = 0.45 * asym_h + 0.35 * asym_a + 0.20 * asym_b
        beat3_is_continuation = float(np.clip(0.50 + 1.25 * asym_total, 0.05, 0.95))

    b3_rel = binary_reliability(beat3_is_continuation)

    p_sub_bin, p_sub_tern, sub_rel, tern_rep = _compute_inner_pulse_subdivision(onset_peaks_times, beat_times_local)

    wm = CONFIG_METER
    eff_w_per = wm["w_periodicity"] * (0.25 + 0.75 * per_rel)
    eff_w_acc = wm["w_accent"] * (0.25 + 0.75 * acc_rel)
    eff_w_harm = wm["w_harmonic_change"] * (0.20 + 0.80 * harm_rel)
    eff_w_rec = wm["w_recurrence"] * (0.20 + 0.80 * rec_rel)
    eff_w_bass = wm["w_bass"] * (0.20 + 0.80 * bass_rel)
    w_sum_grp = eff_w_per + eff_w_acc + eff_w_harm + eff_w_rec + eff_w_bass + 1e-8

    grp_scores = {}
    for g in (2, 3, 4):
        grp_scores[g] = (
            eff_w_per * per_probs[g] +
            eff_w_acc * acc_probs[g] +
            eff_w_harm * harm_probs[g] +
            eff_w_rec * rec_probs[g] +
            eff_w_bass * bass_probs[g]
        ) / w_sum_grp

    if beat3_is_continuation >= wm["beat3_continuation_threshold"]:
        transfer = 0.22 * b3_rel * grp_scores[2]
        grp_scores[4] += transfer
        grp_scores[2] -= transfer
    else:
        transfer = 0.22 * b3_rel * grp_scores[4]
        grp_scores[2] += transfer
        grp_scores[4] -= transfer

    grp_scores = _norm_dict(grp_scores)
    grouping_quality = multiclass_reliability(grp_scores)

    top_g = max(grp_scores.items(), key=lambda x: x[1])[0]
    feat_votes = [
        max(per_probs.items(), key=lambda x: x[1])[0] == top_g,
        max(acc_probs.items(), key=lambda x: x[1])[0] == top_g,
        max(harm_probs.items(), key=lambda x: x[1])[0] == top_g,
        max(rec_probs.items(), key=lambda x: x[1])[0] == top_g,
    ]
    feature_agreement = float(np.mean(feat_votes))

    eff_sub_tern = 0.50 * (1.0 - sub_rel) + p_sub_tern * sub_rel
    eff_sub_bin = 1.0 - eff_sub_tern

    sc_2_4 = grp_scores[2] * (0.55 + 0.45 * eff_sub_bin)
    sc_6_8 = grp_scores[2] * (0.35 + 0.65 * eff_sub_tern) * (1.15 if tern_rep >= 0.35 else 0.72)

    slow_pulse_factor = float(np.clip((92.0 - cand_bpm) / 45.0, 0.15, 0.85))
    sc_3_4 = grp_scores[3] * (1.0 - 0.35 * slow_pulse_factor)
    sc_3_2 = grp_scores[3] * (0.45 + 0.55 * slow_pulse_factor)

    sc_4_4 = grp_scores[4] * (0.60 + 0.40 * eff_sub_bin)

    ts_probs = _norm_dict({"2/4": sc_2_4, "3/4": sc_3_4, "4/4": sc_4_4, "6/8": sc_6_8, "3/2": sc_3_2})

    return {
        "tactus_bpm": cand_bpm,
        "grp_scores": grp_scores,
        "grouping_quality": grouping_quality,
        "sub_bin": p_sub_bin,
        "sub_tern": p_sub_tern,
        "sub_rel": sub_rel,
        "tern_rep": tern_rep,
        "beat3_cont": beat3_is_continuation,
        "beat3_rel": b3_rel,
        "ts_probs": ts_probs,
        "phases": {
            "2/4": best_phases[2],
            "6/8": best_phases[2],
            "3/4": best_phases[3],
            "3/2": best_phases[3],
            "4/4": best_phases[4],
        },
        "reliabilities": {
            "periodicity": per_rel,
            "accent": acc_rel,
            "subdivision": sub_rel,
            "harmonic_change": harm_rel,
            "recurrence": rec_rel,
            "bass": bass_rel,
        },
        "feature_agreement": feature_agreement,
    }


def extract_meter_features(joint_hypotheses, quality_metrics) -> dict:
    top_h = joint_hypotheses[0] if joint_hypotheses else {}
    return {
        "tactus_bpm": float(top_h.get("tactus_bpm", 120.0)),
        "grp_2": float(top_h.get("grp_2", 0.33)),
        "grp_3": float(top_h.get("grp_3", 0.33)),
        "grp_4": float(top_h.get("grp_4", 0.34)),
        "sub_tern": float(top_h.get("sub_tern", 0.5)),
        "beat3_cont": float(top_h.get("beat3_cont", 0.5)),
        "signal_quality": float(quality_metrics.get("signal_quality", 0.5)),
        "grouping_quality": float(quality_metrics.get("grouping_quality", 0.5)),
        "subdivision_quality": float(quality_metrics.get("subdivision_quality", 0.5)),
        "feature_agreement": float(quality_metrics.get("feature_agreement", 0.5)),
        "bootstrap_stability": float(quality_metrics.get("bootstrap_stability", 0.5)),
    }


def run_hierarchical_meter_inference(tempo_data, dedup_notes, sr, log: TechnicalLogger):
    segments = tempo_data["segments"]
    top_tactus = tempo_data["top_tactus"][:3]
    y_harm = tempo_data["y_harm"]
    y_perc = tempo_data["y_perc"]
    onset_peaks = tempo_data["onset_peaks_times"]
    hop = CONFIG_SIGNAL["hop_length"]

    if not segments or not top_tactus:
        return {
            "chosen_bpm": tempo_data["global_raw_bpm"],
            "time_sig": "4/4",
            "status": "ambiguous",
            "confidence_label": "inconclusiva",
            "pulse_ql": 1.0,
            "meter_sections": [{"start_sec": 0.0, "time_sig": "4/4"}],
            "qualities": {
                "signal_quality": tempo_data["signal_quality"],
                "beat_quality": 0.5,
                "tactus_quality": 0.5,
                "grouping_quality": 0.0,
                "subdivision_quality": 0.0,
                "feature_agreement": 0.0,
                "sequence_stability": 0.0,
                "bootstrap_stability": 0.0,
                "decision_quality": 0.0,
            }
        }

    joint_hypotheses = []
    tactus_evaluations = {}

    for t_cand in top_tactus:
        c_bpm = t_cand["bpm"]
        t_prior = t_cand["score"]
        seg_evals = []

        for s in segments:
            s0 = s["frame_start"] * hop
            s1 = s["frame_end"] * hop
            s_notes = [n for n in dedup_notes if s["time_start"] <= n.start_sec <= s["time_end"]]
            ev = _evaluate_segment_for_tactus(
                s, y_harm[s0:s1], y_perc[s0:s1], sr, c_bpm, onset_peaks, s_notes
            )
            if ev is not None:
                ev["pos_label"] = s["pos_label"]
                ev["time_start"] = s["time_start"]
                seg_evals.append(ev)

        if not seg_evals:
            continue

        tactus_evaluations[c_bpm] = seg_evals
        candidates_ts = ["2/4", "3/4", "4/4", "6/8", "3/2"]

        for ts in candidates_ts:
            mean_ts_prob = float(np.mean([e["ts_probs"][ts] for e in seg_evals]))
            mean_fa = float(np.mean([e["feature_agreement"] for e in seg_evals]))
            mean_gq = float(np.mean([e["grouping_quality"] for e in seg_evals]))
            mean_sq = float(np.mean([e["sub_rel"] for e in seg_evals]))
            rep_phase = int(np.median([e["phases"][ts] for e in seg_evals]))

            grp_map = {"2/4": 2, "6/8": 2, "3/4": 3, "3/2": 3, "4/4": 4}
            sub_map = {"2/4": "simples", "3/4": "semínima", "4/4": "simples", "6/8": "composto", "3/2": "mínima"}

            joint_score = mean_ts_prob * (0.65 + 0.35 * t_prior) * (0.75 + 0.25 * mean_fa)
            joint_rel = 0.45 * mean_gq + 0.30 * mean_fa + 0.25 * mean_sq

            joint_hypotheses.append({
                "tactus_bpm": c_bpm,
                "time_sig": ts,
                "grouping": grp_map[ts],
                "subdivision": sub_map[ts],
                "phase": rep_phase,
                "score": round(joint_score, 4),
                "ts_prob": round(mean_ts_prob, 4),
                "reliability": round(joint_rel, 4),
                "grp_2": round(float(np.mean([e["grp_scores"][2] for e in seg_evals])), 3),
                "grp_3": round(float(np.mean([e["grp_scores"][3] for e in seg_evals])), 3),
                "grp_4": round(float(np.mean([e["grp_scores"][4] for e in seg_evals])), 3),
                "sub_tern": round(float(np.mean([e["sub_tern"] for e in seg_evals])), 3),
                "beat3_cont": round(float(np.mean([e["beat3_cont"] for e in seg_evals])), 3),
            })

    if not joint_hypotheses:
        return {
            "chosen_bpm": tempo_data["global_raw_bpm"],
            "time_sig": "4/4",
            "status": "ambiguous",
            "confidence_label": "inconclusiva",
            "pulse_ql": 1.0,
            "meter_sections": [{"start_sec": 0.0, "time_sig": "4/4"}],
            "qualities": {
                "signal_quality": tempo_data["signal_quality"],
                "beat_quality": 0.5,
                "tactus_quality": 0.5,
                "grouping_quality": 0.0,
                "subdivision_quality": 0.0,
                "feature_agreement": 0.0,
                "sequence_stability": 0.0,
                "bootstrap_stability": 0.0,
                "decision_quality": 0.0,
            }
        }

    joint_hypotheses.sort(key=lambda x: x["score"], reverse=True)
    best_h = joint_hypotheses[0]
    chosen_bpm = best_h["tactus_bpm"]
    seg_evals_chosen = tactus_evaluations[chosen_bpm]

    states = ["2/4", "3/4", "4/4", "6/8", "3/2"]
    n_st = len(states)
    n_win = len(seg_evals_chosen)
    self_p = CONFIG_METER["viterbi_self_transition"]
    switch_p = (1.0 - self_p) / float(n_st - 1)

    dp = np.zeros((n_win, n_st), dtype=float)
    back = np.zeros((n_win, n_st), dtype=int)

    for s_i, st_name in enumerate(states):
        dp[0, s_i] = math.log(seg_evals_chosen[0]["ts_probs"][st_name] + 1e-9)

    for t in range(1, n_win):
        for s_curr, st_name in enumerate(states):
            em = math.log(seg_evals_chosen[t]["ts_probs"][st_name] + 1e-9)
            scores_prev = [
                dp[t - 1, s_prev] + math.log(self_p if s_prev == s_curr else switch_p) + em
                for s_prev in range(n_st)
            ]
            best_prev = int(np.argmax(scores_prev))
            dp[t, s_curr] = scores_prev[best_prev]
            back[t, s_curr] = best_prev

    viterbi_idx = [int(np.argmax(dp[-1]))]
    for t in range(n_win - 1, 0, -1):
        viterbi_idx.append(int(back[t, viterbi_idx[-1]]))
    viterbi_idx.reverse()
    viterbi_path = [states[i] for i in viterbi_idx]

    meter_sections = []
    for t, st_name in enumerate(viterbi_path):
        t_sec = 0.0 if t == 0 else round(seg_evals_chosen[t]["time_start"], 2)
        if not meter_sections or meter_sections[-1]["time_sig"] != st_name:
            meter_sections.append({"start_sec": t_sec, "time_sig": st_name})

    sequence_stability = float(viterbi_path.count(viterbi_path[0]) / float(len(viterbi_path)))

    rng = np.random.default_rng(42)
    n_boot = CONFIG_CONFIDENCE["bootstrap_iterations"]
    boot_counts = {c: 0 for c in states}

    for _ in range(n_boot):
        sampled_evals = rng.choice(seg_evals_chosen, size=len(seg_evals_chosen), replace=True)
        agg = {c: 0.0 for c in states}
        for ev in sampled_evals:
            noise_scale = 0.06 * (1.0 - ev["grouping_quality"])
            for c in states:
                agg[c] += max(0.0, ev["ts_probs"][c] + float(rng.normal(0.0, noise_scale)))
        win_c = max(agg.items(), key=lambda x: x[1])[0]
        boot_counts[win_c] += 1

    global_ts_scores = {c: float(np.mean([e["ts_probs"][c] for e in seg_evals_chosen])) for c in states}
    g_tot = sum(global_ts_scores.values()) + 1e-8
    global_ts_scores = {k: v / g_tot for k, v in global_ts_scores.items()}

    ranked_ts = sorted(global_ts_scores.items(), key=lambda x: x[1], reverse=True)
    best_ts, best_sc = ranked_ts[0]
    sec_ts, sec_sc = ranked_ts[1]
    margin = best_sc - sec_sc

    bootstrap_stability = boot_counts[best_ts] / float(n_boot)

    tactus_quality = float(top_tactus[0]["score"])
    grouping_quality = float(np.mean([e["grouping_quality"] for e in seg_evals_chosen]))
    subdivision_quality = float(np.mean([e["sub_rel"] for e in seg_evals_chosen]))
    feature_agreement = float(np.mean([e["feature_agreement"] for e in seg_evals_chosen]))

    decision_quality = float(np.clip(
        0.28 * min(1.0, margin * 6.0) +
        0.24 * bootstrap_stability +
        0.20 * feature_agreement +
        0.16 * grouping_quality +
        0.12 * sequence_stability,
        0.0, 1.0
    ))

    base_thr = CONFIG_CONFIDENCE["base_margin_threshold"]
    adj_thr = -0.025 * (decision_quality - 0.55) - 0.020 * (bootstrap_stability - 0.65)
    eff_thr = float(np.clip(base_thr + adj_thr, CONFIG_CONFIDENCE["min_margin_abs"], CONFIG_CONFIDENCE["max_margin_abs"]))

    if (
        margin >= eff_thr and
        decision_quality >= CONFIG_CONFIDENCE["min_decision_quality"] and
        bootstrap_stability >= CONFIG_CONFIDENCE["min_bootstrap_stability"]
    ):
        status = "detected"
        final_ts = best_ts
        conf_label = "alta" if (decision_quality >= 0.68 and bootstrap_stability >= 0.78) else "média"
        refusal_reason = "Nenhuma (decisão aceita por consenso estatístico)"
    else:
        status = "ambiguous"
        final_ts = "4/4"
        conf_label = "inconclusiva"
        refusal_reason = (
            f"Incerteza elevada (margin={margin:.4f} vs thr={eff_thr:.4f}, "
            f"decision_q={decision_quality:.3f}, bootstrap={bootstrap_stability:.2f})"
        )

    pulse_ql_map = {"2/4": 1.0, "3/4": 1.0, "4/4": 1.0, "6/8": 1.5, "3/2": 2.0}
    pulse_ql = pulse_ql_map.get(final_ts, 1.0)

    avg_rels = {}
    for k_r in ("periodicity", "accent", "subdivision", "harmonic_change", "recurrence", "bass"):
        avg_rels[k_r] = float(np.mean([e["reliabilities"][k_r] for e in seg_evals_chosen]))

    log.add("\n==============================\nHIERARCHICAL METER & VITERBI ENGINE\n==============================")
    log.add("Top Hipóteses Conjuntas (Tactus + Agrupamento + Subdivisão + Fase):")
    for h_i in joint_hypotheses[:6]:
        log.add(
            f"  H: {h_i['tactus_bpm']:6.2f} BPM | {h_i['time_sig']} (grp={h_i['grouping']}, "
            f"sub={h_i['subdivision']}, phase={h_i['phase']}) -> score={h_i['score']:.4f}, rel={h_i['reliability']:.3f}"
        )

    log.add("\nFEATURE RELIABILITY (No Tactus Escolhido):")
    for k_r, v_r in avg_rels.items():
        w_b = CONFIG_METER.get(f"w_{k_r}", 0.15)
        log.add(f"  {k_r:16s}: reliability={v_r:.3f} | peso_base={w_b:.2f} -> peso_efetivo={w_b*(0.25+0.75*v_r):.3f}")

    for e in seg_evals_chosen:
        log.add(
            f"  Seg {e['pos_label']}: Grp(2={e['grp_scores'][2]:.2f}, 3={e['grp_scores'][3]:.2f}, 4={e['grp_scores'][4]:.2f}) | "
            f"Beat3_Cont={e['beat3_cont']:.2f} | SubTern={e['sub_tern']:.2f} (rep={e['tern_rep']:.2f}) | "
            f"TS={ {k: round(v, 2) for k, v in e['ts_probs'].items()} }"
        )

    log.add(f"\nVITERBI SEQUENCE: {viterbi_path} -> Estado Inicial={viterbi_path[0]} | Final={viterbi_path[-1]}")
    log.add(f"Meter Sections detectadas: {meter_sections}")
    log.add(f"BOOTSTRAP ({n_boot} runs): {boot_counts} -> Bootstrap Stability={bootstrap_stability:.2f}")

    log.add("\nQUALITY METRICS (Signal != Decision):")
    log.add(
        f"  signal_quality={tempo_data['signal_quality']:.3f} | tactus_quality={tactus_quality:.3f} | "
        f"grouping_quality={grouping_quality:.3f} | subdivision_quality={subdivision_quality:.3f}\n"
        f"  feature_agreement={feature_agreement:.3f} | sequence_stability={sequence_stability:.3f} | "
        f"bootstrap_stability={bootstrap_stability:.3f} | decision_quality={decision_quality:.3f}"
    )
    log.add(
        f"  Melhor={best_ts} ({best_sc:.4f}) | Segundo={sec_ts} ({sec_sc:.4f}) | Margem={margin:.4f} "
        f"(Limiar Base={base_thr:.4f}, Efetivo={eff_thr:.4f})"
    )
    log.add(f"  Status={status} | Confiança={conf_label} | Motivo: {refusal_reason}")

    qualities_dict = {
        "signal_quality": round(tempo_data["signal_quality"], 3),
        "tactus_quality": round(tactus_quality, 3),
        "grouping_quality": round(grouping_quality, 3),
        "subdivision_quality": round(subdivision_quality, 3),
        "feature_agreement": round(feature_agreement, 3),
        "sequence_stability": round(sequence_stability, 3),
        "bootstrap_stability": round(bootstrap_stability, 3),
        "decision_quality": round(decision_quality, 3),
    }

    _ = extract_meter_features(joint_hypotheses, qualities_dict)

    return {
        "chosen_bpm": round(chosen_bpm, 1),
        "time_sig": final_ts,
        "status": status,
        "best_cand": best_ts,
        "second_cand": sec_ts,
        "margin": round(margin, 4),
        "confidence_label": conf_label,
        "pulse_ql": pulse_ql,
        "meter_sections": meter_sections,
        "qualities": qualities_dict,
    }


# ---------------------------------------------------------
# 8. INFERÊNCIA BASIC PITCH COM OVERLAP E DEDUPLICAÇÃO
# ---------------------------------------------------------
def infer_notes_with_overlap(y_full, sr, session_dir: Path, ai_model, log: TechnicalLogger):
    total_duration = len(y_full) / float(sr)
    chunk_dur = CONFIG_SIGNAL["chunk_duration_sec"]
    overlap = CONFIG_SIGNAL["chunk_overlap_sec"]
    step = chunk_dur - overlap

    raw_notes = []
    start_time = 0.0
    chunk_idx = 0
    notes_per_chunk = []

    while start_time < total_duration:
        end_time = min(start_time + chunk_dur, total_duration)
        if (end_time - start_time) < CONFIG_SIGNAL["min_chunk_sec"] and chunk_idx > 0:
            break

        s0 = int(round(start_time * sr))
        s1 = int(round(end_time * sr))
        y_chunk = y_full[s0:s1]

        chunk_file = session_dir / f"chunk_{chunk_idx}.wav"
        sf.write(str(chunk_file), y_chunk, sr, subtype="PCM_16")

        _, midi_chunk, _ = predict(str(chunk_file.resolve()), ai_model)
        c_count = 0
        for inst in midi_chunk.instruments:
            for n in inst.notes:
                raw_notes.append(MockNote(n.start + start_time, n.end + start_time, n.pitch, n.velocity, source_chunk=chunk_idx))
                c_count += 1
        notes_per_chunk.append((chunk_idx, round(start_time, 1), round(end_time, 1), c_count))

        try:
            chunk_file.unlink(missing_ok=True)
        except Exception:
            pass
        del y_chunk, midi_chunk
        gc.collect()

        if end_time >= total_duration:
            break
        start_time += step
        chunk_idx += 1

    raw_notes.sort(key=lambda x: (x.start_sec, x.pitch))
    dedup_notes = []
    overlap_merged = 0
    tol = CONFIG_SIGNAL["dedup_time_tol_sec"]

    for n in raw_notes:
        if n.duration_sec < CONFIG_SIGNAL["duration_min_abs_sec"]:
            continue
        merged_flag = False
        for prev in reversed(dedup_notes[-15:]):
            if prev.pitch == n.pitch:
                if abs(n.start_sec - prev.start_sec) <= tol or (
                    n.source_chunk != prev.source_chunk and n.start_sec <= prev.end_sec + tol
                ):
                    prev.end_sec = max(prev.end_sec, n.end_sec)
                    prev.velocity = max(prev.velocity, n.velocity)
                    overlap_merged += 1
                    merged_flag = True
                    break
            if prev.end_sec < n.start_sec - 0.5:
                break
        if not merged_flag:
            dedup_notes.append(n)

    log.add("\n==============================\nBASIC PITCH OVERLAP & DEDUPLICATION\n==============================")
    log.add(f"Total de chunks: {len(notes_per_chunk)} | Notas brutas: {len(raw_notes)} | Duplicatas de overlap fundidas: {overlap_merged} | Notas pós-dedup: {len(dedup_notes)}")

    return raw_notes, dedup_notes


# ---------------------------------------------------------
# 9. RASTREAMENTO MELÓDICO COM VITERBI, PHRASE RESET E NO_MELODY
# ---------------------------------------------------------
def track_melody_viterbi_a4(dedup_notes, onset_norm, sr, log: TechnicalLogger):
    log.add("\n==============================\nMELODY TRACKING (VITERBI + NO_MELODY + PHRASE RESET)\n==============================")
    if not dedup_notes:
        return [], 0.0

    hop = CONFIG_SIGNAL["hop_length"]
    vels = np.array([n.velocity for n in dedup_notes], dtype=float)
    pitches = np.array([n.pitch for n in dedup_notes], dtype=float)
    durs = np.array([n.duration_sec for n in dedup_notes], dtype=float)

    v_med, v_p25, v_p75 = float(np.median(vels)), float(np.percentile(vels, 25)), float(np.percentile(vels, 75))
    p_med, p_p20, p_p85 = float(np.median(pitches)), float(np.percentile(pitches, 20)), float(np.percentile(pitches, 85))
    d_med = float(np.median(durs))

    v_base = CONFIG_MELODY["velocity_base"]
    v_adapt = v_med - 0.18 * (v_p75 - v_p25)
    v_eff = float(np.clip(0.45 * v_base + 0.55 * v_adapt, CONFIG_MELODY["velocity_min_floor"], CONFIG_MELODY["velocity_max_ceil"]))

    p_base = CONFIG_MELODY["pitch_base_min"]
    p_adapt = max(CONFIG_MELODY["pitch_floor_abs"], int(round(p_p20 - 2)))
    p_eff = int(round(0.45 * p_base + 0.55 * p_adapt))

    d_base = CONFIG_MELODY["duration_base_sec"]
    d_adapt = max(CONFIG_SIGNAL["duration_min_abs_sec"], min(0.12, d_med * 0.45))
    d_eff = float(0.50 * d_base + 0.50 * d_adapt)

    log.add(f"VELOCITY MELODIA -> Base={v_base:.1f} | Medido(Med)={v_med:.1f} | Adaptativo={v_adapt:.1f} | Efetivo={v_eff:.1f}")
    log.add(f"PITCH MÍNIMO     -> Base={p_base} | Medido(P20)={p_p20:.1f} | Adaptativo={p_adapt} | Efetivo={p_eff}")
    log.add(f"DURAÇÃO MÍNIMA   -> Base={d_base:.3f}s | Medido(Med)={d_med:.3f}s | Adaptativo={d_adapt:.3f}s | Efetivo={d_eff:.3f}s")

    candidates = [n for n in dedup_notes if n.velocity >= v_eff and n.pitch >= p_eff and n.duration_sec >= d_eff]
    if not candidates:
        candidates = [n for n in dedup_notes if n.velocity >= max(24.0, v_p25 * 0.85) and n.pitch >= CONFIG_MELODY["pitch_floor_abs"]]
    if not candidates:
        return [], 0.0

    candidates.sort(key=lambda x: (x.start_sec, x.pitch))
    frames = []
    tol_sim = CONFIG_MELODY["simultaneous_tolerance_sec"]
    for n in candidates:
        if not frames or abs(n.start_sec - frames[-1][0].start_sec) > tol_sim:
            frames.append([n])
        else:
            frames[-1].append(n)

    n_frames = len(frames)
    mel_center_pitch = float(np.percentile([n.pitch for n in candidates], 65))
    no_mel_thr = CONFIG_MELODY["no_melody_threshold"]
    no_mel_trans_pen = CONFIG_MELODY["no_melody_transition_cost"]

    dp_scores = []
    dp_back = []
    dp_last_pitch = []
    dp_last_end = []

    for t_idx, frame_notes in enumerate(frames):
        k_notes = len(frame_notes)
        curr_scores = np.full(k_notes + 1, -1e9, dtype=float)
        curr_back = np.full(k_notes + 1, -1, dtype=int)
        curr_lp = np.full(k_notes + 1, mel_center_pitch, dtype=float)
        curr_le = np.zeros(k_notes + 1, dtype=float)

        em_scores = np.zeros(k_notes, dtype=float)
        for i, nt in enumerate(frame_notes):
            vel_norm = np.clip((nt.velocity - v_eff) / (127.0 - v_eff + 1e-8), 0.0, 1.0)
            dur_norm = min(1.0, nt.duration_sec / 0.75)
            reg_bias = max(0.0, 1.0 - abs(nt.pitch - mel_center_pitch) / 36.0)
            f_idx = min(len(onset_norm) - 1, max(0, int(round(nt.start_sec * sr / hop)))) if len(onset_norm) > 0 else 0
            ons_val = float(onset_norm[f_idx]) if len(onset_norm) > 0 else 0.5
            em_scores[i] = 10.0 * (0.42 * vel_norm + 0.26 * dur_norm + 0.18 * reg_bias + 0.14 * ons_val)

        if t_idx == 0:
            for i in range(k_notes):
                curr_scores[i] = em_scores[i]
                curr_lp[i] = float(frame_notes[i].pitch)
                curr_le[i] = float(frame_notes[i].end_sec)
            curr_scores[k_notes] = no_mel_thr
            curr_lp[k_notes] = mel_center_pitch
            curr_le[k_notes] = 0.0
        else:
            prev_scores = dp_scores[t_idx - 1]
            prev_lp = dp_last_pitch[t_idx - 1]
            prev_le = dp_last_end[t_idx - 1]
            n_prev_states = len(prev_scores)

            for i, nt in enumerate(frame_notes):
                best_sc = -1e9
                best_j = 0
                for j in range(n_prev_states):
                    gap = max(0.0, nt.start_sec - prev_le[j])
                    interval = abs(nt.pitch - prev_lp[j])

                    if gap >= CONFIG_MELODY["phrase_reset_gap_sec"]:
                        gap_mult = CONFIG_MELODY["phrase_reset_jump_mult"]
                        oct_pen = 0.0
                        cont_bon = 0.0
                    elif gap <= CONFIG_MELODY["short_gap_sec"]:
                        gap_mult = CONFIG_MELODY["short_gap_jump_mult"]
                        oct_pen = CONFIG_MELODY["octave_jump_penalty"] if interval >= 12 else 0.0
                        cont_bon = CONFIG_MELODY["continuity_bonus"] if interval <= 4 else 0.0
                    else:
                        gap_mult = 1.0
                        oct_pen = CONFIG_MELODY["octave_jump_penalty"] if interval >= 12 else 0.0
                        cont_bon = CONFIG_MELODY["continuity_bonus"] * 0.7 if interval <= 5 else 0.0

                    jump_pen = (interval * CONFIG_MELODY["jump_penalty_semitone"] + oct_pen) * gap_mult
                    cand_sc = prev_scores[j] + em_scores[i] - jump_pen + cont_bon
                    if cand_sc > best_sc:
                        best_sc = cand_sc
                        best_j = j

                curr_scores[i] = best_sc
                curr_back[i] = best_j
                curr_lp[i] = float(nt.pitch)
                curr_le[i] = float(nt.end_sec)

            best_nm_sc = -1e9
            best_nm_j = 0
            for j in range(n_prev_states):
                sc_nm = prev_scores[j] + no_mel_thr - (0.0 if j == n_prev_states - 1 else no_mel_trans_pen)
                if sc_nm > best_nm_sc:
                    best_nm_sc = sc_nm
                    best_nm_j = j
            curr_scores[k_notes] = best_nm_sc
            curr_back[k_notes] = best_nm_j
            curr_lp[k_notes] = prev_lp[best_nm_j]
            curr_le[k_notes] = prev_le[best_nm_j]

        dp_scores.append(curr_scores)
        dp_back.append(curr_back)
        dp_last_pitch.append(curr_lp)
        dp_last_end.append(curr_le)

    melody_line = []
    no_melody_frames = 0
    curr_st = int(np.argmax(dp_scores[-1]))

    for t_idx in range(n_frames - 1, -1, -1):
        f_notes = frames[t_idx]
        if curr_st < len(f_notes):
            melody_line.append(f_notes[curr_st])
        else:
            no_melody_frames += 1
        curr_st = int(dp_back[t_idx][curr_st])
    melody_line.reverse()

    if not melody_line and frames:
        melody_line = [max(f, key=lambda x: x.velocity) for f in frames]

    phrase_resets = 0
    intervals = []
    for i in range(len(melody_line) - 1):
        if melody_line[i].end_sec > melody_line[i + 1].start_sec:
            melody_line[i].end_sec = max(
                melody_line[i].start_sec + CONFIG_SIGNAL["duration_min_abs_sec"],
                melody_line[i + 1].start_sec
            )
        gap = melody_line[i + 1].start_sec - melody_line[i].end_sec
        if gap >= CONFIG_MELODY["phrase_reset_gap_sec"]:
            phrase_resets += 1
        else:
            intervals.append(abs(melody_line[i + 1].pitch - melody_line[i].pitch))

    if intervals:
        mean_jump = float(np.mean(intervals))
        p50_jump = float(np.percentile(intervals, 50))
        p90_jump = float(np.percentile(intervals, 90))
        max_jump = int(np.max(intervals))
        oct_pct = sum(1 for iv in intervals if iv >= 12) / float(len(intervals))
    else:
        mean_jump, p50_jump, p90_jump, max_jump, oct_pct = 0.0, 0.0, 0.0, 0, 0.0

    avg_cands_per_frame = len(candidates) / float(max(1, n_frames))
    melody_confidence = float(np.clip(
        0.45 * (1.0 - min(1.0, oct_pct * 3.5)) +
        0.30 * (1.0 - min(1.0, mean_jump / 10.0)) +
        0.25 * (1.0 - min(0.8, (avg_cands_per_frame - 1.0) / 4.0)),
        0.1, 1.0
    ))

    log.add(
        f"Frames totais: {n_frames} | NO_MELODY frames: {no_melody_frames} | "
        f"Phrase resets: {phrase_resets} | Notas finais: {len(melody_line)}"
    )
    log.add(
        f"Saltos intra-frase (st) -> Médio={mean_jump:.2f} | P50={p50_jump:.1f} | "
        f"P90={p90_jump:.1f} | Máx={max_jump} | Saltos>=Oitava={oct_pct:.1%}"
    )
    log.add(f"Melody Confidence: {melody_confidence:.3f}")

    return melody_line, melody_confidence


# ---------------------------------------------------------
# 10. MOTOR DE CIFRAS (ROOT PRIMEIRO + QUALITY + HMM SMOOTHING)
# ---------------------------------------------------------
CHORD_QUALITY_SPECS = [
    ("maj",   {4: 1.0, 7: 0.75},                 {3: -0.9, 10: -0.6, 11: -0.6}, "",       ""),
    ("min",   {3: 1.0, 7: 0.75},                 {4: -0.9, 10: -0.6, 11: -0.6}, "m",      "m"),
    ("7",     {4: 0.9, 7: 0.6, 10: 1.0},         {3: -0.8, 11: -0.9},           "7",      "7"),
    ("maj7",  {4: 0.9, 7: 0.6, 11: 1.0},         {3: -0.8, 10: -0.9},           "7M",     "maj7"),
    ("m7",    {3: 0.9, 7: 0.6, 10: 1.0},         {4: -0.8, 11: -0.8},           "m7",     "m7"),
    ("m7b5",  {3: 0.9, 6: 1.0, 10: 0.9},         {4: -0.9, 7: -0.8},            "m7(b5)", "m7b5"),
    ("dim",   {3: 0.9, 6: 1.0},                  {4: -0.9, 7: -0.8, 10: -0.5},  "dim",    "dim"),
    ("sus4",  {5: 1.0, 7: 0.8},                  {3: -0.8, 4: -0.8},            "4",      "sus4"),
    ("7sus4", {5: 0.9, 7: 0.6, 10: 0.9},         {3: -0.8, 4: -0.8},            "4(7)",   "7sus4"),
    ("add9",  {2: 0.85, 4: 0.9, 7: 0.7},         {3: -0.8, 10: -0.7, 11: -0.7}, "(add9)", "add9"),
    ("9",     {2: 0.8, 4: 0.85, 7: 0.5, 10: 0.9},{3: -0.8, 11: -0.8},           "9",      "9"),
    ("m9",    {2: 0.8, 3: 0.85, 7: 0.5, 10: 0.9},{4: -0.8, 11: -0.8},           "m9",     "m9"),
    ("7_13",  {4: 0.85, 9: 0.85, 10: 0.9},       {3: -0.8, 11: -0.8},           "7(13)",  "13"),
]

def analyze_harmony_root_quality_hmm(
    dedup_notes, y_harm, sr, beat_times, pulse_ql, bar_length_ql, detected_key, log: TechnicalLogger
):
    log.add("\n==============================\nHARMONY ENGINE (ROOT -> QUALITY -> HMM SMOOTHING)\n==============================")
    if not dedup_notes or len(beat_times) < 2:
        return {}, 0

    hop = CONFIG_SIGNAL["hop_length"]
    use_flats = prefer_flats_for_key(detected_key)

    key_pcs = set()
    if detected_key is not None:
        try:
            key_pcs = {p.pitchClass for p in detected_key.pitches}
        except Exception:
            pass

    vels = np.array([n.velocity for n in dedup_notes], dtype=float)
    v_h_base = CONFIG_HARMONY["velocity_base"]
    v_h_adapt = float(np.percentile(vels, 28))
    v_h_eff = float(max(CONFIG_HARMONY["velocity_min_floor"], 0.50 * v_h_base + 0.50 * v_h_adapt))

    harm_notes = [n for n in dedup_notes if n.velocity >= v_h_eff] or dedup_notes
    weights_all = np.array([n.duration_sec * (n.velocity / 127.0) for n in harm_notes], dtype=float)
    w_med, w_p70 = float(np.median(weights_all)), float(np.percentile(weights_all, 70))
    peso_base = CONFIG_HARMONY["peso_base"]
    peso_adapt = 0.50 * w_med + 0.50 * w_p70
    peso_eff = float(max(CONFIG_HARMONY["peso_min_floor"], 0.45 * peso_base + 0.55 * peso_adapt))

    log.add(f"VELOCITY HARMONIA -> Base={v_h_base:.1f} | Medido(P28)={v_h_adapt:.1f} | Efetivo={v_h_eff:.1f}")
    log.add(f"PESO ACORDE       -> Base={peso_base:.3f} | Medido(Med={w_med:.3f}, P70={w_p70:.3f}) | Adapt={peso_adapt:.3f} | Efetivo={peso_eff:.3f}")

    try:
        chroma_full = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop)
    except Exception:
        chroma_full = None

    mapped_notes = []
    for n in harm_notes:
        s_ql = seconds_to_musical_position(n.start_sec, beat_times, pulse_ql=pulse_ql)
        e_ql = seconds_to_musical_position(n.end_sec, beat_times, pulse_ql=pulse_ql)
        d_ql = max(0.25, e_ql - s_ql)
        w_val = n.duration_sec * (n.velocity / 127.0)
        mapped_notes.append((s_ql, e_ql, d_ql, n.pitch, w_val, n.start_sec, n.end_sec))

    max_ql = max(x[1] for x in mapped_notes)
    total_bars = int(math.ceil(max_ql / bar_length_ql)) + 1
    n_sub = 2 if bar_length_ql >= 3.0 else 1
    reg_len_ql = bar_length_ql / float(n_sub)

    raw_regions = []
    abstention_count = 0

    for b_idx in range(total_bars):
        bar_start_ql = b_idx * bar_length_ql
        for s_i in range(n_sub):
            r_start_ql = bar_start_ql + s_i * reg_len_ql
            r_end_ql = r_start_ql + reg_len_ql

            active = []
            t_starts, t_ends = [], []
            for s_ql, e_ql, d_ql, pt, w_val, s_sec, e_sec in mapped_notes:
                ov = min(e_ql, r_end_ql) - max(s_ql, r_start_ql)
                if ov > 0.05:
                    m_boost = 1.25 if abs(s_ql - r_start_ql) < 0.30 else 1.0
                    active.append((pt, w_val * (ov / d_ql) * m_boost))
                    t_starts.append(s_sec)
                    t_ends.append(e_sec)

            if not active or sum(x[1] for x in active) < peso_eff * 0.45:
                continue

            pc_notes = np.zeros(12, dtype=float)
            bass_hist = np.zeros(12, dtype=float)
            lowest_pitch = 999
            lowest_pc = None

            for pt, w_e in active:
                pc_notes[pt % 12] += w_e
                if pt <= CONFIG_HARMONY["bass_ceiling_midi"]:
                    bass_hist[pt % 12] += w_e * 1.5
                if pt < lowest_pitch:
                    lowest_pitch = pt
                    lowest_pc = pt % 12

            if np.max(pc_notes) > 0:
                pc_notes = pc_notes / np.max(pc_notes)

            pc_chroma = np.zeros(12, dtype=float)
            if chroma_full is not None and t_starts:
                f0 = max(0, int(round(min(t_starts) * sr / hop)))
                f1 = min(chroma_full.shape[1], int(round(max(t_ends) * sr / hop)) + 1)
                if f1 > f0:
                    c_slice = np.mean(chroma_full[:, f0:f1], axis=1)
                    if np.max(c_slice) > 0:
                        pc_chroma = c_slice / np.max(c_slice)

            pc_comb = 0.62 * pc_notes + 0.38 * pc_chroma
            if np.max(pc_comb) <= 1e-6:
                continue
            pc_comb = pc_comb / np.max(pc_comb)

            # ETAPA 1: Estimativa de ROOT
            root_scores = {}
            for r_pc in range(12):
                if pc_comb[r_pc] < 0.15:
                    root_scores[r_pc] = 0.0
                    continue
                b_supp = float(bass_hist[r_pc] / (np.max(bass_hist) + 1e-8)) if np.max(bass_hist) > 0 else (1.0 if lowest_pc == r_pc else 0.0)
                fifth_supp = pc_comb[(r_pc + 7) % 12]
                third_supp = max(pc_comb[(r_pc + 4) % 12], pc_comb[(r_pc + 3) % 12])
                diatonic_bon = CONFIG_HARMONY["tonal_prior_weight"] if r_pc in key_pcs else 0.0

                r_sc = 0.38 * pc_comb[r_pc] + 0.28 * b_supp + 0.20 * fifth_supp + 0.14 * third_supp + diatonic_bon
                root_scores[r_pc] = float(r_sc)

            ranked_roots = sorted(root_scores.items(), key=lambda x: x[1], reverse=True)
            best_root, r_top1 = ranked_roots[0]
            r_top2 = ranked_roots[1][1]
            root_margin = r_top1 - r_top2
            root_conf = float(np.clip(0.55 * r_top1 + 0.45 * min(1.0, root_margin * 3.5), 0.0, 1.0))

            if root_conf < CONFIG_HARMONY["root_min_confidence"]:
                abstention_count += 1
                continue

            # ETAPA 2: Estimativa de QUALITY
            rel_iv = np.roll(pc_comb, -best_root)
            e_m3 = float(rel_iv[3])
            e_M3 = float(rel_iv[4])
            e_sus4 = float(rel_iv[5])

            max_discriminant = max(e_m3, e_M3, e_sus4)
            root_str = pc_to_note_name(best_root, use_flats=use_flats)

            if max_discriminant < CONFIG_HARMONY["third_presence_min_ratio"]:
                abstention_count += 1
                raw_regions.append({
                    "bar_idx": b_idx,
                    "offset_in_bar_ql": s_i * reg_len_ql,
                    "root_pc": best_root,
                    "root_str": root_str,
                    "root_conf": round(root_conf, 3),
                    "quality": "UNKNOWN",
                    "mpb_sym": None,
                    "xml_sym": None,
                    "qual_top1": ("UNKNOWN", 0.0),
                    "qual_top2": ("NONE", 0.0),
                    "qual_margin": 0.0,
                    "qual_conf": 0.0,
                    "chord_conf": 0.0,
                    "reason": "Ausência de nota discriminante (3ª ou 4ª)",
                })
                continue

            qual_cands = []
            for q_name, req_ivs, pen_ivs, mpb_suf, xml_suf in CHORD_QUALITY_SPECS:
                pos_sc = sum(rel_iv[iv] * w_iv for iv, w_iv in req_ivs.items()) / float(sum(req_ivs.values()))
                pen_sc = sum(rel_iv[iv] * abs(w_pen) for iv, w_pen in pen_ivs.items() if rel_iv[iv] > 0.25)
                ext_pen = 0.04 * max(0, len(req_ivs) - 2)
                q_sc = float(np.clip(pos_sc - 0.35 * pen_sc - ext_pen, 0.0, 1.0))
                qual_cands.append((q_sc, q_name, mpb_suf, xml_suf))

            qual_cands.sort(key=lambda x: x[0], reverse=True)
            q_sc1, q_name1, mpb_s1, xml_s1 = qual_cands[0]
            q_sc2, q_name2, _, _ = qual_cands[1]
            q_margin = q_sc1 - q_sc2
            qual_conf = float(np.clip(0.60 * q_sc1 + 0.40 * min(1.0, q_margin * 4.0), 0.0, 1.0))
            chord_conf = float(np.sqrt(root_conf * qual_conf))

            if qual_conf < CONFIG_HARMONY["quality_min_confidence"] or q_margin < CONFIG_HARMONY["quality_min_margin"]:
                abstention_count += 1
                raw_regions.append({
                    "bar_idx": b_idx,
                    "offset_in_bar_ql": s_i * reg_len_ql,
                    "root_pc": best_root,
                    "root_str": root_str,
                    "root_conf": round(root_conf, 3),
                    "quality": "UNKNOWN",
                    "mpb_sym": None,
                    "xml_sym": None,
                    "qual_top1": (q_name1, round(q_sc1, 3)),
                    "qual_top2": (q_name2, round(q_sc2, 3)),
                    "qual_margin": round(q_margin, 3),
                    "qual_conf": round(qual_conf, 3),
                    "chord_conf": round(chord_conf, 3),
                    "reason": f"Ambiguidade de qualidade ({q_name1}={q_sc1:.2f} vs {q_name2}={q_sc2:.2f})",
                })
                continue

            xml_root = note_name_to_music21(root_str)
            mpb_sym = f"{root_str}{mpb_s1}"
            xml_sym = f"{xml_root}{xml_s1}"

            if lowest_pc is not None and lowest_pc != best_root and lowest_pitch <= 53 and bass_hist[lowest_pc] > 0.45 * np.max(bass_hist):
                b_str = pc_to_note_name(lowest_pc, use_flats=use_flats)
                xml_b = note_name_to_music21(b_str)
                mpb_sym = f"{mpb_sym}/{b_str}"
                xml_sym = f"{xml_sym}/{xml_b}"

            raw_regions.append({
                "bar_idx": b_idx,
                "offset_in_bar_ql": s_i * reg_len_ql,
                "root_pc": best_root,
                "root_str": root_str,
                "root_conf": round(root_conf, 3),
                "quality": q_name1,
                "mpb_sym": mpb_sym,
                "xml_sym": xml_sym,
                "qual_top1": (q_name1, round(q_sc1, 3)),
                "qual_top2": (q_name2, round(q_sc2, 3)),
                "qual_margin": round(q_margin, 3),
                "qual_conf": round(qual_conf, 3),
                "chord_conf": round(chord_conf, 3),
                "reason": "Evidência harmônica suficiente",
            })

    # ETAPA 3: Continuidade Harmônica / HMM Smoothing
    valid_vocabulary = sorted(list({r["mpb_sym"] for r in raw_regions if r["mpb_sym"] is not None}))
    chords_by_bar = defaultdict(list)

    if valid_vocabulary and raw_regions:
        sym_to_xml = {r["mpb_sym"]: r["xml_sym"] for r in raw_regions if r["mpb_sym"] is not None}
        n_v = len(valid_vocabulary)
        stay_p = CONFIG_HARMONY["hmm_stay_prob"]
        change_p = (1.0 - stay_p) / float(max(1, n_v - 1)) if n_v > 1 else 1.0

        n_reg = len(raw_regions)
        dp_h = np.zeros((n_reg, n_v), dtype=float)
        back_h = np.zeros((n_reg, n_v), dtype=int)

        for t, reg in enumerate(raw_regions):
            for v_i, sym in enumerate(valid_vocabulary):
                if reg["mpb_sym"] == sym:
                    em_p = max(0.1, reg["chord_conf"])
                elif reg["quality"] == "UNKNOWN" and sym.startswith(reg["root_str"]):
                    em_p = 0.35 * reg["root_conf"]
                else:
                    em_p = 0.06

                log_em = math.log(em_p + 1e-9)
                if t == 0:
                    dp_h[0, v_i] = log_em
                else:
                    prev_list = [
                        dp_h[t - 1, p_i] + math.log(stay_p if p_i == v_i else change_p) + log_em
                        for p_i in range(n_v)
                    ]
                    best_p = int(np.argmax(prev_list))
                    dp_h[t, v_i] = prev_list[best_p]
                    back_h[t, v_i] = best_p

        path_v = [int(np.argmax(dp_h[-1]))]
        for t in range(n_reg - 1, 0, -1):
            path_v.append(int(back_h[t, path_v[-1]]))
        path_v.reverse()

        prev_chosen = None
        logged_h = 0
        for t, reg in enumerate(raw_regions):
            hmm_sym = valid_vocabulary[path_v[t]]
            if reg["mpb_sym"] is None:
                if prev_chosen and prev_chosen.startswith(reg["root_str"]):
                    chosen_mpb = prev_chosen
                    chosen_xml = sym_to_xml[chosen_mpb]
                    trans_reason = "Mantido acorde anterior (mesma raiz com qualidade local UNKNOWN)"
                else:
                    chosen_mpb = None
                    chosen_xml = None
                    trans_reason = f"ABSTENTION ({reg['reason']})"
            else:
                chosen_mpb = hmm_sym
                chosen_xml = sym_to_xml[chosen_mpb]
                if chosen_mpb != reg["mpb_sym"]:
                    trans_reason = f"Suavizado por HMM ({reg['mpb_sym']} -> {chosen_mpb})"
                elif chosen_mpb == prev_chosen:
                    trans_reason = "Continuidade harmônica"
                else:
                    trans_reason = "Nova transição harmônica confirmada"

            if logged_h < 14:
                log.add(
                    f"  Compasso {reg['bar_idx']+1} (off={reg['offset_in_bar_ql']:.1f} QL): "
                    f"Root={reg['root_str']}(conf={reg['root_conf']:.2f}) | "
                    f"Qual Top1={reg['qual_top1']} Top2={reg['qual_top2']} (m={reg['qual_margin']:.2f}) | "
                    f"Ant={prev_chosen} -> Escolhido={chosen_mpb} [{trans_reason}]"
                )
                logged_h += 1

            if chosen_mpb is not None:
                prev_chosen = chosen_mpb
                chords_by_bar[reg["bar_idx"]].append({
                    "offset_in_bar_ql": reg["offset_in_bar_ql"],
                    "mpb_sym": chosen_mpb,
                    "xml_sym": chosen_xml,
                    "confidence": reg["chord_conf"],
                })

        for b_k in list(chords_by_bar.keys()):
            c_list = chords_by_bar[b_k]
            if len(c_list) >= 2 and c_list[0]["mpb_sym"] == c_list[1]["mpb_sym"]:
                chords_by_bar[b_k] = [c_list[0]]

    log.add(f"Regiões avaliadas: {len(raw_regions)} | Abstentions harmônicas: {abstention_count} | Compassos cifrados: {len(chords_by_bar)}")
    return chords_by_bar, abstention_count


# ---------------------------------------------------------
# 11. QUANTIZAÇÃO GLOBAL ÓTIMA E DECOMPOSIÇÃO DE DURAÇÕES
# ---------------------------------------------------------
def decompose_into_expressible_durations(dur_ql: float) -> list:
    q = round(float(dur_ql) * 4.0) / 4.0
    if q <= 0.0:
        return [0.25]
    expressible = set(CONFIG_QUANTIZATION["expressible_single_qls"])
    if q in expressible:
        return [q]

    parts = []
    rem = q
    ordered_valid = sorted(list(expressible), reverse=True)
    while rem > 1e-4:
        picked = 0.25
        for v in ordered_valid:
            if v <= rem + 1e-4:
                picked = v
                break
        parts.append(round(picked * 4.0) / 4.0)
        rem = round((rem - picked) * 4.0) / 4.0
    return parts if parts else [0.25]


def quantize_melody_dp_global(melody_notes, beat_times, pulse_ql, bar_length_ql, log: TechnicalLogger):
    sub_ql = CONFIG_QUANTIZATION["grid_subdivision_ql"]
    w_on = CONFIG_QUANTIZATION["w_onset_error"]
    w_dur = CONFIG_QUANTIZATION["w_dur_error"]
    w_cx = CONFIG_QUANTIZATION["w_complexity"]
    expressible = set(CONFIG_QUANTIZATION["expressible_single_qls"])

    quantized_events = []
    onset_errors = []
    altered_events = 0

    for n in melody_notes:
        raw_s = seconds_to_musical_position(n.start_sec, beat_times, pulse_ql=pulse_ql)
        raw_e = seconds_to_musical_position(n.end_sec, beat_times, pulse_ql=pulse_ql)
        raw_d = max(sub_ql, raw_e - raw_s)

        s_floor = math.floor(raw_s / sub_ql) * sub_ql
        s_cands = [round(s_floor * 4.0) / 4.0, round((s_floor + sub_ql) * 4.0) / 4.0]

        best_cost = 1e9
        best_s = s_cands[0]
        best_e = best_s + sub_ql

        for sc in s_cands:
            d_floor = max(sub_ql, math.floor(raw_d / sub_ql) * sub_ql)
            d_cands = [round(d_floor * 4.0) / 4.0, round((d_floor + sub_ql) * 4.0) / 4.0]
            for dc in d_cands:
                ec = round((sc + dc) * 4.0) / 4.0
                on_err = abs(sc - raw_s)
                dur_err = abs(dc - raw_d)
                cx_pen = (0.0 if (sc * 2.0).is_integer() else 0.35) + (0.0 if dc in expressible else 0.45)
                cost = w_on * on_err + w_dur * dur_err + w_cx * cx_pen
                if cost < best_cost:
                    best_cost = cost
                    best_s = sc
                    best_e = ec

        err_val = abs(best_s - raw_s)
        onset_errors.append(err_val)
        if err_val > 0.06:
            altered_events += 1

        quantized_events.append({
            "start_ql": best_s,
            "end_ql": best_e,
            "pitch": int(n.pitch),
            "velocity": int(n.velocity),
            "start_sec": n.start_sec,
            "end_sec": n.end_sec,
        })

    quantized_events.sort(key=lambda x: (x["start_ql"], -x["velocity"]))
    clean_events = []
    for ev in quantized_events:
        if not clean_events:
            clean_events.append(ev)
        else:
            prev = clean_events[-1]
            if abs(ev["start_ql"] - prev["start_ql"]) < 1e-4:
                continue
            if prev["end_ql"] > ev["start_ql"]:
                prev["end_ql"] = max(prev["start_ql"] + sub_ql, ev["start_ql"])
            clean_events.append(ev)

    if onset_errors:
        arr_e = np.array(onset_errors, dtype=float)
        e_mean = float(np.mean(arr_e))
        e_p50 = float(np.percentile(arr_e, 50))
        e_p75 = float(np.percentile(arr_e, 75))
        e_p90 = float(np.percentile(arr_e, 90))
        e_p95 = float(np.percentile(arr_e, 95))
        e_max = float(np.max(arr_e))
    else:
        e_mean, e_p50, e_p75, e_p90, e_p95, e_max = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    log.add("\n==============================\nGLOBAL DP QUANTIZATION AUDIT\n==============================")
    log.add(f"Eventos quantizados: {len(clean_events)} | Eventos alterados (>0.06 QL): {altered_events}")
    log.add(
        f"Erro de Onset (QL) -> Médio={e_mean:.4f} | P50={e_p50:.4f} | P75={e_p75:.4f} | "
        f"P90={e_p90:.4f} | P95={e_p95:.4f} | Máx={e_max:.4f}"
    )

    return clean_events, {"p50": round(e_p50, 4), "p90": round(e_p90, 4), "p95": round(e_p95, 4)}


def auditar_e_sanitizar_duracoes_musicxml(stream_obj, log: TechnicalLogger):
    corrigidos = 0
    for m in stream_obj.recurse().getElementsByClass(music21.stream.Measure):
        m_num = m.number
        for el in m.notesAndRests:
            if isinstance(el, music21.harmony.Harmony):
                continue
            ql = float(el.quarterLength)
            dtype = getattr(el.duration, "type", "")
            if ql <= 0.0 or dtype in ("inexpressible", "zero"):
                novo_ql = round(ql * 4.0) / 4.0
                if novo_ql <= 0.0:
                    novo_ql = 0.25
                if abs(novo_ql - ql) > 1e-5 or dtype == "inexpressible":
                    corrigidos += 1
                    log.add(
                        f"[Auditoria MusicXML] Corrigido {type(el).__name__} na medida {m_num}, "
                        f"offset {el.offset}: ql_original={ql} (type={dtype}) -> ql_corrigido={novo_ql}"
                    )
                    try:
                        el.duration = music21.duration.Duration(novo_ql)
                    except Exception:
                        el.quarterLength = novo_ql
    return corrigidos


# ---------------------------------------------------------
# 12. CABEÇALHO E INTERFACE WEB (EXPERIÊNCIA ORIGINAL INTACTA)
# ---------------------------------------------------------
st.title("🎵 Partitura.ai")
st.markdown("### Sua Música em Partitura com Melodia e Cifras Harmônicas")
st.write("Envie um arquivo de áudio para transcrever a melodia, extrair o campo harmônico avançado (cifras e acordes base), visualizar a partitura e exportar em **PDF**, **MusicXML** e **MIDI**.")

with st.sidebar:
    st.markdown("### ⚙️ Configurações de Métrica")
    compasso_selecionado = st.selectbox(
        "Fórmula de Compasso:",
        ["Automático", "2/4", "3/4", "4/4", "6/8", "3/2"],
        index=0,
        help="No modo Automático, o sistema executa inferência hierárquica probabilística. Se conhecer a métrica exata da obra, selecione manualmente."
    )

uploaded_file = st.file_uploader("Escolha um arquivo de áudio", type=["mp3", "wav", "m4a", "flac", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file.getvalue())

    if st.button("✨ Gerar Partitura Completa com Cifras", type="primary"):
        for k in ["result_ready", "midi_bytes", "xml_bytes", "pdf_bytes", "page_images", "info_text", "log_text"]:
            if k in st.session_state:
                del st.session_state[k]

        session_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(uploaded_file.name)[1].lower() or ".wav"
        safe_input_path = UPLOAD_DIR / f"raw_{session_id}{ext}"
        session_dir = UPLOAD_DIR / f"session_{session_id}"
        session_dir.mkdir(exist_ok=True)

        log = TechnicalLogger()
        log.add("===================================")
        log.add("PARTITURA.AI — LOG TÉCNICO DE AUDITORIA")
        log.add("===================================")
        log.add(f"Data/Hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log.add(f"Versão do Aplicativo: {APP_VERSION}")
        log.add(f"Arquivo: {uploaded_file.name} | Session ID: {session_id}")
        log.add(f"Python: {sys.version.split()[0]} | Librosa: {librosa.__version__} | music21: {music21.__version__} | NumPy: {np.__version__}")
        log.add(f"Modo selecionado pelo usuário: {compasso_selecionado}")
        log.add(f"CONFIG_SIGNAL: {CONFIG_SIGNAL}")
        log.add(f"CONFIG_TACTUS: {CONFIG_TACTUS}")
        log.add(f"CONFIG_METER: {CONFIG_METER}")
        log.add(f"CONFIG_CONFIDENCE: {CONFIG_CONFIDENCE}")
        log.add(f"CONFIG_MELODY: {CONFIG_MELODY}")
        log.add(f"CONFIG_HARMONY: {CONFIG_HARMONY}")
        log.add(f"CONFIG_QUANTIZATION: {CONFIG_QUANTIZATION}")

        midi_status = "NÃO EXECUTADO"
        xml_status = "NÃO EXECUTADO"
        pdf_status = "NÃO EXECUTADO"

        with st.spinner("⚡ Processando áudio no Render (compilando LilyPond e IA)... Por favor, aguarde alguns instantes."):
            try:
                ai_model = get_inference_model()

                with open(safe_input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                y_raw, sr = librosa.load(str(safe_input_path), sr=CONFIG_SIGNAL["sample_rate"], mono=False)
                if y_raw.ndim > 1 and y_raw.shape[0] >= 2:
                    orig_channels = y_raw.shape[0]
                    y_mid = 0.5 * (y_raw[0] + y_raw[1])
                    y_side = 0.5 * (y_raw[0] - y_raw[1])
                    mid_rms = float(np.sqrt(np.mean(y_mid ** 2)) + 1e-8)
                    side_rms = float(np.sqrt(np.mean(y_side ** 2)) + 1e-8)
                    ms_ratio = mid_rms / side_rms
                    y_full = (0.65 * y_mid + 0.35 * np.mean(y_raw, axis=0)).astype(np.float32)
                    del y_raw, y_mid, y_side
                else:
                    orig_channels = 1
                    ms_ratio = 1.0
                    y_full = np.asarray(y_raw, dtype=np.float32).reshape(-1)
                    del y_raw

                total_duration = len(y_full) / float(sr)
                log.add(f"Duração: {total_duration:.2f}s | Sample Rate: {sr} Hz | Canais originais: {orig_channels} (Mid/Side Ratio={ms_ratio:.2f})")

                tempo_data = analyze_unified_tempo_and_tatums(y_full, sr, log)
                global_raw_bpm_val = tempo_data["global_raw_bpm"]

                raw_notes, dedup_notes = infer_notes_with_overlap(y_full, sr, session_dir, ai_model, log)
                del y_full
                gc.collect()

                if not dedup_notes:
                    log.add_error("Nenhuma nota musical clara foi detectada pelo Basic Pitch.")
                    st.session_state["log_text"] = log.get_text()
                    st.error("Nenhuma nota musical clara foi detectada neste arquivo de áudio.")
                else:
                    if compasso_selecionado != "Automático":
                        time_sig = compasso_selecionado
                        bpm = round(float(tempo_data["top_tactus"][0]["bpm"]), 1) if tempo_data["top_tactus"] else global_raw_bpm_val
                        compasso_exibicao = f"{time_sig} (Manual)"
                        meter_status = "manual"
                        meter_conf_str = "manual"
                        pulse_ql = {"2/4": 1.0, "3/4": 1.0, "4/4": 1.0, "6/8": 1.5, "3/2": 2.0}.get(time_sig, 1.0)
                        meter_sections = [{"start_sec": 0.0, "time_sig": time_sig}]
                        meter_qualities = {
                            "signal_quality": round(tempo_data["signal_quality"], 3),
                            "tactus_quality": 1.0,
                            "grouping_quality": 1.0,
                            "subdivision_quality": 1.0,
                            "feature_agreement": 1.0,
                            "sequence_stability": 1.0,
                            "bootstrap_stability": 1.0,
                            "decision_quality": 1.0,
                        }
                        log.add(f"\n[METER] Compasso selecionado manualmente: {time_sig} | Tactus associado: {bpm} BPM")
                    else:
                        meter_res = run_hierarchical_meter_inference(tempo_data, dedup_notes, sr, log)
                        bpm = meter_res["chosen_bpm"]
                        time_sig = meter_res["time_sig"]
                        pulse_ql = meter_res["pulse_ql"]
                        meter_status = meter_res["status"]
                        meter_conf_str = meter_res["confidence_label"]
                        meter_sections = meter_res["meter_sections"]
                        meter_qualities = meter_res["qualities"]
                        if meter_status == "detected":
                            compasso_exibicao = f"{time_sig} (Auto - Conf: {meter_conf_str})"
                        else:
                            compasso_exibicao = f"Não determinado (Fallback {time_sig})"

                    beat_times, tatum_times, beat_quality = build_beat_and_tatum_grid(
                        tempo_data["onset_norm"], sr, bpm, total_duration, log
                    )
                    meter_qualities["beat_quality"] = round(beat_quality, 3)

                    num, den = map(int, time_sig.split("/"))
                    bar_length_ql = float(num) * (4.0 / float(den))

                    melody_notes, mel_conf = track_melody_viterbi_a4(dedup_notes, tempo_data["onset_norm"], sr, log)

                    if not melody_notes:
                        log.add_warning("Nenhuma nota atingiu os critérios do rastreador melódico.")
                        st.session_state["log_text"] = log.get_text()
                        st.warning("Foram detectadas notas no áudio, mas nenhuma formou uma linha melódica consistente.")
                    else:
                        midi_path = OUTPUT_DIR / f"partitura_{session_id}.mid"
                        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
                        inst = pretty_midi.Instrument(program=0)
                        for n in melody_notes:
                            inst.notes.append(pretty_midi.Note(
                                velocity=int(n.velocity),
                                pitch=int(n.pitch),
                                start=float(n.start_sec),
                                end=float(n.end_sec)
                            ))
                        pm.instruments.append(inst)
                        pm.write(str(midi_path))
                        midi_status = "OK"

                        quant_events, quant_err_stats = quantize_melody_dp_global(
                            melody_notes, beat_times, pulse_ql, bar_length_ql, log
                        )

                        score = music21.stream.Score()
                        score.insert(0, music21.metadata.Metadata())
                        score.metadata.title = os.path.splitext(uploaded_file.name)[0]
                        score.metadata.composer = "Transcrito por Partitura.ai"

                        part = music21.stream.Part()
                        part.insert(0, music21.clef.TrebleClef())
                        part.insert(0, music21.tempo.MetronomeMark(number=bpm))

                        if len(meter_sections) > 1 and meter_status == "detected":
                            for sec_m in meter_sections:
                                sec_ql = round(seconds_to_musical_position(sec_m["start_sec"], beat_times, pulse_ql=pulse_ql) / bar_length_ql) * bar_length_ql
                                part.insert(float(max(0.0, sec_ql)), music21.meter.TimeSignature(sec_m["time_sig"]))
                        else:
                            part.insert(0, music21.meter.TimeSignature(time_sig))

                        decomposed_count = 0
                        ties_created = 0

                        for idx, ev in enumerate(quant_events):
                            start_ql = ev["start_ql"]
                            dur_ql = round((ev["end_ql"] - start_ql) * 4.0) / 4.0
                            if dur_ql <= 0.0:
                                dur_ql = 0.25

                            if idx + 1 < len(quant_events):
                                max_allowed_ql = round((quant_events[idx + 1]["start_ql"] - start_ql) * 4.0) / 4.0
                                if max_allowed_ql >= 0.25 and dur_ql > max_allowed_ql:
                                    dur_ql = max_allowed_ql

                            dur_ql = max(0.25, min(bar_length_ql, dur_ql))
                            dur_parts = decompose_into_expressible_durations(dur_ql)
                            if len(dur_parts) > 1:
                                decomposed_count += 1

                            curr_off = float(start_ql)
                            for p_i, d_p in enumerate(dur_parts):
                                el = music21.note.Note(ev["pitch"])
                                el.quarterLength = float(d_p)
                                if len(dur_parts) > 1:
                                    if p_i == 0:
                                        el.tie = music21.tie.Tie("start")
                                        ties_created += 1
                                    elif p_i == len(dur_parts) - 1:
                                        el.tie = music21.tie.Tie("stop")
                                    else:
                                        el.tie = music21.tie.Tie("continue")
                                        ties_created += 1
                                part.insert(curr_off, el)
                                curr_off = round((curr_off + d_p) * 4.0) / 4.0

                        log.add(f"Durações compostas decompostas: {decomposed_count} | Ligaduras intra-nota criadas: {ties_created}")

                        detected_key, key_success, key_conf, tonic_pt, mode_pt = detectar_centro_tonal_global(
                            part, y_harm=tempo_data["y_harm"], sr=sr, all_notes=dedup_notes, log=log
                        )
                        part.insert(0, music21.key.KeySignature(detected_key.sharps))

                        part.makeMeasures(inPlace=True)
                        part.makeTies(inPlace=True)
                        corr_pre = auditar_e_sanitizar_duracoes_musicxml(part, log)

                        chords_by_bar, chord_abstentions = analyze_harmony_root_quality_hmm(
                            dedup_notes, tempo_data["y_harm"], sr, beat_times, pulse_ql, bar_length_ql, detected_key, log
                        )

                        del tempo_data
                        gc.collect()

                        measures = list(part.getElementsByClass(music21.stream.Measure))
                        xml_chords_to_insert = []
                        last_mpb_sym = None
                        total_cifras_inseridas = 0

                        for idx_m, m in enumerate(measures):
                            reg_list = chords_by_bar.get(idx_m, [])
                            if not reg_list:
                                continue

                            m_notes = list(m.notes)
                            for reg in reg_list:
                                mpb_sym = reg["mpb_sym"]
                                xml_sym = reg["xml_sym"]
                                off_ql = float(reg["offset_in_bar_ql"])

                                if mpb_sym and (mpb_sym != last_mpb_sym or idx_m % 4 == 0):
                                    last_mpb_sym = mpb_sym
                                    if m_notes:
                                        target_note = min(m_notes, key=lambda nt: abs(float(nt.offset) - off_ql))
                                        target_note.addLyric(mpb_sym)
                                    xml_chords_to_insert.append((m, off_ql, mpb_sym, xml_sym))
                                    total_cifras_inseridas += 1

                        score.insert(0, part)

                        pdf_bytes = None
                        page_images = []
                        try:
                            ly_base = OUTPUT_DIR / f"partitura_{session_id}"
                            generated_pdf = Path(score.write("lily.pdf", fp=str(ly_base)))
                            if generated_pdf.exists():
                                pdf_bytes = generated_pdf.read_bytes()
                                pdf_doc = pdfium.PdfDocument(pdf_bytes)
                                for i in range(len(pdf_doc)):
                                    page = pdf_doc[i]
                                    pil_image = page.render(scale=2).to_pil()
                                    img_buf = io.BytesIO()
                                    pil_image.save(img_buf, format="PNG")
                                    page_images.append(img_buf.getvalue())
                                generated_pdf.unlink(missing_ok=True)
                                pdf_status = f"OK ({len(page_images)} páginas)"
                            for f in OUTPUT_DIR.glob(f"partitura_{session_id}*"):
                                if f.suffix not in [".mid", ".musicxml"]:
                                    f.unlink(missing_ok=True)
                        except Exception as pdf_err:
                            pdf_status = f"ERRO ({pdf_err})"
                            log.add_warning(f"Falha na geração do PDF via LilyPond: {pdf_err}")
                            st.warning(
                                f"O PDF não pôde ser gerado pelo LilyPond ({pdf_err}). "
                                "O processamento do MIDI e do MusicXML continuará normalmente."
                            )

                        for m_obj, off_ql, mpb_sym, xml_sym in xml_chords_to_insert:
                            try:
                                cs_obj = music21.harmony.ChordSymbol(xml_sym)
                                cs_obj.writeAsChord = False
                                m_obj.insert(off_ql, cs_obj)
                            except Exception as cs_err:
                                log.add_warning(
                                    f"Falha em ChordSymbol no compasso {m_obj.number} (offset {off_ql}): "
                                    f"mpb='{mpb_sym}', xml='{xml_sym}' -> {cs_err}"
                                )

                        corr_post = auditar_e_sanitizar_duracoes_musicxml(score, log)

                        xml_path = OUTPUT_DIR / f"partitura_{session_id}.musicxml"
                        try:
                            score.write("musicxml", fp=str(xml_path))
                            xml_status = "OK"
                        except Exception as xml_err:
                            xml_status = f"ERRO ({xml_err})"
                            log.add_error(f"Falha ao exportar MusicXML: {xml_err}")
                            raise RuntimeError(f"Falha na exportação MusicXML: {xml_err}")

                        tom_exibicao = f"{tonic_pt} ({mode_pt})" if key_success else "Não determinado"

                        midi_bytes = midi_path.read_bytes()
                        xml_bytes = xml_path.read_bytes()

                        midi_path.unlink(missing_ok=True)
                        xml_path.unlink(missing_ok=True)

                        log.add("\n===================================")
                        log.add("PARTITURA.AI — RESUMO FINAL")
                        log.add("===================================")
                        log.add(f"Arquivo: {uploaded_file.name}")
                        log.add(f"Duração: {total_duration:.2f}s")
                        log.add(f"BPM bruto: {global_raw_bpm_val:.2f} | Tactus principal: {bpm:.1f} BPM | Tactus Quality: {meter_qualities['tactus_quality']:.3f}")
                        log.add(
                            f"Compasso: {compasso_exibicao} | Status: {meter_status} | "
                            f"Decision Confidence: {meter_qualities['decision_quality']:.3f} | "
                            f"Bootstrap Stability: {meter_qualities['bootstrap_stability']:.2f} | "
                            f"Feature Agreement: {meter_qualities['feature_agreement']:.2f}"
                        )
                        log.add(f"Meter Sections: {meter_sections}")
                        log.add(f"Tom: {tonic_pt} | Modo: {mode_pt} | Key Confidence: {key_conf:.3f}")
                        log.add(f"Basic Pitch raw: {len(raw_notes)} | Melody notes: {len(melody_notes)} | Melody Confidence: {mel_conf:.3f}")
                        log.add(f"Harmonic regions cifradas: {len(chords_by_bar)} | Chords inserted: {total_cifras_inseridas} | Chord abstentions: {chord_abstentions}")
                        log.add(f"Quantization error P50/P90/P95: {quant_err_stats['p50']} / {quant_err_stats['p90']} / {quant_err_stats['p95']} QL")
                        log.add(f"MIDI: {midi_status} | MusicXML: {xml_status} | PDF: {pdf_status}")
                        log.add(f"Durações realmente corrigidas: {corr_pre + corr_post}")
                        log.add(f"Warnings: {len(log.warnings)} | Erros: {len(log.errors)}")

                        st.session_state["result_ready"] = True
                        st.session_state["midi_bytes"] = midi_bytes
                        st.session_state["xml_bytes"] = xml_bytes
                        st.session_state["pdf_bytes"] = pdf_bytes
                        st.session_state["page_images"] = page_images
                        st.session_state["log_text"] = log.get_text()
                        st.session_state["info_text"] = (
                            f"🎵 Andamento: {bpm} BPM | Compasso: {compasso_exibicao} | "
                            f"Tom: {tom_exibicao} | Clave de Sol + Cifras MPB"
                        )

            except Exception as e:
                log.add_error(f"Exceção crítica durante o processamento: {str(e)}")
                st.session_state["log_text"] = log.get_text()
                st.error(f"Erro crítico durante o processamento do arquivo: {str(e)}")
            finally:
                if safe_input_path.exists():
                    try:
                        safe_input_path.unlink()
                    except Exception:
                        pass
                if session_dir.exists():
                    try:
                        shutil.rmtree(session_dir)
                    except Exception:
                        pass
                for leftover in OUTPUT_DIR.glob(f"*{session_id}*"):
                    try:
                        leftover.unlink(missing_ok=True)
                    except Exception:
                        pass
                gc.collect()

    if st.session_state.get("result_ready"):
        st.success("✨ Partitura gerada com sucesso!")
        st.info(st.session_state["info_text"])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.session_state.get("pdf_bytes"):
                st.download_button(
                    label="📄 Baixar Arquivo PDF",
                    data=st.session_state["pdf_bytes"],
                    file_name="partitura_com_cifras.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("PDF indisponível.")
        with col2:
            st.download_button(
                label="🎼 Baixar MusicXML",
                data=st.session_state["xml_bytes"],
                file_name="partitura_com_cifras.musicxml",
                mime="application/vnd.recordare.musicxml+xml",
                use_container_width=True
            )
        with col3:
            st.download_button(
                label="📥 Baixar Arquivo MIDI",
                data=st.session_state["midi_bytes"],
                file_name="partitura_com_cifras.mid",
                mime="audio/midi",
                use_container_width=True
            )
        with col4:
            if st.session_state.get("log_text"):
                st.download_button(
                    label="📋 Baixar Log Técnico",
                    data=st.session_state["log_text"],
                    file_name="partitura_ai_log.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        st.markdown("---")
        if st.button("🔄 Enviar Outro Arquivo de Áudio"):
            st.session_state.clear()
            st.rerun()

        if st.session_state.get("page_images"):
            st.markdown("### 🎼 Visualização da Partitura com Cifras")
            for idx, img_bytes in enumerate(st.session_state["page_images"]):
                st.image(img_bytes, caption=f"Página {idx + 1}", use_column_width=True)
    elif st.session_state.get("log_text"):
        st.download_button(
            label="📋 Baixar Log Técnico do Erro",
            data=st.session_state["log_text"],
            file_name="partitura_ai_log.txt",
            mime="text/plain"
        )
