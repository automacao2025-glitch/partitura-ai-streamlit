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

APP_VERSION = "A4.3-Dual-Level-Bar-Phrase-Evidence"

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
    # Seleção tardia: não elimina cedo demais candidatos metricamente plausíveis.
    "joint_min_candidates": 5,
    "joint_max_candidates": 8,
    "joint_keep_ratio": 0.78,
    # O prior perceptual é apenas desempate; a evidência acústica domina.
    "w_acoustic": 0.54,
    "w_alignment": 0.38,
    "w_perceptual_prior": 0.08,
}

CONFIG_METER = {
    # Pesos de evidência. Todos são multiplicados pela reliability da própria feature.
    "w_periodicity": 0.20,
    "w_accent": 0.18,
    "w_structure": 0.18,
    "w_subdivision": 0.16,
    "w_harmonic_change": 0.10,
    "w_recurrence": 0.10,
    "w_bass": 0.08,
    # O motor A4.2 separa TATUM -> PULSE -> BAR; não usa um threshold isolado para 2/4 x 4/4.
    "ternary_phase_bins": 12,
    "viterbi_self_transition": 0.84,
    "viterbi_section_min_windows": 2,
    # Temperatura da marginalização de tactus. Menor = privilegia evidência forte sem descartar rivais plausíveis.
    "tactus_softmax_temperature": 0.085,
    # Priors estruturais fracos, usados só como regularização contínua.
    "recurrence_2bar_center": 0.72,
    "recurrence_2bar_scale": 0.075,
}

CONFIG_DUAL_BAR = {
    # Especialista binário 2/4 x 4/4: não muda 3/4 e 6/8.
    # Quatro janelas representativas; as features são obtidas do áudio, nunca da referência.
    "enabled": True,
    "analysis_sr": 11025,
    "analysis_window_sec": 28.0,
    "analysis_positions": (0.20, 0.45, 0.70, 0.87),
    "analysis_hop": 256,
    "n_mels": 16,
    "min_valid_windows": 3,
    "binary_family_min_mass": 0.37,
    # Evidence favorável a 4/4 em two-feel: recorrência de 16 vs 8 beats.
    # Soft gate + consistência entre janelas para evitar um único trecho anômalo.
    "phrase_negative_center": -0.095,
    "phrase_negative_scale": 0.018,
    "window_negative_cutoff": -0.090,
    "negative_fraction_center": 0.57,
    "negative_fraction_scale": 0.075,
    "target_four_share": 0.80,
    "max_blend": 0.90,
    # Evitar sobreconfiança: evidência 2/4 x 4/4 discordante permite abstention.
    "conflict_min_support": 0.45,
    "conflict_margin_penalty": 0.020,
}

CONFIG_CONFIDENCE = {
    "bootstrap_iterations": 48,
    "min_decision_quality": 0.48,
    "min_bootstrap_stability": 0.56,
    "min_feature_agreement": 0.36,
    "min_grouping_quality": 0.035,
    "min_subdivision_quality": 0.055,
    "min_decision_specific_quality": 0.055,
    "base_margin_threshold": 0.045,
    "min_margin_abs": 0.045,
    "max_margin_abs": 0.150,
    # Confiança do compasso e confiança do tactus são independentes no A4.2.
    "min_meter_confidence": 0.50,
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
    "hmm_stay_prob": 0.70,
    "tonal_prior_weight": 0.12,
    # Evidência local forte não pode ser apagada por oversmoothing do HMM.
    "local_override_root_conf": 0.72,
    "local_override_quality_conf": 0.64,
    "local_override_margin": 0.075,
    "local_override_chord_conf": 0.68,
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

        # Features pesadas independentes do tactus: calcula uma vez e reutiliza.
        try:
            tg_seg = librosa.feature.tempogram(onset_envelope=env_seg, sr=sr, hop_length=hop)
            tg_ac = np.mean(tg_seg, axis=1)
            tg_ac = tg_ac / (np.max(tg_ac) + 1e-8)
        except Exception:
            tg_ac = np.zeros(max(2, len(env_seg)), dtype=float)
        try:
            s0_samples = max(0, int(f0 * hop))
            s1_samples = min(len(y_harm), int(f1 * hop))
            chroma_seg = librosa.feature.chroma_cqt(y=y_harm[s0_samples:s1_samples], sr=sr, hop_length=hop)
        except Exception:
            chroma_seg = None

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
            "tempogram_ac": tg_ac,
            "chroma": chroma_seg,
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
            errs = [abs(cand - s["bpm"] * r) / max(cand, 1e-8) for r in CONFIG_TACTUS["ratios"]]
            seg_align.append(max(0.0, 1.0 - min(errs) * 4.5))
        align_sc = float(np.mean(seg_align)) if seg_align else 0.5

        perc_prior = math.exp(-0.5 * ((cand - CONFIG_TACTUS["perceptual_center_bpm"]) / CONFIG_TACTUS["perceptual_sigma_bpm"]) ** 2)
        fam_score = (
            CONFIG_TACTUS["w_acoustic"] * ac_sc +
            CONFIG_TACTUS["w_alignment"] * align_sc +
            CONFIG_TACTUS["w_perceptual_prior"] * perc_prior
        )
        tactus_family.append({
            "bpm": cand,
            "score": round(fam_score, 4),
            "ac_score": round(ac_sc, 4),
            "align_score": round(align_sc, 4),
            "perceptual_prior": round(perc_prior, 4),
        })

    tactus_family.sort(key=lambda x: x["score"], reverse=True)
    top_tactus_candidates = tactus_family[:CONFIG_TACTUS["joint_max_candidates"]]

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
def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -30.0, 30.0))
    return float(1.0 / (1.0 + math.exp(-x)))


def _robust_unit(x):
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = np.percentile(arr, [10, 90])
    if hi <= lo + 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _shape_similarity(values, template):
    """Similaridade de forma (0..1), removendo o nível DC para não premiar vetores quase planos."""
    a = np.asarray(values, dtype=float)
    b = np.asarray(template, dtype=float)
    if a.size != b.size or a.size < 2:
        return 0.5
    a = a - np.mean(a)
    b = b - np.mean(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.5
    corr = float(np.dot(a, b) / (na * nb))
    return float(np.clip(0.5 + 0.5 * corr, 0.0, 1.0))


def _best_bar_template(series, template):
    """Retorna score, fase e médias por papel métrico para um template de barra."""
    x = np.asarray(series, dtype=float)
    tpl = np.asarray(template, dtype=float)
    n = len(tpl)
    if len(x) < max(6, n * 3):
        return 0.5, 0, np.full(n, np.mean(x) if len(x) else 0.0)
    best = (-1.0, 0, None)
    idx = np.arange(len(x))
    for phase in range(n):
        roles = (idx - phase) % n
        means = np.array([np.mean(x[roles == j]) if np.any(roles == j) else 0.0 for j in range(n)])
        shape = _shape_similarity(means, tpl)
        scale = np.std(means) + 0.08
        contrast = (means[0] - np.mean(means[1:])) / scale
        contrast_sc = _sigmoid(1.15 * contrast)
        score = 0.62 * shape + 0.38 * contrast_sc
        if score > best[0]:
            best = (float(score), int(phase), means)
    return best


def _compute_pulse_phase_features(env, beat_frames):
    """
    Mede a organização *dentro* do pulso candidato.
    Isso é condicionado ao tactus: o mesmo áudio pode ser binário em um nível e ternário em outro.
    """
    n_bins = int(CONFIG_METER["ternary_phase_bins"])
    env = np.asarray(env, dtype=float)
    beat_frames = np.asarray(beat_frames, dtype=int)
    if len(beat_frames) < 5:
        return {
            "p_bin": 0.5, "p_tern": 0.5, "reliability": 0.0, "ternary_repeat": 0.0,
            "half_ratio": 1.0, "third_ratio": 1.0, "compound_support": 0.5,
        }

    pulse_hists = []
    tern_wins = 0
    valid = 0
    beat_start_strength = []
    half_strength = []
    third_strength = []

    # Templates internos: binário = meio; ternário = 1/3 + 2/3.
    tpl_bin = np.zeros(n_bins, dtype=float)
    tpl_tern = np.zeros(n_bins, dtype=float)
    for k, v in [(n_bins // 2, 1.0), (n_bins // 2 - 1, 0.35), (n_bins // 2 + 1, 0.35)]:
        tpl_bin[k % n_bins] = max(tpl_bin[k % n_bins], v)
    for base in (n_bins // 3, (2 * n_bins) // 3):
        tpl_tern[base] = 1.0
        tpl_tern[(base - 1) % n_bins] = max(tpl_tern[(base - 1) % n_bins], 0.30)
        tpl_tern[(base + 1) % n_bins] = max(tpl_tern[(base + 1) % n_bins], 0.30)

    for a, b in zip(beat_frames[:-1], beat_frames[1:]):
        if b - a < max(6, n_bins // 2):
            continue
        edges = np.linspace(a, b, n_bins + 1).astype(int)
        h = np.zeros(n_bins, dtype=float)
        for i in range(n_bins):
            lo = max(0, int(edges[i]))
            hi = min(len(env), max(lo + 1, int(edges[i + 1])))
            h[i] = float(np.max(env[lo:hi])) if hi > lo else 0.0
        mx = float(np.max(h))
        if mx <= 1e-8:
            continue
        hn = h / mx
        pulse_hists.append(hn)
        valid += 1
        # Ignora bin 0 na comparação interna; ele representa o próprio pulso.
        h_internal = hn.copy()
        h_internal[0] = 0.0
        sb = cosine_sim(h_internal, tpl_bin)
        st = cosine_sim(h_internal, tpl_tern)
        if st > sb + 0.035:
            tern_wins += 1
        beat_start_strength.append(float(np.mean(hn[:2])))
        half_strength.append(float(np.mean(hn[max(0, n_bins//2-1):min(n_bins, n_bins//2+2)])))
        t1 = float(np.mean(hn[max(0, n_bins//3-1):min(n_bins, n_bins//3+2)]))
        t2 = float(np.mean(hn[max(0, 2*n_bins//3-1):min(n_bins, 2*n_bins//3+2)]))
        third_strength.append(min(t1, t2))

    if not pulse_hists:
        return {
            "p_bin": 0.5, "p_tern": 0.5, "reliability": 0.0, "ternary_repeat": 0.0,
            "half_ratio": 1.0, "third_ratio": 1.0, "compound_support": 0.5,
        }

    hmean = np.mean(np.asarray(pulse_hists), axis=0)
    h_internal = hmean.copy()
    h_internal[0] = 0.0
    sim_bin = max(1e-6, cosine_sim(h_internal, tpl_bin))
    sim_tern = max(1e-6, cosine_sim(h_internal, tpl_tern))
    p_tern = float(sim_tern / (sim_bin + sim_tern))
    p_bin = 1.0 - p_tern
    repeat = tern_wins / float(max(1, valid))

    start = float(np.mean(beat_start_strength)) + 1e-8
    half_ratio = float(np.clip(np.mean(half_strength) / start, 0.0, 2.0))
    third_ratio = float(np.clip(np.mean(third_strength) / start, 0.0, 2.0))

    # Em 6/8 real, a energia em 1/3 e 2/3 deve ser repetida e relativamente maior que a de 1/2.
    phase_margin = float(np.clip((third_ratio - half_ratio + 0.35) / 0.70, 0.0, 1.0))
    repeat_sc = float(np.clip((repeat - 0.18) / 0.52, 0.0, 1.0))
    tern_sc = float(np.clip((p_tern - 0.46) / 0.24, 0.0, 1.0))
    compound_support = float(np.clip(0.40 * tern_sc + 0.35 * repeat_sc + 0.25 * phase_margin, 0.0, 1.0))
    rel = float(np.clip(0.55 * binary_reliability(p_tern) + 0.45 * abs(third_ratio - half_ratio), 0.0, 1.0))

    return {
        "p_bin": p_bin,
        "p_tern": p_tern,
        "reliability": rel,
        "ternary_repeat": repeat,
        "half_ratio": half_ratio,
        "third_ratio": third_ratio,
        "compound_support": compound_support,
    }


def _evaluate_segment_for_tactus(seg, y_harm_seg, y_perc_seg, sr, cand_bpm, onset_peaks_times, seg_notes):
    hop = CONFIG_SIGNAL["hop_length"]
    env = np.asarray(seg["onset_env"], dtype=float)
    if len(env) < 16:
        return None

    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, bpm=cand_bpm)
    beats = np.asarray(beats, dtype=int)
    if len(beats) < 6:
        step_f = max(2, int(round((60.0 / cand_bpm * sr) / hop)))
        beats = np.arange(0, len(env), step_f, dtype=int)
    if len(beats) < 6:
        return None

    beat_times_local = librosa.frames_to_time(beats + seg["frame_start"], sr=sr, hop_length=hop)
    frames_per_beat = max(2, int(round((60.0 / cand_bpm * sr) / hop)))

    # ---------------- periodicidade 2/3/4 ----------------
    ac = seg.get("tempogram_ac")
    if ac is None or len(ac) < 2:
        tg = librosa.feature.tempogram(onset_envelope=env, sr=sr, hop_length=hop)
        ac = np.mean(tg, axis=1)
        ac = ac / (np.max(ac) + 1e-8)

    def _lag_e(mult):
        lg = int(round(frames_per_beat * mult))
        if 1 <= lg < len(ac):
            return float(np.max(ac[max(1, lg - 1):min(len(ac), lg + 2)]))
        return 0.0

    per_abs = {2: _lag_e(2.0), 3: _lag_e(3.0), 4: _lag_e(4.0)}
    per_sum = sum(per_abs.values()) + 1e-8
    per_probs = {k: v / per_sum for k, v in per_abs.items()}
    per_rel = multiclass_reliability(per_probs)

    # ---------------- ataques nos beats ----------------
    b_acc = np.array([
        float(np.max(env[max(0, int(b) - 2):min(len(env), int(b) + 3)]))
        for b in beats
    ], dtype=float)
    b_acc_u = _robust_unit(b_acc)
    b_z = (b_acc - np.mean(b_acc)) / (np.std(b_acc) + 1e-8)

    # ---------------- mudança harmônica e recorrência absolutas ----------------
    try:
        chroma = seg.get("chroma")
        if chroma is None:
            chroma = librosa.feature.chroma_cqt(y=y_harm_seg, sr=sr, hop_length=hop)
        beats_sync = beats[(beats >= 0) & (beats < chroma.shape[1])]
        if len(beats_sync) < 2:
            raise ValueError("Poucos beats válidos")
        c_sync = librosa.util.sync(chroma, beats_sync)
        c_norm = c_sync / (np.linalg.norm(c_sync, axis=0, keepdims=True) + 1e-8)
        h_diff = np.sqrt(np.sum(np.diff(c_norm, axis=1) ** 2, axis=0))
        h_diff = np.pad(h_diff, (1, 0), mode="edge")
        h_diff = h_diff / (np.max(h_diff) + 1e-8)
    except Exception:
        c_norm = np.zeros((12, len(beats)))
        h_diff = np.zeros(len(beats))

    # ---------------- baixo e ataques simbólicos (Basic Pitch) ----------------
    bass_beat_env = np.zeros(len(beat_times_local), dtype=float)
    note_attack_env = np.zeros(len(beat_times_local), dtype=float)
    if seg_notes:
        pitches = np.array([n.pitch for n in seg_notes], dtype=float)
        high_cut = float(np.percentile(pitches, 60)) if len(pitches) else 64.0
        for bi, bt in enumerate(beat_times_local):
            near = [n for n in seg_notes if abs(n.start_sec - bt) <= 0.14]
            lows = [n for n in near if n.pitch <= 55]
            highs = [n for n in near if n.pitch >= high_cut]
            if lows:
                bass_beat_env[bi] = max(n.velocity / 127.0 for n in lows)
            if highs:
                note_attack_env[bi] = max((n.velocity / 127.0) * min(1.0, 0.35 + n.duration_sec) for n in highs)

    # ---------------- templates 2/3/4 por fonte ----------------
    templates_acc = {
        2: np.array([1.0, -0.82]),
        3: np.array([1.0, -0.52, -0.48]),
        4: np.array([1.0, -0.62, 0.30, -0.58]),
    }
    acc_probs, harm_probs, bass_probs, best_phases = {}, {}, {}, {}
    acc_abs = {}
    for p_n, tpl in templates_acc.items():
        best = None
        for phase in range(p_n):
            idx_seq = np.arange(len(beats))
            roles = (idx_seq - phase) % p_n
            a_val = float(np.mean(b_z * tpl[roles]))
            down_i = idx_seq[roles == 0]
            h_val = float(np.mean(h_diff[down_i])) if len(down_i) else 0.0
            b_val = float(np.mean(bass_beat_env[down_i])) if len(down_i) else 0.0
            n_val = float(np.mean(note_attack_env[down_i])) if len(down_i) else 0.0
            comb = 0.48 * a_val + 0.23 * h_val + 0.17 * b_val + 0.12 * n_val
            if best is None or comb > best[0]:
                best = (comb, phase, a_val, h_val, b_val)
        _, phase, a_val, h_val, b_val = best
        acc_abs[p_n] = _sigmoid(2.0 * a_val)
        acc_probs[p_n] = acc_abs[p_n]
        harm_probs[p_n] = max(0.0, h_val)
        bass_probs[p_n] = max(0.0, b_val)
        best_phases[p_n] = int(phase)

    def _norm_dict(d):
        z = sum(max(0.0, float(v)) for v in d.values()) + 1e-8
        return {k: max(0.0, float(v)) / z for k, v in d.items()}

    acc_probs = _norm_dict(acc_probs)
    harm_probs = _norm_dict(harm_probs)
    bass_probs = _norm_dict(bass_probs)
    acc_rel = multiclass_reliability(acc_probs)
    harm_rel = multiclass_reliability(harm_probs)
    bass_rel = multiclass_reliability(bass_probs)

    # ---------------- recorrência absoluta e relativa ----------------
    rec_abs = {2: 0.50, 3: 0.50, 4: 0.50}
    if c_norm.shape[1] >= 10:
        sim_m = np.dot(c_norm.T, c_norm)
        for lag in (2, 3, 4):
            vals = [sim_m[i, i + lag] for i in range(sim_m.shape[0] - lag)]
            rec_abs[lag] = float(np.clip(np.mean(vals), 0.0, 1.0)) if vals else 0.50
    rec_probs = _norm_dict(rec_abs)
    rec_rel = multiclass_reliability(rec_probs)

    # ---------------- força estrutural combinada por beat ----------------
    L = min(len(b_acc_u), len(h_diff), len(bass_beat_env), len(note_attack_env))
    if L <= 0:
        return None
    h_u = _robust_unit(h_diff[:L])
    b_u = _robust_unit(bass_beat_env[:L])
    n_u = _robust_unit(note_attack_env[:L])
    struct = 0.46 * b_acc_u[:L] + 0.24 * h_u + 0.18 * b_u + 0.12 * n_u

    struct_scores = {}
    struct_phases = {}
    role_means = {}
    for g, tpl in {
        2: [1.0, 0.30],
        3: [1.0, 0.28, 0.33],
        4: [1.0, 0.28, 0.62, 0.28],
    }.items():
        sc, ph, means = _best_bar_template(struct, tpl)
        struct_scores[g] = sc
        struct_phases[g] = ph
        role_means[g] = means
    struct_probs = _norm_dict(struct_scores)
    struct_rel = multiclass_reliability(struct_probs)

    # ---------------- 2/4 x 4/4: hierarquia do tempo 1 contra tempo 3 ----------------
    means4 = np.asarray(role_means[4], dtype=float)
    sim_4 = _shape_similarity(means4, [1.0, 0.28, 0.62, 0.28])
    sim_2repeat = _shape_similarity(means4, [1.0, 0.28, 1.0, 0.28])
    hierarchy_margin = float(sim_4 - sim_2repeat)
    # Recorrência absoluta de 2 beats alta é evidência adicional de barra curta (2/4).
    rec2_simple = _sigmoid((rec_abs[2] - CONFIG_METER["recurrence_2bar_center"]) / CONFIG_METER["recurrence_2bar_scale"])
    four_hierarchy = float(np.clip(0.58 * _sigmoid(9.0 * hierarchy_margin) + 0.42 * (1.0 - rec2_simple), 0.0, 1.0))
    two_hierarchy = 1.0 - four_hierarchy
    hierarchy_quality = float(np.clip(abs(four_hierarchy - 0.5) * 2.0, 0.0, 1.0))

    # ---------------- subdivisão condicionada ao tactus ----------------
    phase = _compute_pulse_phase_features(env, beats)
    p_sub_bin = phase["p_bin"]
    p_sub_tern = phase["p_tern"]
    sub_rel = phase["reliability"]
    tern_rep = phase["ternary_repeat"]
    compound_support = phase["compound_support"]
    simple_support = 1.0 - compound_support

    # ---------------- grouping 2/3/4 com reliability real ----------------
    wm = CONFIG_METER
    eps = 1e-5
    eff = {
        "periodicity": wm["w_periodicity"] * max(eps, per_rel),
        "accent": wm["w_accent"] * max(eps, acc_rel),
        "structure": wm["w_structure"] * max(eps, struct_rel),
        "harmonic": wm["w_harmonic_change"] * max(eps, harm_rel),
        "recurrence": wm["w_recurrence"] * max(eps, rec_rel),
        "bass": wm["w_bass"] * max(eps, bass_rel),
    }
    wsum = sum(eff.values()) + 1e-8
    grp_scores = {}
    for g in (2, 3, 4):
        grp_scores[g] = (
            eff["periodicity"] * per_probs[g] +
            eff["accent"] * acc_probs[g] +
            eff["structure"] * struct_probs[g] +
            eff["harmonic"] * harm_probs[g] +
            eff["recurrence"] * rec_probs[g] +
            eff["bass"] * bass_probs[g]
        ) / wsum
    grp_scores = _norm_dict(grp_scores)
    grouping_quality = multiclass_reliability(grp_scores)

    # ---------------- time signatures: PULSE -> BAR ----------------
    # Recorrência absoluta a cada 2 beats é um indício importante de barra duple curta.
    rec2_simple = _sigmoid((rec_abs[2] - CONFIG_METER["recurrence_2bar_center"]) / CONFIG_METER["recurrence_2bar_scale"])

    # 2/4: grouping duple + subdivisão simples + recorrência consistente a cada 2 beats.
    sc_2_4 = (
        grp_scores[2]
        * (0.72 + 0.28 * simple_support)
        * (0.78 + 0.22 * two_hierarchy)
        * (0.78 + 0.22 * rec2_simple)
    )

    # 4/4: pode nascer de grouping=4 OU de grouping=2 quando o segundo grupo se comporta como tempo 3.
    # Recorrência 2-beat baixa evita interpretar meia-barra como novo compasso.
    # Se a recorrência a cada 2 beats é fraca e o tempo 3 parece secundário,
    # parte da evidência de grouping=2 pertence na realidade à meia-barra de um 4/4.
    four_bar_base = grp_scores[4] + 0.85 * grp_scores[2] * four_hierarchy * (1.0 - rec2_simple)
    sc_4_4 = (
        four_bar_base
        * (0.76 + 0.24 * simple_support)
        * (0.74 + 0.26 * four_hierarchy)
    )

    # 6/8: dois pulsos por barra, cada pulso com três subdivisões internas repetidas.
    sc_6_8 = (
        grp_scores[2]
        * (0.30 + 0.70 * compound_support)
        * (0.80 + 0.20 * (1.0 - rec2_simple))
    )

    # 3/4: três pulsos por barra. O score NÃO recebe bônus artificial de acento absoluto;
    # ele precisa existir no próprio grouping/structure para evitar falso 3/4 em canções duples.
    three_struct = float(np.clip(0.78 * grp_scores[3] + 0.22 * struct_probs[3], 0.0, 1.0))
    sc_3_4 = three_struct * (0.72 + 0.28 * simple_support)

    # 3/2 compartilha grouping de 3, mas exige nível de tactus mais longo. BPM é só evidência auxiliar.
    slow_level = float(np.clip((82.0 - cand_bpm) / 55.0, 0.0, 1.0))
    long_period_support = float(np.clip(0.50 * slow_level + 0.50 * per_rel, 0.0, 1.0))
    sc_3_2 = three_struct * (0.14 + 0.34 * long_period_support)

    ts_probs = _norm_dict({"2/4": sc_2_4, "3/4": sc_3_4, "4/4": sc_4_4, "6/8": sc_6_8, "3/2": sc_3_2})

    top_g = max(grp_scores.items(), key=lambda kv: kv[1])[0]
    votes = [
        max(per_probs.items(), key=lambda kv: kv[1])[0] == top_g,
        max(acc_probs.items(), key=lambda kv: kv[1])[0] == top_g,
        max(struct_probs.items(), key=lambda kv: kv[1])[0] == top_g,
        max(harm_probs.items(), key=lambda kv: kv[1])[0] == top_g,
        max(rec_probs.items(), key=lambda kv: kv[1])[0] == top_g,
        max(bass_probs.items(), key=lambda kv: kv[1])[0] == top_g,
    ]
    feature_agreement = float(np.mean(votes))

    phases = {
        "2/4": struct_phases[2],
        "6/8": struct_phases[2],
        "3/4": struct_phases[3],
        "3/2": struct_phases[3],
        "4/4": struct_phases[4],
    }

    return {
        "tactus_bpm": cand_bpm,
        "acc_abs": acc_abs,
        "grp_scores": grp_scores,
        "grouping_quality": grouping_quality,
        "sub_bin": p_sub_bin,
        "sub_tern": p_sub_tern,
        "sub_rel": sub_rel,
        "tern_rep": tern_rep,
        "compound_support": compound_support,
        "simple_support": simple_support,
        "half_ratio": phase["half_ratio"],
        "third_ratio": phase["third_ratio"],
        "four_hierarchy": four_hierarchy,
        "two_hierarchy": two_hierarchy,
        "hierarchy_quality": hierarchy_quality,
        "rec_abs": rec_abs,
        "struct_probs": struct_probs,
        "ts_probs": ts_probs,
        "phases": phases,
        "reliabilities": {
            "periodicity": per_rel,
            "accent": acc_rel,
            "structure": struct_rel,
            "subdivision": sub_rel,
            "harmonic_change": harm_rel,
            "recurrence": rec_rel,
            "bass": bass_rel,
        },
        "feature_agreement": feature_agreement,
        "beat3_quality": hierarchy_quality,
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


def _select_joint_tactus_candidates(tempo_data):
    family = list(tempo_data.get("tactus_family", []))
    if not family:
        return []
    family.sort(key=lambda x: x["score"], reverse=True)
    min_n = int(CONFIG_TACTUS["joint_min_candidates"])
    max_n = int(CONFIG_TACTUS["joint_max_candidates"])
    keep_ratio = float(CONFIG_TACTUS["joint_keep_ratio"])
    top_score = max(1e-8, float(family[0]["score"]))
    selected = []
    for cand in family:
        if len(selected) < min_n or (len(selected) < max_n and float(cand["score"]) >= top_score * keep_ratio):
            selected.append(cand)
        if len(selected) >= max_n:
            break
    return selected


def _stabilize_meter_path(path, min_run=2):
    if len(path) < 3 or min_run <= 1:
        return list(path)
    out = list(path)
    changed = True
    passes = 0
    while changed and passes < 2:
        changed = False
        passes += 1
        runs = []
        i = 0
        while i < len(out):
            j = i + 1
            while j < len(out) and out[j] == out[i]:
                j += 1
            runs.append((i, j, out[i]))
            i = j
        for r_i, (a, b, val) in enumerate(runs):
            if (b - a) >= min_run:
                continue
            left = runs[r_i - 1][2] if r_i > 0 else None
            right = runs[r_i + 1][2] if r_i + 1 < len(runs) else None
            replacement = left if left is not None else right
            if left is not None and right is not None and left == right:
                replacement = left
            if replacement is not None and replacement != val:
                for k in range(a, b):
                    out[k] = replacement
                changed = True
    return out


def _phrase_8v16_evidence(window_deltas, tempo_cv=0.0):
    """Retorna evidência contínua de barra longa, nunca um rótulo fechado.

    Usamos a mediana e a repetição entre janelas para não confundir uma única
    quebra de frase com estrutura 4/4. A magnitude não é probabilidade calibrada.
    """
    cfg = CONFIG_DUAL_BAR
    v = np.asarray(window_deltas, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < cfg["min_valid_windows"]:
        return {"support_4": 0.0, "valid": False, "median": None,
                "mean": None, "negative_fraction": 0.0, "window_count": int(v.size),
                "tempo_cv": float(tempo_cv)}
    median = float(np.median(v))
    neg_fraction = float(np.mean(v < cfg["window_negative_cutoff"]))
    depth = _sigmoid((cfg["phrase_negative_center"] - median) /
                     cfg["phrase_negative_scale"])
    persistent = _sigmoid((neg_fraction - cfg["negative_fraction_center"]) /
                          cfg["negative_fraction_scale"])
    # Se o rastreamento do pulso oscila, há mais incerteza sobre a comparação
    # exata de 8/16 beats. Não transforma BPM em rótulo de compasso.
    tempo_rel = float(np.clip(1.0 - max(0.0, tempo_cv - 0.04) / 0.28, 0.40, 1.0))
    support = float(np.clip(depth * persistent * tempo_rel, 0.0, 1.0))
    return {"support_4": support, "valid": True, "median": median,
            "mean": float(np.mean(v)), "negative_fraction": neg_fraction,
            "window_count": int(v.size), "tempo_cv": float(tempo_cv)}


def _dual_bar_phrase_profile(tempo_data, sr):
    """Extrai uma feature musical de frase longa com baixo custo de RAM.

    Trabalha em pequenos trechos, usa o pulso acústico bruto e não requer
    Basic Pitch, modelos extras, matriz N² nem dependências além de Librosa/NumPy.
    """
    cfg = CONFIG_DUAL_BAR
    full_len = int(len(tempo_data["y_harm"]))
    duration = float(tempo_data.get("total_duration", full_len / sr))
    target_sr = int(cfg["analysis_sr"])
    hop = int(cfg["analysis_hop"])
    win_sec = float(cfg["analysis_window_sec"])
    window_deltas = []
    details = []
    raw_bpms = [float(s.get("bpm", 0.0)) for s in tempo_data.get("segments", [])
                if float(s.get("bpm", 0.0)) > 0.0]
    tempo_cv = float(np.std(raw_bpms) / max(np.mean(raw_bpms), 1e-8)) if raw_bpms else 0.5

    for pos in cfg["analysis_positions"]:
        try:
            start = max(0.0, float(duration)*float(pos) - win_sec / 2.0)
            start = min(start, max(0.0, duration - win_sec - 1.0))
            a = max(0, int(start*sr))
            b = min(full_len, a + int(win_sec*sr))
            if b - a < int(18.0*sr):
                continue
            # Reusa os componentes HPSS: só cria um trecho mono curto.
            wave = np.asarray(tempo_data["y_harm"][a:b] +
                              tempo_data["y_perc"][a:b], dtype=np.float32)
            if sr != target_sr:
                wave = librosa.resample(wave, orig_sr=sr, target_sr=target_sr)
            spec = np.abs(librosa.stft(wave, n_fft=1024, hop_length=hop))
            power = np.square(spec)
            onset = librosa.onset.onset_strength(
                S=librosa.power_to_db(power + 1e-7), sr=target_sr, hop_length=hop)
            _tempo, bframes = librosa.beat.beat_track(
                onset_envelope=onset, sr=target_sr, hop_length=hop, trim=False)
            bframes = np.asarray(bframes, dtype=int).reshape(-1)
            if bframes.size < 26:
                continue
            mel = librosa.feature.melspectrogram(
                S=power, sr=target_sr, n_mels=int(cfg["n_mels"]), fmax=5000)
            mel = np.log1p(mel)
            chroma = librosa.feature.chroma_stft(S=power, sr=target_sr, n_chroma=12)
            rms = librosa.feature.rms(S=spec, frame_length=1024)[0]
            centroid = librosa.feature.spectral_centroid(S=spec, sr=target_sr)[0]
            nframes = min(chroma.shape[1], mel.shape[1], len(onset), len(rms), len(centroid))
            feature = np.vstack((chroma[:, :nframes],
                mel[:, :nframes] / max(1e-5, float(np.max(mel))),
                onset[np.newaxis, :nframes],
                rms[np.newaxis, :nframes]*5.0,
                centroid[np.newaxis, :nframes]/3000.0))
            beat_features = []
            for l, r in zip(bframes[:-1], bframes[1:]):
                l = int(max(0,l)); r = int(min(nframes,r))
                if r > l:
                    beat_features.append(np.mean(feature[:, l:r], axis=1))
            if len(beat_features) < 25:
                continue
            B = np.asarray(beat_features, dtype=float)
            B = (B - np.median(B, axis=0)) / (np.std(B, axis=0) + 0.20)
            B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
            sim8 = float(np.mean(np.sum(B[8:]*B[:-8], axis=1)))
            sim16 = float(np.mean(np.sum(B[16:]*B[:-16], axis=1)))
            delta = float(sim16 - sim8)
            if not math.isfinite(delta):
                continue
            window_deltas.append(delta)
            details.append({"center_sec": round(start + win_sec/2.0, 2),
                            "beats": len(B), "sim8": round(sim8, 4),
                            "sim16": round(sim16, 4), "delta": round(delta, 4)})
        except (ValueError, FloatingPointError, RuntimeError, TypeError, IndexError):
            # Problema de feature não derruba o aplicativo: caso ambíguo é
            # resolvido pelo motor métrico original e pela abstention.
            continue

    evidence = _phrase_8v16_evidence(window_deltas, tempo_cv)
    evidence["window_deltas"] = [float(x) for x in window_deltas]
    evidence["windows"] = details
    return evidence


def _apply_dual_bar_specialist(scores, phrase_evidence, raw_grid_stats=None):
    """Corrige apenas a distribuição interna da família binária 2/4 e 4/4.

    O sinal acústico negativo persistente (16 versus 8 beats) é evidência
    extra de barra longa em um two-feel, NÃO uma certeza editorial universal.
    """
    out = {s: float(v) for s,v in scores.items()}
    info = {"applied": False, "support": 0.0, "reason": "sem evidência suficiente"}
    if not phrase_evidence.get("valid"):
        return out, info
    pair_mass = out.get("2/4", 0.0) + out.get("4/4", 0.0)
    if pair_mass < CONFIG_DUAL_BAR["binary_family_min_mass"]:
        info["reason"] = "família binária com pouca evidência"
        return out, info
    # Evita mexer em 3/4 e 6/8 claros, mesmo havendo uma célula binária forte.
    ranked = sorted(out.items(), key=lambda x: x[1], reverse=True)
    if ranked[0][0] not in ("2/4", "4/4"):
        info["reason"] = "líder não pertence à família binária"
        return out, info
    supp = float(phrase_evidence.get("support_4", 0.0))
    mix = float(CONFIG_DUAL_BAR["max_blend"] * supp)
    if mix <= 1e-5:
        return out, info
    current_four = out["4/4"] / max(pair_mass, 1e-8)
    target_four = float(CONFIG_DUAL_BAR["target_four_share"])
    new_four = (1.0 - mix)*current_four + mix*target_four
    out["4/4"] = pair_mass * new_four
    out["2/4"] = pair_mass * (1.0-new_four)
    info.update({"applied": True, "support": supp, "mix": mix,
                 "old_four_share": current_four, "new_four_share": new_four,
                 "reason": "evidência contínua de frase longa"})
    return out, info


def run_hierarchical_meter_inference(tempo_data, dedup_notes, sr, log: TechnicalLogger):
    """
    A4.3: inferência hierárquica TATUM -> PULSE -> BAR + especialista de frase longa.
    A confiança do compasso é marginalizada sobre tactus plausíveis; a confiança do tactus é separada.
    """
    segments = tempo_data["segments"]
    top_tactus = _select_joint_tactus_candidates(tempo_data)
    y_harm = tempo_data["y_harm"]
    y_perc = tempo_data["y_perc"]
    onset_peaks = tempo_data["onset_peaks_times"]
    hop = CONFIG_SIGNAL["hop_length"]
    states = ["2/4", "3/4", "4/4", "6/8", "3/2"]

    fallback = {
        "chosen_bpm": tempo_data["global_raw_bpm"],
        "time_sig": "4/4",
        "status": "ambiguous",
        "confidence_label": "inconclusiva",
        "pulse_ql": 1.0,
        "meter_sections": [{"start_sec": 0.0, "time_sig": "4/4"}],
        "qualities": {
            "signal_quality": round(float(tempo_data.get("signal_quality", 0.5)), 3),
            "beat_quality": 0.5,
            "tactus_quality": 0.0,
            "pulse_quality": 0.0,
            "meter_confidence": 0.0,
            "grouping_quality": 0.0,
            "subdivision_quality": 0.0,
            "feature_agreement": 0.0,
            "decision_specific_quality": 0.0,
            "sequence_stability": 0.0,
            "bootstrap_stability": 0.0,
            "joint_hypothesis_stability": 0.0,
            "decision_quality": 0.0,
        },
    }
    if not segments or not top_tactus:
        return fallback

    tactus_evaluations = {}
    tactus_meta = {round(float(t["bpm"]), 2): t for t in top_tactus}
    joint_hypotheses = []

    # ---------- calcula evidências locais UMA vez por tactus/segmento ----------
    for t_cand in top_tactus:
        bpm = float(t_cand["bpm"])
        evals = []
        for seg in segments:
            s0 = seg["frame_start"] * hop
            s1 = seg["frame_end"] * hop
            notes = [n for n in dedup_notes if seg["time_start"] <= n.start_sec <= seg["time_end"]]
            ev = _evaluate_segment_for_tactus(seg, y_harm[s0:s1], y_perc[s0:s1], sr, bpm, onset_peaks, notes)
            if ev is not None:
                ev["pos_label"] = seg["pos_label"]
                ev["time_start"] = seg["time_start"]
                evals.append(ev)
        if not evals:
            continue
        tactus_evaluations[bpm] = evals
        prior = float(t_cand["score"])
        for ts in states:
            p = float(np.mean([e["ts_probs"][ts] for e in evals]))
            gq = float(np.mean([e["grouping_quality"] for e in evals]))
            fa = float(np.mean([e["feature_agreement"] for e in evals]))
            sq = float(np.mean([e["sub_rel"] for e in evals]))
            hq = float(np.mean([e["hierarchy_quality"] for e in evals]))
            rel = float(np.clip(0.32*gq + 0.24*fa + 0.20*sq + 0.24*hq, 0.0, 1.0))
            score = p * (0.82 + 0.18*prior) * (0.78 + 0.22*rel)
            phase = int(np.median([e["phases"][ts] for e in evals]))
            joint_hypotheses.append({
                "tactus_bpm": bpm,
                "time_sig": ts,
                "phase": phase,
                "score": float(score),
                "ts_prob": p,
                "reliability": rel,
                "compound_support": float(np.mean([e["compound_support"] for e in evals])),
                "four_hierarchy": float(np.mean([e["four_hierarchy"] for e in evals])),
            })

    if not tactus_evaluations or not joint_hypotheses:
        return fallback

    # ---------- especialista secundário: fingerprint da grade bruta ----------
    # Usa apenas medições que o próprio programa já extraiu. Ele só desempata
    # decisões ambíguas; decisões estruturais fortes permanecem dominantes.
    raw_grid_evals = []
    for seg in segments:
        bpm_raw_seg = float(seg.get("bpm", tempo_data.get("global_raw_bpm", 120.0)))
        if not math.isfinite(bpm_raw_seg) or bpm_raw_seg <= 0:
            continue
        s0 = seg["frame_start"] * hop
        s1 = seg["frame_end"] * hop
        notes = [n for n in dedup_notes if seg["time_start"] <= n.start_sec <= seg["time_end"]]
        ev_raw = _evaluate_segment_for_tactus(seg, y_harm[s0:s1], y_perc[s0:s1], sr, bpm_raw_seg, onset_peaks, notes)
        if ev_raw is not None:
            ev_raw["pos_label"] = seg["pos_label"]
            raw_grid_evals.append(ev_raw)

    def _raw_grid_prior(evals):
        if not evals:
            return {"2/4":0.25,"3/4":0.25,"4/4":0.20,"6/8":0.25,"3/2":0.05}, {}
        acc3 = float(np.mean([e.get("acc_abs",{}).get(3,0.5) for e in evals]))
        comp = float(np.mean([e["compound_support"] for e in evals]))
        g2 = float(np.mean([e["grp_scores"][2] for e in evals]))
        g3 = float(np.mean([e["grp_scores"][3] for e in evals]))
        g4 = float(np.mean([e["grp_scores"][4] for e in evals]))
        rec2 = float(np.mean([e["rec_abs"][2] for e in evals]))
        rec4 = float(np.mean([e["rec_abs"][4] for e in evals]))

        # Tudo contínuo: nenhuma regra por música ou degrau abrupto.
        compound_gate = _sigmoid((acc3 - 0.625) / 0.025) * _sigmoid((comp - 0.45) / 0.10)
        duple_gate = _sigmoid(((g2 - g3) - 0.002) / 0.018)
        four_gate = _sigmoid((0.080 - comp) / 0.030) * _sigmoid((0.720 - rec2) / 0.045)

        rem = max(0.0, 1.0 - 0.92 * compound_gate)
        p6 = 0.92 * compound_gate
        p4 = rem * four_gate * (0.45 + 0.55 * duple_gate)
        left = max(0.0, rem - p4)
        p2 = left * duple_gate
        p3 = left * (1.0 - duple_gate)
        p32 = 0.012
        pp = {"2/4":p2,"3/4":p3,"4/4":p4,"6/8":p6,"3/2":p32}
        z = sum(pp.values()) + 1e-8
        pp = {k:float(v/z) for k,v in pp.items()}
        stats={"acc3":acc3,"compound":comp,"grp2":g2,"grp3":g3,"grp4":g4,"rec2":rec2,"rec4":rec4,
               "duple_gate":duple_gate,"compound_gate":compound_gate,"four_gate":four_gate}
        return pp, stats

    raw_calibration_prior, raw_calibration_stats = _raw_grid_prior(raw_grid_evals)

    joint_hypotheses.sort(key=lambda h: h["score"], reverse=True)

    # ---------- marginalização sobre tactus: P(meter) = Σ P(meter|tactus) P(tactus) ----------
    candidate_bpms = sorted(tactus_evaluations.keys())
    raw_weights = []
    temp = float(CONFIG_METER["tactus_softmax_temperature"])
    best_prior = max(float(tactus_meta.get(round(b,2), {}).get("score", 0.0)) for b in candidate_bpms)
    for b in candidate_bpms:
        pr = float(tactus_meta.get(round(b,2), {}).get("score", 0.0))
        raw_weights.append(math.exp((pr - best_prior) / max(temp, 1e-4)))
    tactus_weights = np.asarray(raw_weights, dtype=float)
    tactus_weights /= (np.sum(tactus_weights) + 1e-8)

    meter_scores = {s: 0.0 for s in states}
    meter_best = {s: 0.0 for s in states}
    meter_den = {s: 0.0 for s in states}
    # Cada fórmula de compasso interpreta o papel do tactus de forma diferente.
    # Ex.: em 6/8 um tactus de ~57 BPM com forte subdivisão ternária deve pesar mais
    # que 86/172 BPM, mesmo que o prior perceptual desses últimos seja ligeiramente maior.
    for w, b in zip(tactus_weights, candidate_bpms):
        evals = tactus_evaluations[b]
        comp = float(np.mean([e["compound_support"] for e in evals]))
        four = float(np.mean([e["four_hierarchy"] for e in evals]))
        rec2 = float(np.mean([e["rec_abs"][2] for e in evals]))
        rec2_simple = _sigmoid((rec2 - CONFIG_METER["recurrence_2bar_center"]) / CONFIG_METER["recurrence_2bar_scale"])
        for st in states:
            local = float(np.mean([e["ts_probs"][st] for e in evals]))
            if st == "6/8":
                role = 0.12 + 0.88 * comp
            elif st == "4/4":
                role = (0.38 + 0.62 * (1.0-comp)) * (0.42 + 0.58*four) * (0.58 + 0.42*(1.0-rec2_simple))
            elif st == "2/4":
                role = (0.42 + 0.58 * (1.0-comp)) * (0.50 + 0.50*(1.0-four)) * (0.58 + 0.42*rec2_simple)
            else:
                role = 0.34 + 0.66 * (1.0-comp)
            ww = float(w) * max(0.05, role)
            meter_scores[st] += ww * local
            meter_den[st] += ww
            meter_best[st] = max(meter_best[st], local)
    meter_scores = {s: meter_scores[s]/max(meter_den[s],1e-8) for s in states}
    # Mistura marginalização condicionada + melhor hipótese para preservar estruturas fortes.
    meter_scores = {s: 0.74*meter_scores[s] + 0.26*meter_best[s] for s in states}

    # Relação métrica 3:1: quando o beat tracker rápido está em nível de subdivisão
    # e existe forte evidência composta em T/3, reforça 6/8 de forma contínua.
    raw_bpm = float(tempo_data.get("global_raw_bpm", 120.0))
    best_compound_relation = 0.0
    for b in candidate_bpms:
        evs = tactus_evaluations[b]
        comp = float(np.mean([e["compound_support"] for e in evs]))
        ratio = raw_bpm / max(float(b), 1e-8)
        triple_rel = math.exp(-0.5 * ((ratio - 3.0) / 0.48) ** 2)
        x = float(np.clip((comp - 0.62) / 0.25, 0.0, 1.0))
        strong_comp = x*x*(3.0-2.0*x)
        best_compound_relation = max(best_compound_relation, strong_comp * triple_rel)
    meter_scores["6/8"] *= (1.0 + 0.45 * best_compound_relation)

    total_m = sum(meter_scores.values()) + 1e-8
    meter_scores = {s: meter_scores[s]/total_m for s in states}

    # Especialista raw-grid: só recebe peso quando a margem estrutural é pequena.
    pre_ranked = sorted(meter_scores.items(), key=lambda kv: kv[1], reverse=True)
    pre_margin = float(pre_ranked[0][1] - pre_ranked[1][1])
    ambiguity_gate = float(np.clip((0.060 - pre_margin) / 0.060, 0.0, 1.0))
    raw_calibration_weight = 0.34 * ambiguity_gate
    if raw_calibration_weight > 1e-6 and pre_ranked[0][0] != "3/2":
        meter_scores = {
            st: (1.0 - raw_calibration_weight) * meter_scores[st] + raw_calibration_weight * raw_calibration_prior[st]
            for st in states
        }
        zc = sum(meter_scores.values()) + 1e-8
        meter_scores = {st:meter_scores[st]/zc for st in states}

    # A4.3: análise leve da recorrência de frase longa. Ela também é usada quando
    # o motor A4.2 estaria excessivamente confiante em 2/4.
    phrase_profile = {"valid": False, "support_4": 0.0, "window_deltas": [], "windows": []}
    if (CONFIG_DUAL_BAR["enabled"] and
            meter_scores["2/4"] + meter_scores["4/4"] >= CONFIG_DUAL_BAR["binary_family_min_mass"] and
            max(meter_scores, key=meter_scores.get) in ("2/4", "4/4")):
        phrase_profile = _dual_bar_phrase_profile(tempo_data, sr)
    meter_scores, dual_bar_info = _apply_dual_bar_specialist(
        meter_scores, phrase_profile, raw_calibration_stats)

    ranked = sorted(meter_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_ts, best_sc = ranked[0]
    sec_ts, sec_sc = ranked[1]
    meter_margin = float(best_sc - sec_sc)

    # Tactus escolhido CONDICIONADO ao compasso vencedor.
    conditioned = []
    for w, b in zip(tactus_weights, candidate_bpms):
        local = float(np.mean([e["ts_probs"][best_ts] for e in tactus_evaluations[b]]))
        conditioned.append((float(w)*local, b, local))
    conditioned.sort(reverse=True)
    _, chosen_bpm, chosen_meter_local = conditioned[0]
    second_cond = conditioned[1][0] if len(conditioned) > 1 else 0.0
    tactus_margin = max(0.0, conditioned[0][0] - second_cond)
    prior_chosen = float(tactus_meta.get(round(chosen_bpm,2), {}).get("score", 0.0))
    tactus_quality = float(np.clip(0.62*prior_chosen + 0.38*min(1.0, tactus_margin/0.08), 0.0, 1.0))
    seg_evals_chosen = tactus_evaluations[chosen_bpm]

    # ---------- Viterbi do compasso no tactus condicionado ----------
    n_st, n_win = len(states), len(seg_evals_chosen)
    self_p = float(CONFIG_METER["viterbi_self_transition"])
    switch_p = (1.0-self_p)/float(max(1,n_st-1))
    dp = np.zeros((n_win,n_st),dtype=float)
    back = np.zeros((n_win,n_st),dtype=int)
    for j,st in enumerate(states):
        dp[0,j] = math.log(seg_evals_chosen[0]["ts_probs"][st]+1e-9)
    for t in range(1,n_win):
        for j,st in enumerate(states):
            em = math.log(seg_evals_chosen[t]["ts_probs"][st]+1e-9)
            arr = np.array([dp[t-1,k]+math.log(self_p if k==j else switch_p) for k in range(n_st)])
            k=int(np.argmax(arr)); back[t,j]=k; dp[t,j]=arr[k]+em
    idx=[int(np.argmax(dp[-1]))]
    for t in range(n_win-1,0,-1): idx.append(int(back[t,idx[-1]]))
    idx.reverse()
    raw_path=[states[i] for i in idx]
    path=_stabilize_meter_path(raw_path,int(CONFIG_METER["viterbi_section_min_windows"]))
    meter_sections=[]
    for t,st in enumerate(path):
        ts0=0.0 if t==0 else round(float(seg_evals_chosen[t]["time_start"]),2)
        if not meter_sections or meter_sections[-1]["time_sig"]!=st:
            meter_sections.append({"start_sec":ts0,"time_sig":st})
    transitions=sum(1 for i in range(1,len(path)) if path[i]!=path[i-1])
    dominant=max(path.count(s) for s in set(path))/float(len(path))
    sequence_stability=float(np.clip(0.58*dominant+0.42*(1.0-transitions/max(1,len(path)-1)),0,1))

    # ---------- bootstrap marginalizado (tactus + meter) ----------
    # Usa a MESMA interpretação hierárquica da decisão principal; confiança não pode medir outro algoritmo.
    rng=np.random.default_rng(42)
    n_boot=int(CONFIG_CONFIDENCE["bootstrap_iterations"])
    meter_boot={s:0 for s in states}; joint_boot=defaultdict(int)
    raw_bpm=float(tempo_data.get("global_raw_bpm",120.0))
    for _ in range(n_boot):
        numer={s:0.0 for s in states}; den={s:0.0 for s in states}; best_local={s:0.0 for s in states}
        conditional_pairs={st:{} for st in states}
        best_comp_rel_boot=0.0
        for w,b in zip(tactus_weights,candidate_bpms):
            evals=tactus_evaluations[b]
            ids=rng.integers(0,len(evals),size=len(evals))
            sample=[evals[int(i)] for i in ids]
            loc={st:float(np.mean([e["ts_probs"][st] for e in sample])) for st in states}
            comp=float(np.mean([e["compound_support"] for e in sample]))
            four=float(np.mean([e["four_hierarchy"] for e in sample]))
            rec2=float(np.mean([e["rec_abs"][2] for e in sample]))
            rec2_simple=_sigmoid((rec2-CONFIG_METER["recurrence_2bar_center"])/CONFIG_METER["recurrence_2bar_scale"])
            ratio=raw_bpm/max(float(b),1e-8)
            triple_rel=math.exp(-0.5*((ratio-3.0)/0.48)**2)
            xx=float(np.clip((comp-0.62)/0.25,0.0,1.0)); strong_comp=xx*xx*(3.0-2.0*xx)
            best_comp_rel_boot=max(best_comp_rel_boot,strong_comp*triple_rel)
            for st in states:
                if st=="6/8":
                    role=0.12+0.88*comp
                elif st=="4/4":
                    role=(0.38+0.62*(1.0-comp))*(0.42+0.58*four)*(0.58+0.42*(1.0-rec2_simple))
                elif st=="2/4":
                    role=(0.42+0.58*(1.0-comp))*(0.50+0.50*(1.0-four))*(0.58+0.42*rec2_simple)
                else:
                    role=0.34+0.66*(1.0-comp)
                ww=float(w)*max(0.05,role)
                numer[st]+=ww*loc[st]; den[st]+=ww; best_local[st]=max(best_local[st],loc[st])
                conditional_pairs[st][float(b)] = ww * loc[st]
        bscores={st:0.74*(numer[st]/max(den[st],1e-8))+0.26*best_local[st] for st in states}
        bscores["6/8"]*=1.0+0.45*best_comp_rel_boot
        bz=sum(bscores.values())+1e-8
        bscores={st:bscores[st]/bz for st in states}

        # Bootstrap do mesmo algoritmo final, inclusive o calibrador raw-grid.
        if raw_grid_evals:
            ridx=rng.integers(0,len(raw_grid_evals),size=len(raw_grid_evals))
            rsample=[raw_grid_evals[int(i)] for i in ridx]
            rprior,_=_raw_grid_prior(rsample)
            br=sorted(bscores.items(),key=lambda kv:kv[1],reverse=True)
            bm=float(br[0][1]-br[1][1])
            bg=float(np.clip((0.060-bm)/0.060,0.0,1.0))
            bw=0.34*bg
            if bw>1e-6 and br[0][0] != "3/2":
                bscores={st:(1.0-bw)*bscores[st]+bw*rprior[st] for st in states}
                bz=sum(bscores.values())+1e-8
                bscores={st:bscores[st]/bz for st in states}

        # Reamostra somente as quatro pequenas medições já calculadas, sem
        # reprocessar áudio e sem custo extra de inferência neural.
        if phrase_profile.get("valid") and phrase_profile.get("window_deltas"):
            dv = np.asarray(phrase_profile["window_deltas"], dtype=float)
            rids = rng.integers(0, len(dv), size=len(dv))
            b_phrase = _phrase_8v16_evidence(dv[rids], phrase_profile["tempo_cv"])
            bscores, _ = _apply_dual_bar_specialist(bscores, b_phrase, raw_calibration_stats)

        st_win=max(bscores,key=bscores.get)
        meter_boot[st_win]+=1
        # A hipótese conjunta DEVE usar o mesmo compasso final que recebeu voto,
        # não um vencedor de um estágio intermediário anterior à calibração.
        conditioned_votes = conditional_pairs[st_win]
        bwin = max(conditioned_votes, key=conditioned_votes.get) if conditioned_votes else chosen_bpm
        joint_boot[(round(float(bwin),2), st_win)]+=1
    bootstrap_stability=meter_boot[best_ts]/float(max(1,n_boot))
    joint_stability=joint_boot.get((round(float(chosen_bpm),2),best_ts),0)/float(max(1,n_boot))

    # ---------- qualidades separadas ----------
    grouping_quality=float(np.mean([e["grouping_quality"] for e in seg_evals_chosen]))
    subdivision_quality=float(np.mean([e["sub_rel"] for e in seg_evals_chosen]))
    feature_agreement=float(np.mean([e["feature_agreement"] for e in seg_evals_chosen]))
    hierarchy_quality=float(np.mean([e["hierarchy_quality"] for e in seg_evals_chosen]))
    compound_support=float(np.mean([e["compound_support"] for e in seg_evals_chosen]))

    pair={best_ts,sec_ts}
    if pair=={"2/4","6/8"}:
        decision_specific_quality=max(subdivision_quality,abs(compound_support-0.5)*2.0)
    elif pair=={"2/4","4/4"}:
        decision_specific_quality=hierarchy_quality
    elif pair=={"3/4","3/2"}:
        decision_specific_quality=float(np.clip(0.55*grouping_quality+0.45*tactus_quality,0,1))
    else:
        decision_specific_quality=float(np.clip(0.55*grouping_quality+0.45*feature_agreement,0,1))

    meter_confidence=float(np.clip(
        0.34*min(1.0,meter_margin/max(CONFIG_CONFIDENCE["base_margin_threshold"],1e-8))+
        0.24*bootstrap_stability+
        0.16*sequence_stability+
        0.14*decision_specific_quality+
        0.12*feature_agreement,
        0.0,1.0))
    pulse_quality=float(np.clip(0.55*grouping_quality+0.45*decision_specific_quality,0,1))
    decision_quality=float(np.clip(0.62*meter_confidence+0.16*pulse_quality+0.12*tempo_data["signal_quality"]+0.10*tactus_quality,0,1))

    # Threshold só aumenta com evidência fraca.
    conf=CONFIG_CONFIDENCE
    penalty=0.0
    if decision_specific_quality<conf["min_decision_specific_quality"]:
        penalty += 0.028*(1.0-decision_specific_quality/max(conf["min_decision_specific_quality"],1e-8))
    if feature_agreement<conf["min_feature_agreement"]:
        penalty += 0.020*(1.0-feature_agreement/max(conf["min_feature_agreement"],1e-8))
    eff_thr=float(np.clip(conf["base_margin_threshold"]+penalty,conf["base_margin_threshold"],conf["max_margin_abs"]))

    # 6/8 exige suporte composto explícito; 4/4/2/4 exigem hierarquia minimamente informativa.
    structural_ok=True
    reasons=[]
    if best_ts=="6/8" and compound_support<0.52:
        structural_ok=False; reasons.append(f"compound_support insuficiente ({compound_support:.3f})")
    if pair=={"2/4","4/4"} and hierarchy_quality<0.035:
        structural_ok=False; reasons.append(f"hierarquia 2x4 pouco discriminante ({hierarchy_quality:.3f})")

    # Contradição de evidências: a classificação binária não deve ficar
    # confiante quando as frases se opõem ao motor local.
    if best_ts == "2/4" and phrase_profile.get("valid") and \
            phrase_profile.get("support_4", 0.0) >= CONFIG_DUAL_BAR["conflict_min_support"]:
        penalty_dual = CONFIG_DUAL_BAR["conflict_margin_penalty"]
        eff_thr = min(conf["max_margin_abs"], eff_thr + penalty_dual)
        reasons.append("dupla interpretação 2/4 x 4/4 nas frases longas")

    accept=(
        meter_margin>=eff_thr and
        meter_confidence>=conf["min_meter_confidence"] and
        decision_quality>=conf["min_decision_quality"] and
        bootstrap_stability>=conf["min_bootstrap_stability"] and
        structural_ok
    )
    if not accept:
        if meter_margin<eff_thr: reasons.append(f"margem insuficiente ({meter_margin:.4f} < {eff_thr:.4f})")
        if meter_confidence<conf["min_meter_confidence"]: reasons.append(f"meter_confidence baixa ({meter_confidence:.3f})")
        if bootstrap_stability<conf["min_bootstrap_stability"]: reasons.append(f"bootstrap baixo ({bootstrap_stability:.2f})")

    if accept:
        final_ts=best_ts; status="detected"; conf_label="alta" if meter_confidence>=0.72 else "média"
        refusal_reason="Nenhuma (estrutura pulse/bar consistente)"
    else:
        final_ts="4/4"; status="ambiguous"; conf_label="inconclusiva"
        refusal_reason="; ".join(dict.fromkeys(reasons)) if reasons else "Incerteza estrutural"

    pulse_ql_map={"2/4":1.0,"3/4":1.0,"4/4":1.0,"6/8":1.5,"3/2":2.0}
    pulse_ql=pulse_ql_map.get(final_ts,1.0)

    # Se só existe uma seção local, sincroniza a timeline com o consenso global final.
    # Mudanças reais de múltiplas seções continuam preservadas.
    if status == "detected" and len(meter_sections) == 1 and meter_sections[0]["time_sig"] != best_ts:
        meter_sections = [{"start_sec":0.0,"time_sig":best_ts}]

    # ---------- log auditável ----------
    log.add("\n==============================\nA4.3 TATUM -> PULSE -> BAR + DUAL-LEVEL BAR ENGINE\n==============================")
    log.add(f"Tactus avaliados: {[round(float(b),2) for b in candidate_bpms]}")
    log.add(f"Meter marginalizado/calibrado: { {k:round(v,4) for k,v in meter_scores.items()} }")
    log.add(f"RAW-GRID CALIBRATION: peso={raw_calibration_weight:.3f} | stats={ {k:round(v,3) for k,v in raw_calibration_stats.items()} } | prior={ {k:round(v,3) for k,v in raw_calibration_prior.items()} }")
    log.add(f"DUAL-LEVEL BAR: valid={phrase_profile.get('valid')} | "
            f"phrase_delta_median={phrase_profile.get('median')} | "
            f"negative_fraction={phrase_profile.get('negative_fraction')} | "
            f"tempo_cv={phrase_profile.get('tempo_cv')} | "
            f"support_4={phrase_profile.get('support_4',0.0):.3f} | "
            f"adjustment={dual_bar_info}")
    for item in phrase_profile.get("windows", []):
        log.add(f"  PHRASE WINDOW: {item}")
    log.add(f"Melhor compasso={best_ts} ({best_sc:.4f}) | 2º={sec_ts} ({sec_sc:.4f}) | margem={meter_margin:.4f}")
    log.add(f"Tactus condicionado ao compasso: {chosen_bpm:.2f} BPM | tactus_quality={tactus_quality:.3f}")
    log.add(f"VITERBI RAW={raw_path} | ESTABILIZADO={path} | Sections={meter_sections}")
    log.add(f"BOOTSTRAP METER ({n_boot})={meter_boot}")
    log.add(f"BOOTSTRAP JOINT top={dict(sorted(joint_boot.items(),key=lambda kv:kv[1],reverse=True)[:8])}")
    for e in seg_evals_chosen:
        log.add(
            f"  Seg {e['pos_label']}: grp={ {k:round(v,3) for k,v in e['grp_scores'].items()} } | "
            f"compound={e['compound_support']:.3f} (tern={e['sub_tern']:.3f}, rep={e['tern_rep']:.3f}, "
            f"half={e['half_ratio']:.3f}, thirds={e['third_ratio']:.3f}) | "
            f"4bar={e['four_hierarchy']:.3f} | rec2={e['rec_abs'][2]:.3f} | "
            f"TS={ {k:round(v,3) for k,v in e['ts_probs'].items()} }"
        )
    log.add(
        f"QUALITY: signal={tempo_data['signal_quality']:.3f} | tactus={tactus_quality:.3f} | pulse={pulse_quality:.3f} | "
        f"grouping={grouping_quality:.3f} | subdivision={subdivision_quality:.3f} | hierarchy={hierarchy_quality:.3f} | "
        f"agreement={feature_agreement:.3f} | meter_confidence={meter_confidence:.3f} | decision={decision_quality:.3f}"
    )
    log.add(f"Status={status} | Confiança={conf_label} | Threshold={eff_thr:.4f} | Motivo={refusal_reason}")

    qualities={
        "signal_quality":round(float(tempo_data["signal_quality"]),3),
        "tactus_quality":round(tactus_quality,3),
        "pulse_quality":round(pulse_quality,3),
        "meter_confidence":round(meter_confidence,3),
        "grouping_quality":round(grouping_quality,3),
        "subdivision_quality":round(subdivision_quality,3),
        "feature_agreement":round(feature_agreement,3),
        "decision_specific_quality":round(decision_specific_quality,3),
        "sequence_stability":round(sequence_stability,3),
        "bootstrap_stability":round(bootstrap_stability,3),
        "joint_hypothesis_stability":round(joint_stability,3),
        "decision_quality":round(decision_quality,3),
    }
    return {
        "chosen_bpm":round(float(chosen_bpm),1),
        "time_sig":final_ts,
        "status":status,
        "best_cand":best_ts,
        "second_cand":sec_ts,
        "margin":round(meter_margin,4),
        "confidence_label":conf_label,
        "pulse_ql":pulse_ql,
        "meter_sections":meter_sections if status=="detected" else [{"start_sec":0.0,"time_sig":"4/4"}],
        "qualities":qualities,
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

    # Respeita NO_MELODY: se nenhuma trajetória confiável foi encontrada, não inventa notas.

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
    dedup_notes, y_harm, sr, beat_times, pulse_ql, bar_length_ql, time_sig, detected_key, log: TechnicalLogger
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

    log.add(f"VELOCITY HARMONIA -> Base={v_h_base:.1f} | Medido(P28)={v_h_adapt:.1f} | Efetivo={v_h_eff:.1f}")

    try:
        chroma_full = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop)
    except Exception:
        chroma_full = None

    mapped_notes = []
    for n in harm_notes:
        s_ql = seconds_to_musical_position(n.start_sec, beat_times, pulse_ql=pulse_ql)
        e_ql = seconds_to_musical_position(n.end_sec, beat_times, pulse_ql=pulse_ql)
        d_ql = max(0.25, e_ql - s_ql)
        w_val = d_ql * (n.velocity / 127.0)
        mapped_notes.append((s_ql, e_ql, d_ql, n.pitch, w_val, n.start_sec, n.end_sec))

    weights_all = np.array([x[4] for x in mapped_notes], dtype=float)
    w_med, w_p70 = float(np.median(weights_all)), float(np.percentile(weights_all, 70))
    peso_base = CONFIG_HARMONY["peso_base"]
    peso_adapt = 0.50 * w_med + 0.50 * w_p70
    peso_eff = float(max(CONFIG_HARMONY["peso_min_floor"], 0.45 * peso_base + 0.55 * peso_adapt))
    log.add(f"PESO ACORDE (QL)  -> Base={peso_base:.3f} | Medido(Med={w_med:.3f}, P70={w_p70:.3f}) | Adapt={peso_adapt:.3f} | Efetivo={peso_eff:.3f}")

    max_ql = max(x[1] for x in mapped_notes)
    total_bars = int(math.ceil(max_ql / bar_length_ql)) + 1

    # Regiões harmônicas sincronizadas aos pulsos reais do compasso.
    if time_sig == "2/4":
        region_offsets = [0.0, 1.0]
        region_lengths = [1.0, 1.0]
    elif time_sig == "3/4":
        region_offsets = [0.0, 1.0, 2.0]
        region_lengths = [1.0, 1.0, 1.0]
    elif time_sig == "4/4":
        region_offsets = [0.0, 2.0]
        region_lengths = [2.0, 2.0]
    elif time_sig == "6/8":
        region_offsets = [0.0, 1.5]
        region_lengths = [1.5, 1.5]
    elif time_sig == "3/2":
        region_offsets = [0.0, 2.0, 4.0]
        region_lengths = [2.0, 2.0, 2.0]
    else:
        region_offsets = [0.0]
        region_lengths = [bar_length_ql]

    raw_regions = []
    abstention_count = 0

    for b_idx in range(total_bars):
        bar_start_ql = b_idx * bar_length_ql
        for s_i, (r_off, r_len) in enumerate(zip(region_offsets, region_lengths)):
            r_start_ql = bar_start_ql + r_off
            r_end_ql = r_start_ql + r_len

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
                    "offset_in_bar_ql": r_off,
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
                    "offset_in_bar_ql": r_off,
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
                "offset_in_bar_ql": r_off,
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
        sym_root_pc = {r["mpb_sym"]: r["root_pc"] for r in raw_regions if r["mpb_sym"] is not None}
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
                elif reg["quality"] == "UNKNOWN" and sym_root_pc.get(sym) == reg["root_pc"]:
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
        prev_root_pc = None
        logged_h = 0
        for t, reg in enumerate(raw_regions):
            hmm_sym = valid_vocabulary[path_v[t]]
            if reg["mpb_sym"] is None:
                if prev_chosen and prev_root_pc == reg["root_pc"]:
                    chosen_mpb = prev_chosen
                    chosen_xml = sym_to_xml[chosen_mpb]
                    trans_reason = "Mantido acorde anterior (mesma raiz com qualidade local UNKNOWN)"
                else:
                    chosen_mpb = None
                    chosen_xml = None
                    trans_reason = f"ABSTENTION ({reg['reason']})"
            else:
                strong_local = (
                    reg["root_conf"] >= CONFIG_HARMONY["local_override_root_conf"] and
                    reg["qual_conf"] >= CONFIG_HARMONY["local_override_quality_conf"] and
                    reg["qual_margin"] >= CONFIG_HARMONY["local_override_margin"] and
                    reg["chord_conf"] >= CONFIG_HARMONY["local_override_chord_conf"]
                )
                if strong_local:
                    chosen_mpb = reg["mpb_sym"]
                    chosen_xml = reg["xml_sym"]
                    trans_reason = "Evidência local forte prevaleceu sobre o HMM"
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
                prev_root_pc = sym_root_pc.get(chosen_mpb, reg["root_pc"])
                chords_by_bar[reg["bar_idx"]].append({
                    "offset_in_bar_ql": reg["offset_in_bar_ql"],
                    "mpb_sym": chosen_mpb,
                    "xml_sym": chosen_xml,
                    "confidence": reg["chord_conf"],
                })

        for b_k in list(chords_by_bar.keys()):
            c_list = chords_by_bar[b_k]
            compact = []
            for item in c_list:
                if not compact or item["mpb_sym"] != compact[-1]["mpb_sym"]:
                    compact.append(item)
            chords_by_bar[b_k] = compact

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
        log.add(f"CONFIG_DUAL_BAR: {CONFIG_DUAL_BAR}")
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
                    # Mix mono preserva todo o conteúdo L/R para ritmo, harmonia e Basic Pitch.
                    # MID/SIDE é usado apenas para auditoria nesta versão, sem descartar conteúdo lateral.
                    y_full = np.mean(y_raw[:2], axis=0).astype(np.float32)
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

                            dur_ql = max(0.25, dur_ql)
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

                        harmony_key_context = detected_key if key_success else None
                        chords_by_bar, chord_abstentions = analyze_harmony_root_quality_hmm(
                            dedup_notes, tempo_data["y_harm"], sr, beat_times, pulse_ql, bar_length_ql, time_sig, harmony_key_context, log
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
                            f"Joint Stability: {meter_qualities.get('joint_hypothesis_stability', meter_qualities['bootstrap_stability']):.2f} | "
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
