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

APP_VERSION = "A3.1-Adaptive"

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
# 2. CALIBRAÇÃO MODULARIZADA EM 4 GRUPOS
# ---------------------------------------------------------
CONFIG_SIGNAL = {
    "sample_rate": 22050,
    "chunk_duration_sec": 60.0,
    "chunk_overlap_sec": 1.0,
    "min_chunk_sec": 1.5,
    "dedup_time_tol_sec": 0.045,
    "duration_min_abs_sec": 0.08,
    "min_ql": 0.25,
    "subdivision_ql": 0.25,
}

CONFIG_METER = {
    "w_accent": 0.28,
    "w_periodicity": 0.30,
    "w_segment_consistency": 0.16,
    "w_subdivision": 0.10,
    "w_harmonic_change": 0.08,
    "w_recurrence": 0.08,
    "base_margin_threshold": 0.065,
    "min_margin_abs": 0.030,
    "max_margin_abs": 0.130,
    "segment_positions": (0.20, 0.50, 0.80),
    "segment_window_sec": 18.0,
}

CONFIG_MELODY = {
    "velocity_base": 50.0,
    "velocity_min_floor": 34.0,
    "velocity_max_ceil": 66.0,
    "pitch_base_min": 52,
    "pitch_floor_abs": 45,
    "duration_base_sec": 0.10,
    "jump_penalty_semitone": 0.55,
    "octave_jump_penalty": 4.5,
    "continuity_bonus": 3.0,
    "simultaneous_tolerance_sec": 0.045,
}

CONFIG_HARMONY = {
    "velocity_base": 42.0,
    "velocity_min_floor": 28.0,
    "peso_base": 0.20,
    "peso_min_floor": 0.08,
    "beats_per_region": 2.0,
    "template_min_confidence": 0.42,
    "bass_root_bonus": 0.14,
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
# 4. ENARMONIA, TONALIDADE E FORMATAÇÃO PT-BR
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

def format_root_pt_br(pitch_name: str, use_flats: bool = False) -> str:
    if not pitch_name:
        return "C"
    clean = pitch_name.replace("-", "b")
    enarm_map = {"Fb": "E", "Cb": "B", "E#": "F", "B#": "C"}
    clean = enarm_map.get(clean, clean)
    try:
        p = music21.pitch.Pitch(note_name_to_music21(clean))
        return pc_to_note_name(p.pitchClass, use_flats=use_flats)
    except Exception:
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


def detectar_centro_tonal_global(part, y_mono=None, sr=22050, all_notes=None, log: TechnicalLogger = None):
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_profile = major_profile / np.linalg.norm(major_profile)
    minor_profile = minor_profile / np.linalg.norm(minor_profile)

    pc_Floor = np.zeros(12, dtype=float)

    if y_mono is not None and len(y_mono) > sr * 2:
        try:
            sample_lim = min(len(y_mono), int(60 * sr))
            chroma = librosa.feature.chroma_cqt(y=y_mono[:sample_lim], sr=sr)
            chroma_vec = np.mean(chroma, axis=1)
            if np.max(chroma_vec) > 0:
                pc_Floor += 0.45 * (chroma_vec / np.max(chroma_vec))
        except Exception:
            pass

    if all_notes:
        hist_notes = np.zeros(12, dtype=float)
        for n in all_notes:
            hist_notes[n.pitch % 12] += n.duration_sec * (n.velocity / 127.0)
        if np.max(hist_notes) > 0:
            pc_Floor += 0.35 * (hist_notes / np.max(hist_notes))

    mel_notes = list(part.recurse().notes) if part is not None else []
    if mel_notes:
        hist_mel = np.zeros(12, dtype=float)
        for el in mel_notes:
            for p in el.pitches:
                hist_mel[p.pitchClass] += float(el.quarterLength)
        if np.max(hist_mel) > 0:
            pc_Floor += 0.20 * (hist_mel / np.max(hist_mel))

    if np.sum(pc_Floor) <= 1e-6:
        if log:
            log.add_warning("Material insuficiente para análise tonal. Aplicando fallback C major.")
        return music21.key.Key("C"), False, 0.0

    pc_norm = pc_Floor / np.linalg.norm(pc_Floor)
    candidates = []
    for pc in range(12):
        maj_rot = np.roll(major_profile, pc)
        min_rot = np.roll(minor_profile, pc)
        corr_maj = float(np.dot(pc_norm, maj_rot))
        corr_min = float(np.dot(pc_norm, min_rot))
        candidates.append((corr_maj, pc, "major"))
        candidates.append((corr_min, pc, "minor"))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_corr, best_pc, best_mode = candidates[0]
    second_corr = candidates[1][0]
    margin = best_corr - second_corr

    if best_mode == "major":
        use_flats = best_pc in {1, 3, 5, 8, 10}
    else:
        use_flats = best_pc in {2, 3, 5, 7, 10}
    tonic_str = pc_to_note_name(best_pc, use_flats=use_flats)
    m21_tonic = note_name_to_music21(tonic_str)

    if log:
        log.add("\n==============================\nTONALITY & ENHARMONIC ANALYSIS\n==============================")
        for rank_i, (sc, pc_i, md_i) in enumerate(candidates[:5]):
            nm = pc_to_note_name(pc_i, use_flats=(pc_i in {1, 3, 5, 8, 10}))
            log.add(f"  Candidato {rank_i+1}: {nm} {md_i} = {sc:.4f}")
        log.add(f"Escolhido: {tonic_str} {best_mode} (corr={best_corr:.4f}, margem={margin:.4f})")

    try:
        key_obj = music21.key.Key(m21_tonic, best_mode)
        conf = min(1.0, max(0.0, (best_corr - 0.5) * 1.5 + margin * 2.0))
        if best_corr < 0.58:
            return music21.key.Key("C"), False, conf
        return key_obj, True, conf
    except Exception:
        return music21.key.Key("C"), False, 0.0


# ---------------------------------------------------------
# 5. RESOLUÇÃO DE TACTUS E BEAT GRID REAL
# ---------------------------------------------------------
def estimate_local_bpm_and_beats(y_seg, sr, hop_length=512):
    if len(y_seg) < sr * 2:
        return 120.0, np.array([]), np.array([])
    onset_env = librosa.onset.onset_strength(y=y_seg, sr=sr, hop_length=hop_length)
    if len(onset_env) == 0 or np.max(onset_env) <= 0:
        return 120.0, np.array([]), onset_env
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    t_arr = np.asarray(tempo).reshape(-1)
    bpm_val = float(t_arr[0]) if t_arr.size > 0 and math.isfinite(float(t_arr[0])) else 120.0
    if bpm_val <= 0:
        bpm_val = 120.0
    return bpm_val, beats, onset_env

def resolve_tactus_level(raw_bpm_45s, local_bpms, onset_env_45s, sr, log: TechnicalLogger):
    base_candidates = []
    for b in [raw_bpm_45s] + list(local_bpms):
        if b > 0 and math.isfinite(b):
            for mult in [0.5, 1.0, 2.0, 2.0/3.0, 1.5]:
                cand = round(b * mult, 2)
                if 48.0 <= cand <= 208.0:
                    base_candidates.append(cand)

    if not base_candidates:
        base_candidates = [120.0]

    base_candidates.sort()
    merged = []
    for c in base_candidates:
        if not merged or abs(c - merged[-1][-1]) / merged[-1][-1] > 0.03:
            merged.append([c])
        else:
            merged[-1].append(c)
    unique_cands = [round(float(np.mean(g)), 2) for g in merged]

    hop_length = 512
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env_45s, sr=sr, hop_length=hop_length)
    ac_mean = np.mean(tempogram, axis=1)
    ac_mean = ac_mean / (np.max(ac_mean) + 1e-8)

    scored_cands = []
    for cand in unique_cands:
        sec_per_beat = 60.0 / cand
        lag_frames = int(round((sec_per_beat * sr) / hop_length))
        periodicity_val = float(ac_mean[lag_frames]) if 0 < lag_frames < len(ac_mean) else 0.0

        perceptual_weight = math.exp(-0.5 * ((cand - 102.0) / 38.0) ** 2)

        agree_scores = []
        for lb in local_bpms:
            ratios = [ abs(cand - lb * m) / cand for m in [0.5, 1.0, 2.0] ]
            min_err = min(ratios)
            agree_scores.append(max(0.0, 1.0 - min_err * 5.0))
        seg_cons = float(np.mean(agree_scores)) if agree_scores else 0.5

        total_sc = 0.45 * periodicity_val + 0.35 * perceptual_weight + 0.20 * seg_cons
        scored_cands.append((total_sc, cand, periodicity_val, perceptual_weight, seg_cons))

    scored_cands.sort(key=lambda x: x[0], reverse=True)
    best_sc, chosen_bpm, p_val, perc_w, s_cons = scored_cands[0]

    log.add("\n==============================\nBPM & TACTUS RESOLUTION\n==============================")
    log.add(f"BPM bruto primeiros 45s: {raw_bpm_45s:.2f}")
    for i, lb in enumerate(local_bpms):
        diff_pct = ((lb - raw_bpm_45s) / (raw_bpm_45s + 1e-8)) * 100.0
        log.add(f"BPM local segmento {i+1}: {lb:.2f} (diferença p/ global: {diff_pct:+.1f}%)")
    log.add("Candidatos de Tactus avaliados:")
    for sc, cd, pv, pw, scs in scored_cands[:6]:
        log.add(f"  {cd:6.2f} BPM -> Score={sc:.3f} (periodicity={pv:.3f}, perceptual={pw:.3f}, consistency={scs:.3f})")
    log.add(f"Tactus selecionado: {chosen_bpm:.1f} BPM")

    return round(chosen_bpm, 1)


def build_real_beat_grid(y_full, sr, tactus_bpm, total_duration, log: TechnicalLogger):
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y_full, sr=sr, hop_length=hop_length)
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, bpm=tactus_bpm
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    sec_per_beat = 60.0 / tactus_bpm

    if len(beat_times) < 4:
        beat_times = np.arange(0.0, total_duration + sec_per_beat, sec_per_beat)
        stability = 0.5
        med_ibi = sec_per_beat
        std_ibi = 0.0
    else:
        ibis = np.diff(beat_times)
        med_ibi = float(np.median(ibis))
        std_ibi = float(np.std(ibis))
        cv = std_ibi / (med_ibi + 1e-8)
        stability = float(max(0.0, min(1.0, 1.0 - cv * 2.5)))

        first_beat = float(beat_times[0])
        pre_beats = []
        curr = first_beat - med_ibi
        while curr >= 0.0:
            pre_beats.append(curr)
            curr -= med_ibi
        pre_beats.reverse()

        last_beat = float(beat_times[-1])
        post_beats = []
        curr = last_beat + med_ibi
        while curr <= total_duration + med_ibi:
            post_beats.append(curr)
            curr += med_ibi

        beat_times = np.concatenate([np.array(pre_beats), beat_times, np.array(post_beats)])

    log.add("\n==============================\nREAL BEAT GRID AUDIT\n==============================")
    log.add(f"Quantidade total de beats na grade: {len(beat_times)}")
    log.add(f"Duração média/mediana do pulso: {med_ibi:.4f}s | Desvio padrão: {std_ibi:.4f}s")
    log.add(f"Estabilidade da grade (0-1): {stability:.3f}")
    first_20 = [round(float(x), 3) for x in beat_times[:20]]
    last_20 = [round(float(x), 3) for x in beat_times[-20:]]
    log.add(f"Primeiros 20 beats (s): {first_20}")
    log.add(f"Últimos 20 beats (s): {last_20}")

    return beat_times, stability, onset_env


def seconds_to_musical_position(time_sec: float, beat_times: np.ndarray, pulse_ql: float = 1.0) -> float:
    if beat_times is None or len(beat_times) < 2:
        return max(0.0, float(time_sec))
    idx_float = float(np.interp(time_sec, beat_times, np.arange(len(beat_times), dtype=float)))
    return max(0.0, idx_float * pulse_ql)

def musical_position_to_seconds(pos_ql: float, beat_times: np.ndarray, pulse_ql: float = 1.0) -> float:
    if beat_times is None or len(beat_times) < 2:
        return max(0.0, float(pos_ql))
    beat_idx = pos_ql / (pulse_ql + 1e-8)
    return float(np.interp(beat_idx, np.arange(len(beat_times), dtype=float), beat_times))


# ---------------------------------------------------------
# 6. DETECTOR MÉTRICO HÍBRIDO A3 (PULSOS + UNIDADE + ABSTENTION)
# ---------------------------------------------------------
def _analisar_segmento_a3(y_seg, sr, ref_bpm):
    hop_length = 512
    try:
        y_harm, y_perc = librosa.effects.hpss(y_seg)
    except Exception:
        y_harm, y_perc = y_seg, y_seg

    local_bpm, beats, onset_env = estimate_local_bpm_and_beats(y_perc, sr, hop_length=hop_length)
    eff_bpm = local_bpm if (0.75 * ref_bpm <= local_bpm <= 1.33 * ref_bpm) else ref_bpm

    if len(beats) < 8:
        _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length, bpm=eff_bpm)

    if len(beats) < 6 or len(onset_env) == 0:
        return None

    onset_norm = onset_env / (np.max(onset_env) + 1e-8)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_norm, sr=sr, hop_length=hop_length)
    ac_mean = np.mean(tempogram, axis=1)
    ac_mean = ac_mean / (np.max(ac_mean) + 1e-8)

    frames_per_beat = max(2, int(round((60.0 / eff_bpm * sr) / hop_length)))

    def _lag_energy(mult_factor):
        lag = int(round(frames_per_beat * mult_factor))
        if 1 <= lag < len(ac_mean):
            w_a = max(1, lag - 1)
            w_b = min(len(ac_mean), lag + 2)
            return float(np.max(ac_mean[w_a:w_b]))
        return 0.0

    per_2 = _lag_energy(2.0)
    per_3 = _lag_energy(3.0)
    per_4 = _lag_energy(4.0)

    sub_bin = _lag_energy(0.5)
    sub_tern = _lag_energy(1.0 / 3.0)
    if hasattr(librosa.feature, "tempogram_ratio"):
        try:
            tr = librosa.feature.tempogram_ratio(tg=tempogram, sr=sr, hop_length=hop_length, bpm=eff_bpm)
            tr_mean = np.mean(tr, axis=1)
            if len(tr_mean) >= 6:
                sub_bin = float(max(sub_bin, np.mean(tr_mean[1:3])))
                sub_tern = float(max(sub_tern, np.mean(tr_mean[3:6])))
        except Exception:
            pass

    sub_tot = sub_bin + sub_tern + 1e-8
    sub_bin_norm = sub_bin / sub_tot
    sub_tern_norm = sub_tern / sub_tot

    beat_accents = []
    for b in beats:
        bi = int(b)
        w0, w1 = max(0, bi - 2), min(len(onset_norm), bi + 3)
        beat_accents.append(float(np.max(onset_norm[w0:w1])) if w1 > w0 else 0.0)
    beat_accents = np.array(beat_accents)

    try:
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
        c_sync = librosa.util.sync(chroma, beats)
        c_norm = c_sync / (np.linalg.norm(c_sync, axis=0, keepdims=True) + 1e-8)
        h_diff = np.sqrt(np.sum(np.diff(c_norm, axis=1) ** 2, axis=0))
        h_diff = np.pad(h_diff, (1, 0), mode="edge")
        h_diff = h_diff / (np.max(h_diff) + 1e-8)
    except Exception:
        c_norm = np.zeros((12, len(beats)))
        h_diff = np.zeros(len(beats))

    templates_acc = {
        2: np.array([1.0, -0.85]),
        3: np.array([1.0, -0.50, -0.50]),
        4: np.array([1.0, -0.60, 0.35, -0.60]),
    }

    accent_scores = {}
    harm_scores = {}
    best_phases = {}

    b_z = (beat_accents - np.mean(beat_accents)) / (np.std(beat_accents) + 1e-8)
    for pulse_n, tpl in templates_acc.items():
        best_f = 0
        best_a_sc = -999.0
        best_h_sc = 0.0
        for f in range(pulse_n):
            idx_seq = np.arange(len(beats))
            roles = (idx_seq - f) % pulse_n
            weights = tpl[roles]
            a_sc = float(np.mean(b_z * weights))
            down_idx = idx_seq[roles == 0]
            h_sc = float(np.mean(h_diff[down_idx])) if len(down_idx) > 0 else 0.0
            comb = 0.72 * a_sc + 0.28 * h_sc
            if comb > best_a_sc:
                best_a_sc = comb
                best_f = f
                best_h_sc = h_sc
        accent_scores[pulse_n] = float(1.0 / (1.0 + math.exp(-2.2 * best_a_sc)))
        harm_scores[pulse_n] = float(best_h_sc)
        best_phases[pulse_n] = int(best_f)

    rec_scores = {2: 0.5, 3: 0.5, 4: 0.5}
    if c_norm.shape[1] >= 12:
        sim_m = np.dot(c_norm.T, c_norm)
        for lag_k in [2, 3, 4]:
            diags = [sim_m[i, i + lag_k] for i in range(sim_m.shape[0] - lag_k)]
            rec_scores[lag_k] = float(np.clip(np.mean(diags), 0.0, 1.0)) if diags else 0.5

    ev_quality = float(np.clip(
        0.4 * (len(beats) / 24.0) + 0.3 * np.std(beat_accents) * 3.0 + 0.3 * np.max([per_2, per_3, per_4]),
        0.1, 1.0
    ))

    return {
        "local_bpm": round(local_bpm, 2),
        "n_beats": len(beats),
        "per": {2: per_2, 3: per_3, 4: per_4},
        "acc": accent_scores,
        "harm": harm_scores,
        "rec": rec_scores,
        "phases": best_phases,
        "sub_bin": sub_bin_norm,
        "sub_tern": sub_tern_norm,
        "ev_quality": ev_quality,
    }


def detect_meter_a3(y_full, sr, tactus_bpm, beat_stability, log: TechnicalLogger):
    log.add("\n==============================\nMETER DETECTION (A3 ENGINE)\n==============================")
    total_len = len(y_full)
    win_samples = int(CONFIG_METER["segment_window_sec"] * sr)
    seg_results = []

    for idx, pos in enumerate(CONFIG_METER["segment_positions"]):
        c_idx = int(pos * total_len)
        s0 = max(0, c_idx - win_samples // 2)
        s1 = min(total_len, s0 + win_samples)
        res = _analisar_segmento_a3(y_full[s0:s1], sr, tactus_bpm)
        if res is not None:
            res["pos_label"] = f"{int(pos*100)}%"
            seg_results.append(res)
            log.add(
                f"\nSEGMENTO {res['pos_label']} (BPM local={res['local_bpm']}, beats={res['n_beats']}, ev_q={res['ev_quality']:.2f}):\n"
                f"  periodicidade -> 2: {res['per'][2]:.3f} | 3: {res['per'][3]:.3f} | 4: {res['per'][4]:.3f}\n"
                f"  accent score  -> 2: {res['acc'][2]:.3f} (f={res['phases'][2]}) | "
                f"3: {res['acc'][3]:.3f} (f={res['phases'][3]}) | 4: {res['acc'][4]:.3f} (f={res['phases'][4]})\n"
                f"  subdivisão    -> binária: {res['sub_bin']:.3f} | ternária: {res['sub_tern']:.3f}\n"
                f"  harm change   -> 2: {res['harm'][2]:.3f} | 3: {res['harm'][3]:.3f} | 4: {res['harm'][4]:.3f}\n"
                f"  recurrence    -> 2: {res['rec'][2]:.3f} | 3: {res['rec'][3]:.3f} | 4: {res['rec'][4]:.3f}"
            )

    if not seg_results:
        log.add_warning("Nenhum segmento produziu evidência métrica suficiente. Usando fallback 4/4.")
        return {
            "time_sig": "4/4",
            "status": "ambiguous",
            "confidence_label": "inconclusiva",
            "evidence_quality": 0.0,
            "margin": 0.0,
            "pulse_ql": 1.0,
            "meter_sections": [{"start_sec": 0.0, "time_sig": "4/4"}],
        }

    w = CONFIG_METER
    candidates = ["2/4", "3/4", "4/4", "6/8", "3/2"]
    seg_ts_scores = []
    seg_winners = []

    for r in seg_results:
        g2 = (
            w["w_periodicity"] * r["per"][2]
            + w["w_accent"] * r["acc"][2]
            + w["w_harmonic_change"] * r["harm"][2]
            + w["w_recurrence"] * r["rec"][2]
        )
        g3 = (
            w["w_periodicity"] * r["per"][3]
            + w["w_accent"] * r["acc"][3]
            + w["w_harmonic_change"] * r["harm"][3]
            + w["w_recurrence"] * r["rec"][3]
        )
        g4 = (
            w["w_periodicity"] * r["per"][4]
            + w["w_accent"] * r["acc"][4]
            + w["w_harmonic_change"] * r["harm"][4]
            + w["w_recurrence"] * r["rec"][4]
        )

        sc_2_4 = g2 + w["w_subdivision"] * r["sub_bin"]
        sc_6_8 = g2 * 0.85 + w["w_subdivision"] * (r["sub_tern"] * 1.35) + 0.10 * r["per"][3]
        sc_3_4 = g3 + w["w_subdivision"] * r["sub_bin"]
        sc_3_2 = g3 * 0.88 + w["w_subdivision"] * r["sub_bin"] * (0.9 if tactus_bpm < 95 else 0.6)
        sc_4_4 = g4 + w["w_subdivision"] * r["sub_bin"]

        raw_d = {"2/4": sc_2_4, "3/4": sc_3_4, "4/4": sc_4_4, "6/8": sc_6_8, "3/2": sc_3_2}
        tot_d = sum(raw_d.values()) + 1e-8
        norm_d = {k: v / tot_d for k, v in raw_d.items()}
        seg_ts_scores.append(norm_d)
        seg_winners.append(max(norm_d.items(), key=lambda x: x[1])[0])

    ev_weights = np.array([r["ev_quality"] for r in seg_results])
    ev_weights = ev_weights / (np.sum(ev_weights) + 1e-8)

    global_scores = {c: 0.0 for c in candidates}
    for c in candidates:
        global_scores[c] = float(np.sum([seg_ts_scores[i][c] * ev_weights[i] for i in range(len(seg_results))]))

    top_seg_winner = max(set(seg_winners), key=seg_winners.count)
    agreement_ratio = seg_winners.count(top_seg_winner) / float(len(seg_winners))

    global_scores[top_seg_winner] += w["w_segment_consistency"] * agreement_ratio
    g_sum = sum(global_scores.values()) + 1e-8
    global_scores = {k: v / g_sum for k, v in global_scores.items()}

    ranking = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)
    best_ts, best_sc = ranking[0]
    sec_ts, sec_sc = ranking[1]
    margin = best_sc - sec_sc

    mean_ev_q = float(np.mean([r["ev_quality"] for r in seg_results]))
    evidence_quality = float(np.clip(0.50 * mean_ev_q + 0.30 * beat_stability + 0.20 * agreement_ratio, 0.0, 1.0))

    base_thr = w["base_margin_threshold"]
    adj_thr = -0.030 * (evidence_quality - 0.60) - 0.015 * (agreement_ratio - 0.66)
    eff_thr = float(np.clip(base_thr + adj_thr, w["min_margin_abs"], w["max_margin_abs"]))

    if margin >= eff_thr and evidence_quality >= 0.45:
        status = "detected"
        chosen_ts = best_ts
        conf_label = "alta" if (margin >= eff_thr * 1.4 and agreement_ratio >= 0.66) else "média"
        refusal_reason = "Nenhuma (aceito automaticamente)"
    else:
        status = "ambiguous"
        chosen_ts = "4/4"
        conf_label = "inconclusiva"
        refusal_reason = f"Margem ({margin:.4f}) < Limiar Efetivo ({eff_thr:.4f}) ou Evidência Baixa ({evidence_quality:.2f})"

    pulse_ql_map = {"2/4": 1.0, "3/4": 1.0, "4/4": 1.0, "6/8": 1.5, "3/2": 2.0}
    pulse_ql = pulse_ql_map.get(chosen_ts, 1.0)

    log.add("\nGLOBAL METER SUMMARY:")
    log.add(f"  Time signature candidates: " + " | ".join([f"{k}={global_scores[k]:.4f}" for k in candidates]))
    log.add(f"  Melhor: {best_ts} ({best_sc:.4f}) | Segundo: {sec_ts} ({sec_sc:.4f}) | Margem: {margin:.4f}")
    log.add(f"  Evidence Quality: {evidence_quality:.3f} | Concordância entre segmentos: {agreement_ratio:.0%}")
    log.add(f"  Limiar Base: {base_thr:.4f} | Ajuste: {adj_thr:+.4f} | Limiar Efetivo: {eff_thr:.4f}")
    log.add(f"  Status: {status} | Confiança: {conf_label} | Motivo: {refusal_reason}")
    log.add(f"  Resultado Efetivo: {chosen_ts}")

    return {
        "time_sig": chosen_ts,
        "status": status,
        "best_cand": best_ts,
        "second_cand": sec_ts,
        "margin": round(margin, 4),
        "evidence_quality": round(evidence_quality, 3),
        "confidence_label": conf_label,
        "pulse_ql": pulse_ql,
        "meter_sections": [{"start_sec": 0.0, "time_sig": chosen_ts}],
    }


# ---------------------------------------------------------
# 7. INFERÊNCIA BASIC PITCH COM OVERLAP E DEDUPLICAÇÃO
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
    overlap_Merged = 0
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
                    overlap_Merged += 1
                    merged_flag = True
                    break
            if prev.end_sec < n.start_sec - 0.5:
                break
        if not merged_flag:
            dedup_notes.append(n)

    log.add("\n==============================\nBASIC PITCH OVERLAP & DEDUPLICATION\n==============================")
    log.add(f"Total de chunks processados: {len(notes_per_chunk)}")
    for c_i, st_t, en_t, cnt in notes_per_chunk:
        log.add(f"  Chunk {c_i} ({st_t}s - {en_t}s): {cnt} notas brutas")
    log.add(f"Notas brutas totais: {len(raw_notes)} | Duplicatas de borda fundidas: {overlap_Merged} | Notas válidas pós-dedup: {len(dedup_notes)}")

    return raw_notes, dedup_notes


# ---------------------------------------------------------
# 8. CALIBRAÇÃO ADAPTATIVA E MELODY TRACKING (VITERBI / DP)
# ---------------------------------------------------------
def track_melody_viterbi(dedup_notes, log: TechnicalLogger):
    log.add("\n==============================\nADAPTIVE MELODY CALIBRATION & TRACKING\n==============================")
    if not dedup_notes:
        return [], {}

    vels = np.array([n.velocity for n in dedup_notes], dtype=float)
    pitches = np.array([n.pitch for n in dedup_notes], dtype=float)
    durs = np.array([n.duration_sec for n in dedup_notes], dtype=float)

    v_mean, v_med = float(np.mean(vels)), float(np.median(vels))
    v_p25, v_p75 = float(np.percentile(vels, 25)), float(np.percentile(vels, 75))
    p_med, p_p20, p_p85 = float(np.median(pitches)), float(np.percentile(pitches, 20)), float(np.percentile(pitches, 85))
    d_med = float(np.median(durs))

    v_base = CONFIG_MELODY["velocity_base"]
    v_adapt = v_med - 0.15 * (v_p75 - v_p25)
    v_eff = float(np.clip(0.50 * v_base + 0.50 * v_adapt, CONFIG_MELODY["velocity_min_floor"], CONFIG_MELODY["velocity_max_ceil"]))

    p_base = CONFIG_MELODY["pitch_base_min"]
    p_adapt = max(CONFIG_MELODY["pitch_floor_abs"], int(round(p_p20 - 2)))
    p_eff = int(round(0.45 * p_base + 0.55 * p_adapt))

    d_base = CONFIG_MELODY["duration_base_sec"]
    d_adapt = max(CONFIG_SIGNAL["duration_min_abs_sec"], min(0.12, d_med * 0.45))
    d_eff = float(0.50 * d_base + 0.50 * d_adapt)

    log.add(f"Estatísticas da Música -> Velocity med={v_med:.1f} (P25={v_p25:.1f}, P75={v_p75:.1f}) | Pitch med={p_med:.1f} (P20={p_p20:.1f}, P85={p_p85:.1f}) | Dur med={d_med:.3f}s")
    log.add(f"VELOCITY MELODIA -> Base: {v_base:.1f} | Adaptativo: {v_adapt:.1f} | Efetivo: {v_eff:.1f}")
    log.add(f"CORTE PITCH      -> Base: {p_base} | Adaptativo: {p_adapt} | Efetivo: {p_eff}")
    log.add(f"DURAÇÃO MÍNIMA   -> Base: {d_base:.3f}s | Adaptativo: {d_adapt:.3f}s | Efetivo: {d_eff:.3f}s")

    candidates = [
        n for n in dedup_notes
        if n.velocity >= v_eff and n.pitch >= p_eff and n.duration_sec >= d_eff
    ]

    if not candidates:
        v_eff_relax = max(25.0, v_p25 * 0.85)
        p_eff_relax = CONFIG_MELODY["pitch_floor_abs"]
        log.add_warning(f"Filtro inicial zerou candidatas. Relaxando para vel>={v_eff_relax:.1f}, pitch>={p_eff_relax}")
        candidates = [n for n in dedup_notes if n.velocity >= v_eff_relax and n.pitch >= p_eff_relax]

    if not candidates:
        return [], {}

    candidates.sort(key=lambda x: (x.start_sec, x.pitch))
    frames = []
    tol_sim = CONFIG_MELODY["simultaneous_tolerance_sec"]
    for n in candidates:
        if not frames or abs(n.start_sec - frames[-1][0].start_sec) > tol_sim:
            frames.append([n])
        else:
            frames[-1].append(n)

    n_frames = len(frames)
    dp_scores = []
    dp_back = []

    mel_center_pitch = float(np.percentile([n.pitch for n in candidates], 65))

    for t_idx, frame_notes in enumerate(frames):
        curr_scores = np.zeros(len(frame_notes), dtype=float)
        curr_back = np.full(len(frame_notes), -1, dtype=int)

        for i, note_curr in enumerate(frame_notes):
            vel_norm = (note_curr.velocity - v_eff) / (127.0 - v_eff + 1e-8)
            dur_norm = min(1.0, note_curr.duration_sec / 0.8)
            reg_bias = 1.0 - min(0.5, abs(note_curr.pitch - mel_center_pitch) / 36.0)
            local_sc = 10.0 * (0.50 * vel_norm + 0.30 * dur_norm + 0.20 * reg_bias)

            if t_idx == 0:
                curr_scores[i] = local_sc
            else:
                prev_frame = frames[t_idx - 1]
                prev_scores = dp_scores[t_idx - 1]
                best_prev_sc = -1e9
                best_prev_idx = 0

                for j, note_prev in enumerate(prev_frame):
                    interval = abs(note_curr.pitch - note_prev.pitch)
                    time_gap = max(0.0, note_curr.start_sec - note_prev.end_sec)

                    jump_pen = interval * CONFIG_MELODY["jump_penalty_semitone"]
                    if interval >= 12:
                        jump_pen += CONFIG_MELODY["octave_jump_penalty"]

                    cont_bon = CONFIG_MELODY["continuity_bonus"] if (interval <= 5 and time_gap <= 0.35) else 0.0

                    trans_sc = prev_scores[j] + local_sc - jump_pen + cont_bon
                    if trans_sc > best_prev_sc:
                        best_prev_sc = trans_sc
                        best_prev_idx = j

                curr_scores[i] = best_prev_sc
                curr_back[i] = best_prev_idx

        dp_scores.append(curr_scores)
        dp_back.append(curr_back)

    melody_line = []
    curr_idx = int(np.argmax(dp_scores[-1]))
    for t_idx in range(n_frames - 1, -1, -1):
        chosen_note = frames[t_idx][curr_idx]
        melody_line.append(chosen_note)
        curr_idx = int(dp_back[t_idx][curr_idx])
    melody_line.reverse()

    for i in range(len(melody_line) - 1):
        if melody_line[i].end_sec > melody_line[i + 1].start_sec:
            melody_line[i].end_sec = max(
                melody_line[i].start_sec + CONFIG_SIGNAL["duration_min_abs_sec"],
                melody_line[i + 1].start_sec
            )

    intervals = [abs(melody_line[i+1].pitch - melody_line[i].pitch) for i in range(len(melody_line)-1)]
    mean_jump = float(np.mean(intervals)) if intervals else 0.0
    max_jump = int(np.max(intervals)) if intervals else 0
    oct_jumps = sum(1 for iv in intervals if iv >= 12)

    log.add(f"Candidatas melódicas: {len(candidates)} | Frames: {n_frames} | Melody Tracker Final: {len(melody_line)} notas")
    log.add(f"Descartadas no tracker: {len(candidates) - len(melody_line)} | Salto médio: {mean_jump:.2f} st | Maior salto: {max_jump} st | Saltos >= 1 oitava: {oct_jumps}")

    return melody_line, {"v_eff": v_eff, "p_eff": p_eff, "d_eff": d_eff}


# ---------------------------------------------------------
# 9. MOTOR HARMÔNICO POR TEMPLATES E SUB-REGIÕES
# ---------------------------------------------------------
CHORD_TEMPLATES = [
    ("maj",    [0, 4, 7],          "",       ""),
    ("min",    [0, 3, 7],          "m",      "m"),
    ("7",      [0, 4, 7, 10],      "7",      "7"),
    ("maj7",   [0, 4, 7, 11],      "7M",     "maj7"),
    ("m7",     [0, 3, 7, 10],      "m7",     "m7"),
    ("m7b5",   [0, 3, 6, 10],      "m7(b5)", "m7b5"),
    ("dim",    [0, 3, 6],          "dim",    "dim"),
    ("sus4",   [0, 5, 7],          "4",      "sus4"),
    ("7sus4",  [0, 5, 7, 10],      "4(7)",   "7sus4"),
    ("add9",   [0, 2, 4, 7],       "(add9)", "add9"),
    ("9",      [0, 2, 4, 7, 10],   "9",      "9"),
    ("m9",     [0, 2, 3, 7, 10],   "m9",     "m9"),
    ("7_13",   [0, 4, 7, 9, 10],   "7(13)",  "13"),
    ("7_b9",   [0, 1, 4, 7, 10],   "7(b9)",  "7"),
    ("7_#9",   [0, 3, 4, 7, 10],   "7(#9)",  "7"),
]

def build_template_vectors():
    vecs = []
    for name, ivs, mpb_suf, xml_suf in CHORD_TEMPLATES:
        v = np.zeros(12, dtype=float)
        for iv in ivs:
            v[iv % 12] = 1.0
        v[0] = 1.25
        v = v / np.linalg.norm(v)
        vecs.append((name, v, len(ivs), mpb_suf, xml_suf))
    return vecs

TEMPLATE_VECTORS = build_template_vectors()

def analyze_harmony_regions(dedup_notes, beat_times, pulse_ql, bar_length_ql, detected_key, log: TechnicalLogger):
    log.add("\n==============================\nADAPTIVE HARMONY & TEMPLATE ENGINE\n==============================")
    if not dedup_notes:
        return {}

    use_flats = prefer_flats_for_key(detected_key)

    vels = np.array([n.velocity for n in dedup_notes], dtype=float)
    v_h_base = CONFIG_HARMONY["velocity_base"]
    v_h_adapt = float(np.percentile(vels, 30))
    v_h_eff = float(max(CONFIG_HARMONY["velocity_min_floor"], 0.50 * v_h_base + 0.50 * v_h_adapt))

    harm_notes = [n for n in dedup_notes if n.velocity >= v_h_eff]
    if not harm_notes:
        harm_notes = dedup_notes

    weights_all = np.array([n.duration_sec * (n.velocity / 127.0) for n in harm_notes], dtype=float)
    w_med = float(np.median(weights_all))
    w_p70 = float(np.percentile(weights_all, 70))
    peso_base = CONFIG_HARMONY["peso_base"]
    peso_adapt = 0.50 * w_med + 0.50 * w_p70
    peso_eff = float(max(CONFIG_HARMONY["peso_min_floor"], 0.45 * peso_base + 0.55 * peso_adapt))

    log.add(f"VELOCITY HARMONIA -> Base: {v_h_base:.1f} | Adaptativo (P30): {v_h_adapt:.1f} | Efetivo: {v_h_eff:.1f}")
    log.add(f"PESO ACORDE       -> Base: {peso_base:.3f} | Mediana: {w_med:.3f} | P70: {w_p70:.3f} | Adaptativo: {peso_adapt:.3f} | Efetivo: {peso_eff:.3f}")

    mapped_harm = []
    for n in harm_notes:
        s_ql = seconds_to_musical_position(n.start_sec, beat_times, pulse_ql=pulse_ql)
        e_ql = seconds_to_musical_position(n.end_sec, beat_times, pulse_ql=pulse_ql)
        d_ql = max(0.25, e_ql - s_ql)
        w_val = n.duration_sec * (n.velocity / 127.0)
        mapped_harm.append((s_ql, e_ql, d_ql, n.pitch, w_val))

    if not mapped_harm:
        return {}

    max_ql = max(item[1] for item in mapped_harm)
    total_bars = int(math.ceil(max_ql / bar_length_ql)) + 1

    n_sub = 2 if bar_length_ql >= 3.0 else 1
    region_len_ql = bar_length_ql / float(n_sub)

    chords_by_bar = defaultdict(list)
    logged_regions = 0

    for b_idx in range(total_bars):
        bar_start_ql = b_idx * bar_length_ql
        sub_results = []

        for s_i in range(n_sub):
            r_start = bar_start_ql + s_i * region_len_ql
            r_end = r_start + region_len_ql

            reg_notes = []
            for s_ql, e_ql, d_ql, pitch, w_val in mapped_harm:
                ov = min(e_ql, r_end) - max(s_ql, r_start)
                if ov > 0.05:
                    metric_boost = 1.25 if abs(s_ql - r_start) < 0.35 else 1.0
                    reg_notes.append((pitch, w_val * (ov / d_ql) * metric_boost))

            if not reg_notes:
                continue

            total_reg_weight = sum(x[1] for x in reg_notes)
            if total_reg_weight < peso_eff * 0.5:
                continue

            pc_hist = np.zeros(12, dtype=float)
            lowest_pitch = 999
            bass_pc = None

            for pt, w_eff in reg_notes:
                pc_hist[pt % 12] += w_eff
                if pt < lowest_pitch:
                    lowest_pitch = pt
                    bass_pc = pt % 12

            if np.linalg.norm(pc_hist) <= 1e-8:
                continue
            pc_norm = pc_hist / np.linalg.norm(pc_hist)

            cand_scores = []
            for root_pc in range(12):
                if pc_hist[root_pc] <= 0.05 * np.max(pc_hist):
                    continue
                rot_vec = np.roll(pc_norm, -root_pc)
                for t_name, t_vec, n_notes_tpl, mpb_suf, xml_suf in TEMPLATE_VECTORS:
                    cos_sim = float(np.dot(rot_vec, t_vec))
                    complexity_pen = 0.03 * max(0, n_notes_tpl - 3)
                    bass_bon = CONFIG_HARMONY["bass_root_bonus"] if (bass_pc == root_pc) else 0.0
                    final_sc = cos_sim - complexity_pen + bass_bon
                    cand_scores.append((final_sc, root_pc, t_name, mpb_suf, xml_suf))

            if not cand_scores:
                continue

            cand_scores.sort(key=lambda x: x[0], reverse=True)
            best_sc, best_root_pc, best_tname, best_mpb_suf, best_xml_suf = cand_scores[0]

            if best_sc < CONFIG_HARMONY["template_min_confidence"]:
                continue

            root_str = pc_to_note_name(best_root_pc, use_flats=use_flats)
            xml_root_str = note_name_to_music21(root_str)

            mpb_sym = f"{root_str}{best_mpb_suf}"
            xml_sym = f"{xml_root_str}{best_xml_suf}"

            if bass_pc is not None and bass_pc != best_root_pc and lowest_pitch <= 55:
                bass_str = pc_to_note_name(bass_pc, use_flats=use_flats)
                xml_bass_str = note_name_to_music21(bass_str)
                mpb_sym = f"{mpb_sym}/{bass_str}"
                xml_sym = f"{xml_sym}/{xml_bass_str}"

            sub_results.append({
                "offset_in_bar_ql": s_i * region_len_ql,
                "mpb_sym": mpb_sym,
                "xml_sym": xml_sym,
                "confidence": round(best_sc, 3),
                "top3": [(
                    f"{pc_to_note_name(r_p, use_flats)}{m_s}", round(sc_v, 3)
                ) for sc_v, r_p, _, m_s, _ in cand_scores[:3]]
            })

            if logged_regions < 12:
                top_pcs = [f"{pc_to_note_name(i, use_flats)}:{pc_hist[i]:.2f}" for i in np.argsort(pc_hist)[::-1][:4] if pc_hist[i] > 0]
                log.add(
                    f"  Compasso {b_idx+1} (offset {s_i*region_len_ql:.1f} QL) -> "
                    f"PCs={top_pcs} | Top3={sub_results[-1]['top3']} -> Escolhido: {mpb_sym} (conf={best_sc:.3f})"
                )
                logged_regions += 1

        if len(sub_results) == 2 and sub_results[0]["mpb_sym"] == sub_results[1]["mpb_sym"]:
            chords_by_bar[b_idx] = [sub_results[0]]
        elif sub_results:
            chords_by_bar[b_idx] = sub_results

    log.add(f"Total de compassos com harmonia extraída: {len(chords_by_bar)}")
    return chords_by_bar


# ---------------------------------------------------------
# 10. QUANTIZAÇÃO NA GRADE REAL E AUDITORIA DE EXPORTAÇÃO
# ---------------------------------------------------------
def quantize_melody_on_beat_grid(melody_notes, beat_times, pulse_ql, bar_length_ql, log: TechnicalLogger):
    sub_ql = CONFIG_SIGNAL["subdivision_ql"]
    quantized_events = []
    errors_ql = []
    significant_corrections = 0

    for n in melody_notes:
        raw_start_ql = seconds_to_musical_position(n.start_sec, beat_times, pulse_ql=pulse_ql)
        raw_end_ql = seconds_to_musical_position(n.end_sec, beat_times, pulse_ql=pulse_ql)

        q_start_ql = round(raw_start_ql / sub_ql) * sub_ql
        q_end_ql = round(raw_end_ql / sub_ql) * sub_ql
        if q_end_ql <= q_start_ql:
            q_end_ql = q_start_ql + sub_ql

        q_start_ql = round(q_start_ql * 4.0) / 4.0
        q_end_ql = round(q_end_ql * 4.0) / 4.0

        err = abs(q_start_ql - raw_start_ql)
        errors_ql.append(err)
        if err > 0.08:
            significant_corrections += 1

        quantized_events.append({
            "start_ql": q_start_ql,
            "end_ql": q_end_ql,
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

    mean_err = float(np.mean(errors_ql)) if errors_ql else 0.0
    max_err = float(np.max(errors_ql)) if errors_ql else 0.0
    log.add("\n==============================\nQUANTIZATION AUDIT\n==============================")
    log.add(f"Subdivisão mínima: {sub_ql} QL | Eventos quantizados: {len(clean_events)}")
    log.add(f"Erro médio de quantização: {mean_err:.4f} QL | Maior erro: {max_err:.4f} QL | Correções > 0.08 QL: {significant_corrections}")

    return clean_events


def auditar_e_sanitizar_duracoes_musicxml(stream_obj, log: TechnicalLogger):
    corrigidos = 0
    for m in stream_obj.recurse().getElementsByClass(music21.stream.Measure):
        m_num = m.number
        for el in m.notesAndRests:
            ql = float(el.quarterLength)
            dtype = getattr(el.duration, "type", "")
            if ql <= 0.0 or dtype in ("inexpressible", "zero", "complex"):
                novo_ql = round(ql * 4.0) / 4.0
                if novo_ql <= 0.0:
                    novo_ql = 0.25
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
# 11. CABEÇALHO E INTERFACE WEB (LAYOUT ORIGINAL INTACTO)
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
        help="No modo Automático, o sistema estima o compasso. Se conhecer a métrica exata da obra, selecione manualmente."
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
        log.add(f"CONFIG_SIGNAL Base: {CONFIG_SIGNAL}")
        log.add(f"CONFIG_METER Base: {CONFIG_METER}")
        log.add(f"CONFIG_MELODY Base: {CONFIG_MELODY}")
        log.add(f"CONFIG_HARMONY Base: {CONFIG_HARMONY}")

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

                sample_lim_45 = min(len(y_full), int(45.0 * sr))
                raw_bpm_45, _, onset_45 = estimate_local_bpm_and_beats(y_full[:sample_lim_45], sr)

                local_bpms = []
                for pos in CONFIG_METER["segment_positions"]:
                    c_i = int(pos * len(y_full))
                    w_i = int(CONFIG_METER["segment_window_sec"] * sr)
                    s0, s1 = max(0, c_i - w_i // 2), min(len(y_full), c_i + w_i // 2)
                    lb, _, _ = estimate_local_bpm_and_beats(y_full[s0:s1], sr)
                    local_bpms.append(lb)

                bpm = resolve_tactus_level(raw_bpm_45, local_bpms, onset_45, sr, log)
                del onset_45
                gc.collect()

                beat_times, beat_stability, _ = build_real_beat_grid(y_full, sr, bpm, total_duration, log)

                if compasso_selecionado != "Automático":
                    time_sig = compasso_selecionado
                    compasso_exibicao = f"{time_sig} (Manual)"
                    meter_conf_str = "manual"
                    meter_ev_q = 1.0
                    pulse_ql = {"2/4": 1.0, "3/4": 1.0, "4/4": 1.0, "6/8": 1.5, "3/2": 2.0}.get(time_sig, 1.0)
                    log.add(f"\n[METER] Definido manualmente pelo usuário: {time_sig}")
                else:
                    meter_res = detect_meter_a3(y_full, sr, bpm, beat_stability, log)
                    time_sig = meter_res["time_sig"]
                    pulse_ql = meter_res["pulse_ql"]
                    meter_conf_str = meter_res["confidence_label"]
                    meter_ev_q = meter_res["evidence_quality"]
                    if meter_res["status"] == "detected":
                        compasso_exibicao = f"{time_sig} (Auto - Conf: {meter_conf_str})"
                    else:
                        compasso_exibicao = f"Não determinado (Fallback {time_sig})"

                num, den = map(int, time_sig.split("/"))
                bar_length_ql = float(num) * (4.0 / float(den))

                raw_notes, dedup_notes = infer_notes_with_overlap(y_full, sr, session_dir, ai_model, log)

                if not dedup_notes:
                    log.add_error("Nenhuma nota musical clara foi detectada pelo Basic Pitch.")
                    st.session_state["log_text"] = log.get_text()
                    st.error("Nenhuma nota musical clara foi detectada neste arquivo de áudio.")
                else:
                    melody_notes, mel_calib = track_melody_viterbi(dedup_notes, log)

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

                        quant_events = quantize_melody_on_beat_grid(melody_notes, beat_times, pulse_ql, bar_length_ql, log)

                        score = music21.stream.Score()
                        score.insert(0, music21.metadata.Metadata())
                        score.metadata.title = os.path.splitext(uploaded_file.name)[0]
                        score.metadata.composer = "Transcrito por Partitura.ai"

                        part = music21.stream.Part()
                        part.insert(0, music21.clef.TrebleClef())
                        part.insert(0, music21.meter.TimeSignature(time_sig))
                        part.insert(0, music21.tempo.MetronomeMark(number=bpm))

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
                            dur_ql = round(dur_ql * 4.0) / 4.0

                            el = music21.note.Note(ev["pitch"])
                            el.quarterLength = dur_ql
                            part.insert(float(start_ql), el)

                        detected_key, key_success, key_conf = detectar_centro_tonal_global(
                            part, y_mono=y_full, sr=sr, all_notes=dedup_notes, log=log
                        )
                        part.insert(0, music21.key.KeySignature(detected_key.sharps))

                        del y_full
                        gc.collect()

                        part.makeMeasures(inPlace=True)
                        part.makeTies(inPlace=True)
                        corr_pre = auditar_e_sanitizar_duracoes_musicxml(part, log)

                        chords_by_bar = analyze_harmony_regions(
                            dedup_notes, beat_times, pulse_ql, bar_length_ql, detected_key, log
                        )

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

                        if key_success:
                            tonic_pt = sanitizar_nome_tom(detected_key.tonic.name)
                            mode_pt = "Maior" if detected_key.mode == "major" else "Menor"
                            tom_exibicao = f"{tonic_pt} {mode_pt}"
                        else:
                            tom_exibicao = "Não determinado"

                        midi_bytes = midi_path.read_bytes()
                        xml_bytes = xml_path.read_bytes()

                        midi_path.unlink(missing_ok=True)
                        xml_path.unlink(missing_ok=True)

                        log.add("\n===================================")
                        log.add("PARTITURA.AI — RESUMO")
                        log.add("===================================")
                        log.add(f"Arquivo: {uploaded_file.name}")
                        log.add(f"Duração: {total_duration:.2f}s")
                        log.add(f"BPM bruto (45s): {raw_bpm_45:.2f} | Tactus final: {bpm:.1f} BPM")
                        log.add(f"Compasso: {compasso_exibicao} | Confidence: {meter_conf_str} | Evidence Quality: {meter_ev_q:.3f}")
                        log.add(f"Tom: {tom_exibicao} | Confidence: {key_conf:.3f}")
                        log.add(f"Basic Pitch raw: {len(raw_notes)} | Dedup: {len(dedup_notes)} | Melody notes finais: {len(melody_notes)}")
                        log.add(f"Compassos gerados: {len(measures)} | Regiões harmônicas com cifra: {len(chords_by_bar)} | Cifras inseridas: {total_cifras_inseridas}")
                        log.add(f"Durações inexpressible corrigidas: {corr_pre + corr_post}")
                        log.add(f"MIDI: {midi_status} | MusicXML: {xml_status} | PDF: {pdf_status}")
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
