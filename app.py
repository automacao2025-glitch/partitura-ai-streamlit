import sys
import os
import io
import gc
import math
import uuid
import shutil
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

# ---------------------------------------------------------
# 1. PAINEL CENTRAL DE CALIBRAÇÃO (CONFIG_CALIBRACAO - TESTE A/A2)
# ---------------------------------------------------------
CONFIG_CALIBRACAO = {
    "CORTE_PITCH_MELODIA": 55,             # 55 (reduzir notas graves do acompanhamento)
    "MAX_NOTAS_CLUSTER_VISUAL": 1,         # 1 (gerar linha melódica monofônica limpa)
    "VELOCITY_MIN_MELODIA": 50,            # 50 (eliminar notas fracas e ressonâncias)
    "VELOCITY_MIN_HARMONIA": 45,           # 45 (reduzir contaminação na análise harmônica)
    "DURACAO_MIN_NOTA_SEG": 0.11,          # 0.11 (filtrar transientes curtos)
    "SUBDIVISAO_MINIMA_QL": 0.25,          # Semicolcheia como grade mínima
    "PESO_MINIMO_ACORDE": 0.22,            # Peso de consolidação harmônica
}

# ---------------------------------------------------------
# PESOS DO CLASSIFICADOR MULTIFATOR DE COMPASSO (TESTE A2)
# ---------------------------------------------------------
PESOS_DETECCAO_METRICA = {
    "w_periodicity": 0.40,
    "w_accent_phase": 0.25,
    "w_harmonic_change": 0.15,
    "w_recurrence": 0.10,
    "w_segment_consistency": 0.10,
    "limiar_margem_ambiguidade": 0.075     # Margem mínima para aceitar automaticamente
}

# ---------------------------------------------------------
# CARREGAMENTO OTIMIZADO VIA ONNX RUNTIME
# ---------------------------------------------------------
@st.cache_resource
def get_inference_model():
    try:
        onnx_model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
        return Model(onnx_model_path)
    except Exception:
        return Model(ICASSP_2022_MODEL_PATH)

class MockNote:
    def __init__(self, start_sec, end_sec, pitch, velocity):
        self.start_sec = float(start_sec)
        self.end_sec = float(end_sec)
        self.pitch = int(pitch)
        self.velocity = int(velocity)

# ---------------------------------------------------------
# 2. DETECTOR HÍBRIDO LEVE DE COMPASSO (TESTE A2)
# ---------------------------------------------------------
def _analisar_segmento_metrico(y_seg, sr, bpm):
    """Analisa um segmento de áudio e pontua 2/4, 3/4, 4/4 e 6/8."""
    candidates = ["2/4", "3/4", "4/4", "6/8"]
    scores = {c: 0.0 for c in candidates}
    best_phase = {c: 0 for c in candidates}

    if len(y_seg) < sr * 3:
        return scores, best_phase

    # HPSS leve
    try:
        y_harm, y_perc = librosa.effects.hpss(y_seg)
    except Exception:
        y_harm, y_perc = y_seg, y_seg

    # Onset Envelope
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)
    if len(onset_env) == 0 or np.max(onset_env) == 0:
        return scores, best_phase

    onset_env = onset_env / (np.max(onset_env) + 1e-8)

    # Beat tracking local
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length, bpm=bpm)
    if len(beats) < 8:
        return scores, best_phase

    # Periodicidade via Autocorrelação de Tempogram
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    ac_global = np.mean(tempogram, axis=1)
    
    # 1. Periodicidade Rítmica
    # Razão de pulso a cada 2, 3 e 4 beats
    n_frames = len(onset_env)
    sec_per_beat = 60.0 / bpm
    frames_per_beat = max(1, int((sec_per_beat * sr) / hop_length))

    def _pegar_energia_lag(mult):
        lag = int(round(frames_per_beat * mult))
        if lag < len(ac_global):
            return float(ac_global[lag])
        return 0.0

    e_2 = _pegar_energia_lag(2.0)
    e_3 = _pegar_energia_lag(3.0)
    e_4 = _pegar_energia_lag(4.0)
    e_total = e_2 + e_3 + e_4 + 1e-8

    p_2_4 = (e_2 * 0.6 + e_4 * 0.4) / e_total
    p_3_4 = e_3 / e_total
    p_4_4 = (e_4 * 0.7 + e_2 * 0.3) / e_total

    # Subdivisão binária vs ternária para 3/4 x 6/8
    # 6/8 tem 2 tempos pontuados compostos (subdivisão ternária de cada beat)
    # 3/4 tem 3 tempos simples com subdivisão binária
    subdiv_ternaria = 0.0
    lag_tern = int(round(frames_per_beat / 3.0))
    lag_bin = int(round(frames_per_beat / 2.0))
    if lag_tern < len(ac_global) and lag_bin < len(ac_global):
        e_tern = ac_global[lag_tern]
        e_bin = ac_global[lag_bin]
        subdiv_ternaria = float(e_tern / (e_tern + e_bin + 1e-8))

    p_6_8 = (e_2 * 0.5 + e_3 * 0.5) / e_total
    if subdiv_ternaria > 0.52:
        p_6_8 *= 1.3
        p_3_4 *= 0.8
    else:
        p_3_4 *= 1.2
        p_6_8 *= 0.7

    # 2. Teste de Fase / Downbeat e Acentuação
    # Intensidade do onset_env sincronizado com cada beat
    beat_strengths = []
    for b in beats:
        b_idx = int(b)
        if b_idx < len(onset_env):
            # Janela de tolerância em torno do beat
            w_start = max(0, b_idx - 2)
            w_end = min(len(onset_env), b_idx + 3)
            beat_strengths.append(float(np.max(onset_env[w_start:w_end])))
        else:
            beat_strengths.append(0.0)
    beat_strengths = np.array(beat_strengths)

    # Mudança harmônica via Chroma nos beats
    try:
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length)
        chroma_beats = librosa.util.sync(chroma, beats)
        # Distância euclidiana entre frames de chroma consecutivos sincronizados
        diff_chroma = np.sqrt(np.sum(np.diff(chroma_beats, axis=1)**2, axis=0))
        diff_chroma = np.pad(diff_chroma, (1, 0), mode='edge')
        diff_chroma = diff_chroma / (np.max(diff_chroma) + 1e-8)
    except Exception:
        diff_chroma = np.zeros(len(beats))

    def _avaliar_fase_e_acentos(periodo):
        melhor_fase = 0
        melhor_score_fase = -1.0
        melhor_h_score = 0.0

        for fase in range(periodo):
            indices = np.arange(fase, len(beats), periodo)
            if len(indices) < 2:
                continue
            downbeat_accents = beat_strengths[indices]
            other_accents = np.delete(beat_strengths, indices)

            media_down = np.mean(downbeat_accents) if len(downbeat_accents) > 0 else 0.0
            media_outros = np.mean(other_accents) if len(other_accents) > 0 else 0.0
            score_acc = max(0.0, media_down - media_outros * 0.7)

            # Mudança harmônica no downbeat
            h_acc = np.mean(diff_chroma[indices]) if len(indices) > 0 else 0.0

            comb_fase = score_acc * 0.7 + h_acc * 0.3
            if comb_fase > melhor_score_fase:
                melhor_score_fase = comb_fase
                melhor_fase = fase
                melhor_h_score = h_acc

        return max(0.0, melhor_score_fase), melhor_fase, melhor_h_score

    acc_2, f_2, h_2 = _avaliar_fase_e_acentos(2)
    acc_3, f_3, h_3 = _avaliar_fase_e_acentos(3)
    acc_4, f_4, h_4 = _avaliar_fase_e_acentos(4)
    acc_6, f_6, h_6 = _avaliar_fase_e_acentos(2) # 6/8 tem 2 pulsos compostos

    best_phase["2/4"] = f_2
    best_phase["3/4"] = f_3
    best_phase["4/4"] = f_4
    best_phase["6/8"] = f_6

    # 3. Recorrência / Beat Similarity Simplificada (Para desempate 2/4 vs 4/4)
    rec_2 = 0.5
    rec_4 = 0.5
    if len(beats) >= 16:
        try:
            # Matriz de similaridade cosseno de chroma_beats
            norm_chroma = chroma_beats / (np.linalg.norm(chroma_beats, axis=0, keepdims=True) + 1e-8)
            sim_matrix = np.dot(norm_chroma.T, norm_chroma)
            # Avalia autocorrelação de diagonais em lag=2 e lag=4
            n_diag = min(len(sim_matrix), 32)
            diag_2 = [sim_matrix[i, i+2] for i in range(n_diag - 2)]
            diag_4 = [sim_matrix[i, i+4] for i in range(n_diag - 4)]
            mean_d2 = float(np.mean(diag_2)) if diag_2 else 0.5
            mean_d4 = float(np.mean(diag_4)) if diag_4 else 0.5
            total_rec = mean_d2 + mean_d4 + 1e-8
            rec_2 = mean_d2 / total_rec
            rec_4 = mean_d4 / total_rec
        except Exception:
            pass

    # Consolidação dos Scores Multifatoriais do Segmento
    w = PESOS_DETECCAO_METRICA
    raw_scores = {
        "2/4": w["w_periodicity"] * p_2_4 + w["w_accent_phase"] * acc_2 + w["w_harmonic_change"] * h_2 + w["w_recurrence"] * rec_2,
        "3/4": w["w_periodicity"] * p_3_4 + w["w_accent_phase"] * acc_3 + w["w_harmonic_change"] * h_3 + w["w_recurrence"] * 0.5,
        "4/4": w["w_periodicity"] * p_4_4 + w["w_accent_phase"] * acc_4 + w["w_harmonic_change"] * h_4 + w["w_recurrence"] * rec_4,
        "6/8": w["w_periodicity"] * p_6_8 + w["w_accent_phase"] * acc_6 + w["w_harmonic_change"] * h_6 + w["w_recurrence"] * 0.5,
    }

    # Normalização
    s_soma = sum(raw_scores.values()) + 1e-8
    for c in candidates:
        scores[c] = float(raw_scores[c] / s_soma)

    return scores, best_phase


def detect_time_signature_light(y, sr, bpm):
    """
    Motor híbrido leve de detecção métrica multi-segmento.
    Retorna métrica detectada, confiança, scores e auditoria completa.
    """
    total_len = len(y)
    total_sec = total_len / float(sr)

    # Três regiões representativas: 20%, 50% e 80%
    window_sec = 18.0
    window_samples = int(window_sec * sr)

    positions = [0.20, 0.50, 0.80]
    segment_scores = []
    segment_phases = []
    segment_winners = []

    print(f"\n[Meter Detection] BPM: {bpm} | Duração: {total_sec:.1f}s")

    for idx, pos in enumerate(positions):
        center_sample = int(pos * total_len)
        start_s = max(0, center_sample - window_samples // 2)
        end_s = min(total_len, start_s + window_samples)
        y_seg = y[start_s:end_s]

        s_scores, s_phases = _analisar_segmento_metrico(y_seg, sr, bpm)
        segment_scores.append(s_scores)
        segment_phases.append(s_phases)
        
        # Vencedor local
        winner = max(s_scores.items(), key=lambda kv: kv[1])[0]
        segment_winners.append(winner)

        print(
            f"  Segmento {idx + 1} (~{int(pos*100)}%): "
            f"2/4={s_scores['2/4']:.3f} | 3/4={s_scores['3/4']:.3f} | "
            f"4/4={s_scores['4/4']:.3f} | 6/8={s_scores['6/8']:.3f} -> Líder: {winner}"
        )

    # Média agregada ponderada dos segmentos
    candidates = ["2/4", "3/4", "4/4", "6/8"]
    global_scores = {c: 0.0 for c in candidates}
    for c in candidates:
        valores = [seg[c] for seg in segment_scores]
        global_scores[c] = float(np.mean(valores))

    # Concordância entre segmentos
    most_common_winner = max(set(segment_winners), key=segment_winners.count)
    agreement_count = segment_winners.count(most_common_winner)
    agreement_ratio = agreement_count / float(len(segment_winners))

    # Bônus de consistência inter-segmentos
    for c in candidates:
        if c == most_common_winner:
            global_scores[c] += PESOS_DETECCAO_METRICA["w_segment_consistency"] * agreement_ratio

    # Normalização final dos scores globais
    g_soma = sum(global_scores.values()) + 1e-8
    for c in candidates:
        global_scores[c] = float(global_scores[c] / g_soma)

    # Ranking
    ranking = sorted(global_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_cand, best_score = ranking[0]
    second_cand, second_score = ranking[1]
    margin = best_score - second_score

    # Decisão de Confiança e Política de Abstention
    limiar_margem = PESOS_DETECCAO_METRICA["limiar_margem_ambiguidade"]
    
    if margin >= limiar_margem and agreement_ratio >= 0.66:
        confidence_label = "alta"
        status = "detected"
        detected_time_sig = best_cand
    elif margin >= (limiar_margem * 0.6) and agreement_ratio >= 0.66:
        confidence_label = "média"
        status = "detected"
        detected_time_sig = best_cand
    else:
        confidence_label = "inconclusiva"
        status = "ambiguous"
        detected_time_sig = "4/4"  # Fallback técnico seguro para não paralisar o pipeline

    chosen_phase = segment_phases[1].get(best_cand, 0)

    print(
        f"[Meter Detection] Resultado Global:\n"
        f"  Scores: 2/4={global_scores['2/4']:.3f} | 3/4={global_scores['3/4']:.3f} | "
        f"4/4={global_scores['4/4']:.3f} | 6/8={global_scores['6/8']:.3f}\n"
        f"  Melhor: {best_cand} ({best_score:.3f}) | Segundo: {second_cand} ({second_score:.3f}) | "
        f"Margem: {margin:.3f} | Concordância: {agreement_ratio:.0%} | "
        f"Confiança: {confidence_label} | Fase: {chosen_phase} | Status: {status}\n"
    )

    return {
        "time_sig": detected_time_sig,
        "status": status,
        "best_cand": best_cand,
        "best_score": round(best_score, 3),
        "second_cand": second_cand,
        "second_score": round(second_score, 3),
        "margin": round(margin, 3),
        "segment_agreement": round(agreement_ratio, 2),
        "confidence_label": confidence_label,
        "chosen_phase": chosen_phase,
        "scores": {k: round(v, 3) for k, v in global_scores.items()}
    }


# ---------------------------------------------------------
# 3. CONVERSORES SEGUROS E SANITIZADORES
# ---------------------------------------------------------
def format_root_pt_br(pitch_name: str) -> str:
    """Converte flats do music21 ('B-', 'E-') para o padrão de cifra ('Bb', 'Eb')."""
    if not pitch_name:
        return "C"
    if len(pitch_name) > 1 and pitch_name[1] == '-':
        return pitch_name[0] + 'b' + pitch_name[2:]
    return pitch_name

def sanitizar_nome_tom(tonic_name: str) -> str:
    """Converte representações do music21 para exibição em português limpa (com ♭ e #)."""
    if not tonic_name:
        return "Não determinado"
    clean = tonic_name.replace('-', 'b')
    mapa_pt = {
        'C': 'Dó', 'C#': 'Dó#', 'Db': 'Ré♭',
        'D': 'Ré', 'D#': 'Ré#', 'Eb': 'Mi♭',
        'E': 'Mi', 'Fb': 'Fá♭', 'E#': 'Mi#',
        'F': 'Fá', 'F#': 'Fá#', 'Gb': 'Sol♭',
        'G': 'Sol', 'G#': 'Sol#', 'Ab': 'Lá♭',
        'A': 'Lá', 'A#': 'Lá#', 'Bb': 'Si♭',
        'B': 'Si', 'Cb': 'Dó♭', 'B#': 'Si#'
    }
    return mapa_pt.get(clean, clean)

def sanitize_chord_figure(ch: music21.chord.Chord) -> tuple:
    """
    Traduz o acorde do music21 para o padrão MPB/FUNARTE limpo,
    preservando extensões (7M, 7, m7, m7(b5), 9, 13, #9, b9) e baixos invertidos (/).
    Retorna (cifra_mpb, cifra_xml).
    """
    try:
        if not ch or not ch.pitches:
            return None, None

        root_pitch = ch.root()
        root_str = format_root_pt_br(root_pitch.name)
        root_pc = root_pitch.pitchClass
        bass_pitch = ch.bass()
        bass_str = format_root_pt_br(bass_pitch.name) if bass_pitch else root_str

        pcs = set(p.pitchClass for p in ch.pitches)
        intervals = set((pc - root_pc) % 12 for pc in pcs)

        has_m3 = 3 in intervals
        has_M3 = 4 in intervals
        has_d5 = 6 in intervals
        has_P5 = 7 in intervals
        has_m7 = 10 in intervals
        has_M7 = 11 in intervals
        has_9 = 2 in intervals
        has_sharp9 = 3 in intervals and has_M3
        has_b9 = 1 in intervals
        has_13 = 9 in intervals

        if has_M3:
            if has_M7:
                mpb_sym = f"{root_str}7M"
                xml_sym = f"{root_str}maj7"
            elif has_m7:
                if has_sharp9:
                    mpb_sym = f"{root_str}7(#9)"
                    xml_sym = f"{root_str}7"
                elif has_b9:
                    mpb_sym = f"{root_str}7(b9)"
                    xml_sym = f"{root_str}7"
                elif has_13:
                    mpb_sym = f"{root_str}7(13)"
                    xml_sym = f"{root_str}13"
                elif has_9:
                    mpb_sym = f"{root_str}9"
                    xml_sym = f"{root_str}9"
                else:
                    mpb_sym = f"{root_str}7"
                    xml_sym = f"{root_str}7"
            elif has_9:
                mpb_sym = f"{root_str}(add9)"
                xml_sym = f"{root_str}"
            else:
                mpb_sym = f"{root_str}"
                xml_sym = f"{root_str}"
        elif has_m3:
            if has_d5 and not has_P5:
                if has_m7:
                    mpb_sym = f"{root_str}m7(b5)"
                    xml_sym = f"{root_str}m7b5"
                else:
                    mpb_sym = f"{root_str}dim"
                    xml_sym = f"{root_str}dim"
            elif has_m7:
                if has_9:
                    mpb_sym = f"{root_str}m9"
                    xml_sym = f"{root_str}m9"
                else:
                    mpb_sym = f"{root_str}m7"
                    xml_sym = f"{root_str}m7"
            elif has_M7:
                mpb_sym = f"{root_str}m(7M)"
                xml_sym = f"{root_str}m"
            else:
                mpb_sym = f"{root_str}m"
                xml_sym = f"{root_str}m"
        else:
            if 5 in intervals:
                mpb_sym = f"{root_str}4(7)" if has_m7 else f"{root_str}4"
                xml_sym = f"{root_str}sus4"
            else:
                mpb_sym = f"{root_str}"
                xml_sym = f"{root_str}"

        if bass_str != root_str:
            mpb_sym = f"{mpb_sym}/{bass_str}"
            xml_sym = f"{xml_sym}/{bass_str}"

        return mpb_sym, xml_sym
    except Exception:
        return None, None

def quantizar_e_filtrar_fluxos(all_notes, bpm, time_sig, config):
    """
    Passagem única O(N):
    - Mantém start_sec / end_sec estritamente para o MIDI.
    - Converte start_ql / end_ql estritamente para o music21.
    - Calcula bar_length_ql considerando numerador e denominador reais.
    """
    sec_per_quarter = 60.0 / bpm
    num, den = map(int, time_sig.split("/"))
    bar_length_ql = float(num) * (4.0 / float(den))
    grid_step_ql = float(config["SUBDIVISAO_MINIMA_QL"])

    melody_clusters = defaultdict(list)
    harmonic_stream_by_bar = defaultdict(list)

    for n in all_notes:
        dur_sec = n.end_sec - n.start_sec
        if dur_sec >= config["DURACAO_MIN_NOTA_SEG"]:
            start_ql_raw = n.start_sec / sec_per_quarter
            end_ql_raw = n.end_sec / sec_per_quarter

            q_start_ql = round(start_ql_raw / grid_step_ql) * grid_step_ql
            q_end_ql = round(end_ql_raw / grid_step_ql) * grid_step_ql
            if q_end_ql <= q_start_ql:
                q_end_ql = q_start_ql + grid_step_ql

            q_start_ql = round(q_start_ql * 4.0) / 4.0
            q_end_ql = round(q_end_ql * 4.0) / 4.0

            bar_idx = int(q_start_ql // bar_length_ql)

            if n.velocity >= config["VELOCITY_MIN_HARMONIA"]:
                weight = dur_sec * (n.velocity / 127.0)
                harmonic_stream_by_bar[bar_idx].append({
                    'pitch': int(n.pitch),
                    'weight': weight
                })

            if n.velocity >= config["VELOCITY_MIN_MELODIA"] and n.pitch >= config["CORTE_PITCH_MELODIA"]:
                melody_pitch = int(n.pitch)
                while melody_pitch < 55:
                    melody_pitch += 12
                while melody_pitch > 88:
                    melody_pitch -= 12

                melody_clusters[q_start_ql].append({
                    'start_ql': q_start_ql,
                    'end_ql': q_end_ql,
                    'start_sec': n.start_sec,
                    'end_sec': n.end_sec,
                    'pitch': melody_pitch,
                    'velocity': n.velocity
                })

    return melody_clusters, harmonic_stream_by_bar, bar_length_ql

def detectar_centro_tonal_global(part):
    """Detecta a tonalidade com fallback técnico transparente."""
    try:
        if len(list(part.recurse().notes)) < 4:
            return music21.key.Key('C'), False
        ks = music21.analysis.discrete.KrumhanslSchmuckler()
        key_res = ks.getSolution(part)
        if key_res and getattr(key_res, 'tonic', None) is not None:
            return key_res, True
    except Exception as key_err:
        print(f"[Auditoria Tonal] Falha na detecção de centro tonal: {key_err}")
    return music21.key.Key('C'), False

def auditar_e_sanitizar_duracoes_musicxml(stream_obj):
    """
    Varre o Score/Part antes da exportação MusicXML, registrando e corrigindo
    qualquer elemento com duration.type == 'inexpressible' ou quarterLength inválido.
    """
    for m in stream_obj.recurse().getElementsByClass(music21.stream.Measure):
        m_num = m.number
        for el in m.notesAndRests:
            ql = float(el.quarterLength)
            dtype = getattr(el.duration, 'type', '')
            if ql <= 0.0 or dtype == 'inexpressible' or dtype == 'zero':
                novo_ql = round(ql * 4.0) / 4.0
                if novo_ql <= 0.0:
                    novo_ql = 0.25
                print(
                    f"[Auditoria MusicXML] Corrigido elemento {type(el).__name__} "
                    f"na medida {m_num}, offset {el.offset}: "
                    f"ql={ql} (type={dtype}) -> novo_ql={novo_ql}"
                )
                el.quarterLength = novo_ql


# ---------------------------------------------------------
# 4. CABEÇALHO E INTERFACE WEB
# ---------------------------------------------------------
st.title("🎵 Partitura.ai")
st.markdown("### Sua Música em Partitura com Melodia e Cifras Harmônicas")
st.write("Envie um arquivo de áudio para transcrever a melodia, extrair o campo harmônico avançado (cifras e acordes base), visualizar a partitura e exportar em **PDF**, **MusicXML** e **MIDI**.")

with st.sidebar:
    st.markdown("### ⚙️ Configurações de Métrica")
    compasso_selecionado = st.selectbox(
        "Fórmula de Compasso:",
        ["Automático", "2/4", "3/4", "4/4", "6/8"],
        index=0,
        help="No modo Automático, o sistema executa detecção híbrida inteligente. Se conhecer o compasso de antemão, selecione manualmente."
    )

uploaded_file = st.file_uploader("Escolha um arquivo de áudio", type=["mp3", "wav", "m4a", "flac", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file.getvalue())
    
    if st.button("✨ Gerar Partitura Completa com Cifras", type="primary"):
        # Limpeza preventiva do session_state antes de iniciar nova transcrição
        for k in ["result_ready", "midi_bytes", "xml_bytes", "pdf_bytes", "page_images", "info_text"]:
            if k in st.session_state:
                del st.session_state[k]

        session_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(uploaded_file.name)[1].lower() or ".wav"
        safe_input_path = UPLOAD_DIR / f"raw_{session_id}{ext}"
        session_dir = UPLOAD_DIR / f"session_{session_id}"
        session_dir.mkdir(exist_ok=True)

        with st.spinner("⚡ Processando áudio no Render (análise métrica, IA e LilyPond)... Aguarde."):
            try:
                ai_model = get_inference_model()

                with open(safe_input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                y_full, sr = librosa.load(str(safe_input_path), sr=22050, mono=True)
                total_duration = len(y_full) / sr

                # Detecção e normalização robusta de BPM
                sample_limit = min(len(y_full), int(45.0 * sr))
                onset_env = librosa.onset.onset_strength(y=y_full[:sample_limit], sr=sr)
                tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                
                tempo_arr = np.asarray(tempo).reshape(-1)
                bpm = float(tempo_arr[0]) if tempo_arr.size > 0 else 120.0
                if not math.isfinite(bpm) or bpm <= 0.0:
                    bpm = 120.0
                if bpm > 165.0:
                    bpm /= 2.0
                elif bpm < 58.0:
                    bpm *= 2.0
                bpm = round(bpm, 1)

                del onset_env
                gc.collect()

                # ---------------------------------------------------------
                # DETERMINAÇÃO DO COMPASSO (TESTE A2)
                # ---------------------------------------------------------
                if compasso_selecionado != "Automático":
                    time_sig = compasso_selecionado
                    compasso_exibicao = f"{time_sig} (Manual)"
                else:
                    # Executa análise métrica híbrida multi-segmento
                    res_meter = detect_time_signature_light(y_full, sr, bpm)
                    if res_meter["status"] == "detected":
                        time_sig = res_meter["time_sig"]
                        compasso_exibicao = f"{time_sig} (Auto - Confiança {res_meter['confidence_label']})"
                    else:
                        time_sig = "4/4"  # Fallback técnico seguro para construção da partitura
                        compasso_exibicao = "Não determinado (fallback 4/4)"

                # Inferência do Basic Pitch em blocos de 60s
                chunk_duration = 60.0
                all_notes = []
                start_time = 0.0
                chunk_idx = 0

                while start_time < total_duration:
                    end_time = min(start_time + chunk_duration, total_duration)
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    y_chunk = y_full[start_sample:end_sample]
                    if len(y_chunk) < int(sr * 1.5):
                        break

                    chunk_file = session_dir / f"chunk_{chunk_idx}.wav"
                    sf.write(str(chunk_file), y_chunk, sr, subtype='PCM_16')

                    _, midi_data_chunk, _ = predict(str(chunk_file.resolve()), ai_model)
                    
                    for inst in midi_data_chunk.instruments:
                        for n in inst.notes:
                            all_notes.append(MockNote(
                                start_sec=n.start + start_time,
                                end_sec=n.end + start_time,
                                pitch=n.pitch,
                                velocity=n.velocity
                            ))

                    try:
                        chunk_file.unlink()
                    except Exception:
                        pass
                    del y_chunk, midi_data_chunk
                    gc.collect()

                    start_time += chunk_duration
                    chunk_idx += 1

                del y_full
                gc.collect()

                if not all_notes:
                    st.error("Nenhuma nota musical clara foi detectada neste arquivo de áudio.")
                else:
                    melody_clusters, harmonic_stream_by_bar, bar_length_ql = quantizar_e_filtrar_fluxos(
                        all_notes, bpm, time_sig, CONFIG_CALIBRACAO
                    )

                    if not melody_clusters:
                        st.warning(
                            "Foram detectadas notas no áudio, mas nenhuma atingiu os limiares do filtro melódico do Teste A. "
                            "Ajuste os filtros ou selecione o modo manual se necessário."
                        )
                    else:
                        # ---------------------------------------------------------
                        # EXPORTAÇÃO MIDI (Usa estritamente segundos: start_sec / end_sec)
                        # ---------------------------------------------------------
                        midi_path = OUTPUT_DIR / f"partitura_{session_id}.mid"
                        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
                        inst = pretty_midi.Instrument(program=0)
                        for _, cluster in sorted(melody_clusters.items()):
                            for item in cluster:
                                inst.notes.append(pretty_midi.Note(
                                    velocity=int(item['velocity']),
                                    pitch=int(item['pitch']),
                                    start=float(item['start_sec']),
                                    end=float(item['end_sec'])
                                ))
                        pm.instruments.append(inst)
                        pm.write(str(midi_path))

                        # ---------------------------------------------------------
                        # CONSTRUÇÃO DO SCORE NO MUSIC21 (Usa estritamente QuarterLength)
                        # ---------------------------------------------------------
                        score = music21.stream.Score()
                        score.insert(0, music21.metadata.Metadata())
                        score.metadata.title = os.path.splitext(uploaded_file.name)[0]
                        score.metadata.composer = "Transcrito por Partitura.ai"

                        part = music21.stream.Part()
                        part.insert(0, music21.clef.TrebleClef())
                        part.insert(0, music21.meter.TimeSignature(time_sig))
                        part.insert(0, music21.tempo.MetronomeMark(number=bpm))

                        sorted_offsets_ql = sorted(melody_clusters.keys())
                        for idx, start_ql in enumerate(sorted_offsets_ql):
                            cluster = melody_clusters[start_ql]
                            raw_dur_ql = max((item['end_ql'] - item['start_ql']) for item in cluster)
                            
                            dur_ql = round(raw_dur_ql * 4.0) / 4.0
                            if dur_ql <= 0.0:
                                dur_ql = 0.25

                            if idx + 1 < len(sorted_offsets_ql):
                                next_start_ql = sorted_offsets_ql[idx + 1]
                                max_allowed_ql = round((next_start_ql - start_ql) * 4.0) / 4.0
                                if max_allowed_ql >= 0.25 and dur_ql > max_allowed_ql:
                                    dur_ql = max_allowed_ql

                            dur_ql = max(0.25, min(bar_length_ql, dur_ql))
                            dur_ql = round(dur_ql * 4.0) / 4.0

                            pitches_in_cluster = sorted(list({int(item['pitch']) for item in cluster}))[-CONFIG_CALIBRACAO["MAX_NOTAS_CLUSTER_VISUAL"]:]

                            if len(pitches_in_cluster) == 1:
                                el = music21.note.Note(pitches_in_cluster[0])
                            else:
                                el = music21.chord.Chord(pitches_in_cluster)

                            el.quarterLength = dur_ql
                            part.insert(float(start_ql), el)

                        # Detecção Tonal Global com fallback seguro
                        detected_key, key_success = detectar_centro_tonal_global(part)
                        part.insert(0, music21.key.KeySignature(detected_key.sharps))
                        
                        # Sequência Canônica: makeMeasures -> makeTies -> Auditoria
                        part.makeMeasures(inPlace=True)
                        part.makeTies(inPlace=True)
                        auditar_e_sanitizar_duracoes_musicxml(part)

                        # Extração e ancoragem das cifras por compasso
                        measures = list(part.getElementsByClass(music21.stream.Measure))
                        xml_chords_to_insert = []
                        last_mpb_sym = None

                        for idx_m, m in enumerate(measures):
                            bar_notes_data = harmonic_stream_by_bar.get(idx_m, [])
                            if bar_notes_data:
                                valid_pitches = [
                                    item['pitch'] for item in bar_notes_data
                                    if item['weight'] >= CONFIG_CALIBRACAO["PESO_MINIMO_ACORDE"]
                                ]
                                if not valid_pitches:
                                    valid_pitches = [item['pitch'] for item in bar_notes_data]

                                if valid_pitches:
                                    ch = music21.chord.Chord(valid_pitches)
                                    mpb_sym, xml_sym = sanitize_chord_figure(ch)

                                    if mpb_sym and (mpb_sym != last_mpb_sym or idx_m % 4 == 0):
                                        last_mpb_sym = mpb_sym
                                        first_notes = list(m.notes)
                                        if first_notes:
                                            first_notes[0].addLyric(mpb_sym)
                                            xml_chords_to_insert.append((m, mpb_sym, xml_sym))

                        score.insert(0, part)

                        # ---------------------------------------------------------
                        # ETAPA 1: Renderização em PDF via LilyPond (Sem Duration 0.0)
                        # ---------------------------------------------------------
                        pdf_bytes = None
                        page_images = []
                        try:
                            ly_base = OUTPUT_DIR / f"partitura_{session_id}"
                            generated_pdf = Path(score.write('lily.pdf', fp=str(ly_base)))
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
                            for f in OUTPUT_DIR.glob(f"partitura_{session_id}*"):
                                if f.suffix not in ['.mid', '.musicxml']:
                                    f.unlink(missing_ok=True)
                        except Exception as pdf_err:
                            st.warning(
                                f"O PDF não pôde ser gerado pelo LilyPond ({pdf_err}). "
                                "O processamento do MIDI e do MusicXML continuará normalmente."
                            )

                        # ---------------------------------------------------------
                        # ETAPA 2: Inserção de ChordSymbol com Log e Exportação MusicXML
                        # ---------------------------------------------------------
                        for m_obj, mpb_sym, xml_sym in xml_chords_to_insert:
                            try:
                                cs_obj = music21.harmony.ChordSymbol(xml_sym)
                                cs_obj.writeAsChord = False
                                m_obj.insert(0, cs_obj)
                            except Exception as cs_err:
                                print(
                                    f"[Auditoria ChordSymbol] Compasso {m_obj.number}: "
                                    f"mpb='{mpb_sym}', xml='{xml_sym}' falhou -> {cs_err}"
                                )

                        auditar_e_sanitizar_duracoes_musicxml(score)

                        xml_path = OUTPUT_DIR / f"partitura_{session_id}.musicxml"
                        try:
                            score.write('musicxml', fp=str(xml_path))
                        except Exception as xml_err:
                            print(f"[Erro MusicXML Principal] score.write falhou: {xml_err}")
                            part.write('musicxml', fp=str(xml_path))

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

                        st.session_state["result_ready"] = True
                        st.session_state["midi_bytes"] = midi_bytes
                        st.session_state["xml_bytes"] = xml_bytes
                        st.session_state["pdf_bytes"] = pdf_bytes
                        st.session_state["page_images"] = page_images
                        st.session_state["info_text"] = (
                            f"🎵 Andamento: {bpm} BPM | Compasso: {compasso_exibicao} | "
                            f"Tom: {tom_exibicao} | Clave de Sol + Cifras MPB"
                        )

            except Exception as e:
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

        col1, col2, col3 = st.columns(3)
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
                st.info("PDF indisponível nesta execução.")
        with col2:
            st.download_button(
                label="🎼 Baixar Arquivo MusicXML",
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

        st.markdown("---")
        if st.button("🔄 Enviar Outro Arquivo de Áudio"):
            st.session_state.clear()
            st.rerun()

        if st.session_state.get("page_images"):
            st.markdown("### 🎼 Visualização da Partitura com Cifras")
            for idx, img_bytes in enumerate(st.session_state["page_images"]):
                st.image(img_bytes, caption=f"Página {idx + 1}", use_column_width=True)
