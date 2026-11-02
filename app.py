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
import numpy as np
import librosa
import soundfile as sf
import pretty_midi
import music21
import pypdfium2 as pdfium
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH, FilenameSuffix, build_icassp_2022_model_path

st.set_page_config(page_title="Partitura.ai A3", page_icon="🎵", layout="centered")

st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span { display: none; }
    [data-testid="stFileUploaderDropzoneInstructions"] > div::before {
        content: "Arraste e solte o arquivo de áudio aqui";
        font-weight: 500; font-size: 1rem; color: #1e293b; display: block; margin-bottom: 4px;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > small { font-size: 0 !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] > div > small::after {
        content: "Limite de 200MB por arquivo • MP3, WAV, M4A, FLAC, OGG";
        font-size: 0.85rem; color: #64748b;
    }
    [data-testid="stFileUploaderDropzone"] button { font-size: 0 !important; }
    [data-testid="stFileUploaderDropzone"] button::after { content: "Selecionar Arquivo"; font-size: 0.875rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

UPLOAD_DIR = Path("temp_uploads")
OUTPUT_DIR = Path("temp_outputs")
for d in [UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 1. SISTEMA DE LOG TÉCNICO (AUDITORIA)
# ---------------------------------------------------------
class PartituraLogger:
    def __init__(self):
        self.lines = []
    
    def log(self, text):
        msg = str(text)
        self.lines.append(msg)
        print(msg)
        
    def get_log_text(self):
        return "\n".join(self.lines)

logger = PartituraLogger()


# ---------------------------------------------------------
# 2. CALIBRAÇÃO SEPARADA (BASE)
# ---------------------------------------------------------
CONFIG_SIGNAL = {
    "chunk_duration": 60.0,
    "chunk_overlap": 1.0,
    "dedup_tolerance_sec": 0.05,
    "max_sample_limit_sec": 45.0
}

CONFIG_METER = {
    "w_periodicity": 0.40,
    "w_accent_phase": 0.25,
    "w_subdivision": 0.15,
    "w_segment_consistency": 0.20,
    "base_margin_threshold": 0.08
}

CONFIG_MELODY = {
    "velocity_base": 45,
    "velocity_adapt_weight": 0.5,
    "duration_min_abs": 0.08,
    "jump_penalty_per_semitone": 1.5,
    "max_polyphony_overlap": 0.05
}

CONFIG_HARMONY = {
    "peso_base": 0.15,
    "peso_adapt_weight": 0.3,
    "beat_resolution": 2.0  # Analisa a cada 2 beats (meia medida típica)
}


# ---------------------------------------------------------
# 3. MODELOS E ESTRUTURAS
# ---------------------------------------------------------
@st.cache_resource
def get_inference_model():
    try:
        return Model(build_icassp_2022_model_path(FilenameSuffix.onnx))
    except Exception:
        return Model(ICASSP_2022_MODEL_PATH)

class MockNote:
    def __init__(self, start_sec, end_sec, pitch, velocity):
        self.start_sec = float(start_sec)
        self.end_sec = float(end_sec)
        self.pitch = int(pitch)
        self.velocity = int(velocity)
        self.duration = self.end_sec - self.start_sec


# ---------------------------------------------------------
# 4. MOTOR HARMÔNICO E TEMPLATES
# ---------------------------------------------------------
CHORD_TEMPLATES = {
    "maj":  ([1,0,0,0,1,0,0,1,0,0,0,0], ""),
    "min":  ([1,0,0,1,0,0,0,1,0,0,0,0], "m"),
    "dim":  ([1,0,0,1,0,0,1,0,0,0,0,0], "dim"),
    "aug":  ([1,0,0,0,1,0,0,0,1,0,0,0], "aug"),
    "7":    ([1,0,0,0,1,0,0,1,0,0,1,0], "7"),
    "maj7": ([1,0,0,0,1,0,0,1,0,0,0,1], "maj7"),
    "m7":   ([1,0,0,1,0,0,0,1,0,0,1,0], "m7"),
    "m7b5": ([1,0,0,1,0,0,1,0,0,0,1,0], "m7b5"),
    "sus4": ([1,0,0,0,0,1,0,1,0,0,0,0], "sus4"),
    "add9": ([1,0,1,0,1,0,0,1,0,0,0,0], "add9"),
    "9":    ([1,0,1,0,1,0,0,1,0,0,1,0], "9"),
}

def clean_enharmonics(tonic_name: str) -> str:
    if not tonic_name: return "C"
    replacements = {'Fb': 'E', 'Cb': 'B', 'E#': 'F', 'B#': 'C'}
    clean = tonic_name.replace('-', 'b')
    return replacements.get(clean, clean)

def sanitizar_nome_tom_pt(tonic_name: str) -> str:
    clean = clean_enharmonics(tonic_name)
    mapa_pt = {
        'C': 'Dó', 'C#': 'Dó#', 'Db': 'Ré♭', 'D': 'Ré', 'D#': 'Ré#', 'Eb': 'Mi♭',
        'E': 'Mi', 'F': 'Fá', 'F#': 'Fá#', 'Gb': 'Sol♭', 'G': 'Sol', 'G#': 'Sol#', 
        'Ab': 'Lá♭', 'A': 'Lá', 'A#': 'Lá#', 'Bb': 'Si♭', 'B': 'Si'
    }
    return mapa_pt.get(clean, clean)

def match_chord_template(pc_vector, root_pc):
    shifted_vector = np.roll(pc_vector, -root_pc)
    best_score = -1
    best_suffix = ""
    for name, (template, suffix) in CHORD_TEMPLATES.items():
        score = np.dot(shifted_vector, template) / (np.sum(template) + 1e-6)
        if score > best_score:
            best_score = score
            best_suffix = suffix
    return best_suffix, best_score

def analisar_harmonia_templates(notes, beat_times, adaptative_peso, bpm):
    logger.log("\n--- ANÁLISE HARMÔNICA (MOTOR TEMPLATES) ---")
    res = CONFIG_HARMONY["beat_resolution"]
    chords_by_measure = defaultdict(list)
    if len(beat_times) < 2: return chords_by_measure
    
    sec_per_beat = 60.0 / bpm
    window_sec = res * sec_per_beat
    
    max_time = beat_times[-1] + window_sec
    windows = np.arange(0, max_time, window_sec)
    
    last_chord = None
    for i in range(len(windows)-1):
        w_start, w_end = windows[i], windows[i+1]
        active_notes = [n for n in notes if n.start_sec < w_end and n.end_sec > w_start]
        
        if not active_notes: continue
        
        pc_vector = np.zeros(12)
        total_weight = 0
        for n in active_notes:
            overlap = min(n.end_sec, w_end) - max(n.start_sec, w_start)
            weight = overlap * (n.velocity / 127.0)
            pc_vector[n.pitch % 12] += weight
            total_weight += weight
            
        if total_weight < adaptative_peso:
            continue
            
        pc_vector = pc_vector / np.max(pc_vector)
        
        # Heurística de baixo
        bass_notes = sorted(active_notes, key=lambda x: x.pitch)
        bass_pc = bass_notes[0].pitch % 12 if bass_notes else np.argmax(pc_vector)
        
        # Testa todos os roots possíveis
        best_overall = -1
        best_root = 0
        best_suffix = ""
        
        for root_test in range(12):
            if pc_vector[root_test] < 0.3: continue
            suffix, score = match_chord_template(pc_vector, root_test)
            if root_test == bass_pc: score *= 1.2 # Bônus se root = bass
            if score > best_overall:
                best_overall = score
                best_root = root_test
                best_suffix = suffix
                
        if best_overall > 0.5:
            root_name = music21.pitch.Pitch(best_root).name
            root_name = clean_enharmonics(root_name).replace('b', '-')
            
            bass_name = music21.pitch.Pitch(bass_pc).name
            bass_name = clean_enharmonics(bass_name).replace('b', '-')
            
            xml_sym = f"{root_name}{best_suffix}"
            if bass_pc != best_root:
                xml_sym += f"/{bass_name}"
                
            if xml_sym != last_chord:
                # Estima a posição musical (aproximada para o dicionário por compasso)
                # bar_idx será recalculado estritamente depois, agrupamos por tempo absoluto
                chords_by_measure[w_start].append({
                    'xml_sym': xml_sym,
                    'score': best_overall
                })
                last_chord = xml_sym
                
    logger.log(f"Acordes distintos gerados: {len(chords_by_measure)}")
    return chords_by_measure


# ---------------------------------------------------------
# 5. RASTREAMENTO MELÓDICO (LITE) E GRADE DE BEATS
# ---------------------------------------------------------
def seconds_to_musical_position(time_sec, beat_times, time_sig):
    """Mapeia segundos reais para QuarterLength usando a grade de beats."""
    if len(beat_times) < 2: return time_sec
    
    # Interpola para encontrar o beat fracionário
    beat_idx = np.interp(time_sec, beat_times, np.arange(len(beat_times)))
    
    num, den = map(int, time_sig.split("/"))
    # Em 4/4, 1 beat = 1 QL. Em 6/8, 1 beat (colcheia pontuada) = 1.5 QL
    # Assumimos que o beat tracker do librosa segue o pulso natural.
    if den == 8 and num in [6, 9, 12]:
        ql_per_beat = 1.5
    else:
        ql_per_beat = 4.0 / den
        
    return beat_idx * ql_per_beat

def rastreamento_melodico_adaptativo(notes):
    logger.log("\n--- CALIBRAÇÃO ADAPTATIVA DA MELODIA ---")
    if not notes: return [], 0
    
    velocities = [n.velocity for n in notes]
    durations = [n.duration for n in notes]
    
    v_med = np.median(velocities)
    v_base = CONFIG_MELODY["velocity_base"]
    v_eff = v_base + (v_med - v_base) * CONFIG_MELODY["velocity_adapt_weight"]
    v_eff = max(30, min(v_eff, 90)) # Safety bounds
    
    d_min = CONFIG_MELODY["duration_min_abs"]
    
    logger.log(f"Velocity Base: {v_base} | Mediana: {v_med:.1f} | Efetiva: {v_eff:.1f}")
    logger.log(f"Duração Mínima Absoluta: {d_min}s")
    
    candidatas = [n for n in notes if n.velocity >= v_eff and n.duration >= d_min]
    candidatas.sort(key=lambda x: x.start_sec)
    
    # Viterbi-lite greedy
    melodia_final = []
    last_pitch = None
    
    for n in candidatas:
        if not melodia_final:
            melodia_final.append(n)
            last_pitch = n.pitch
            continue
            
        last_n = melodia_final[-1]
        overlap = max(0, min(n.end_sec, last_n.end_sec) - max(n.start_sec, last_n.start_sec))
        
        if overlap > CONFIG_MELODY["max_polyphony_overlap"]:
            # Escolhe a melhor nota baseada em velocity e menor salto
            jump_current = abs(n.pitch - last_pitch) if last_pitch else 0
            jump_prev = abs(last_n.pitch - last_pitch) if last_pitch else 0
            
            score_current = n.velocity - (jump_current * CONFIG_MELODY["jump_penalty_per_semitone"])
            score_prev = last_n.velocity - (jump_prev * CONFIG_MELODY["jump_penalty_per_semitone"])
            
            if score_current > score_prev:
                melodia_final[-1] = n
                last_pitch = n.pitch
        else:
            melodia_final.append(n)
            last_pitch = n.pitch
            
    logger.log(f"Notas brutas: {len(notes)} | Candidatas: {len(candidatas)} | Melodia Final: {len(melodia_final)}")
    return melodia_final, v_eff


# ---------------------------------------------------------
# 6. DETECTOR MÉTRICO A3
# ---------------------------------------------------------
def extract_tactus_candidates(tempo_array):
    base = float(tempo_array[0]) if tempo_array.size > 0 else 120.0
    if not math.isfinite(base) or base <= 0: base = 120.0
    
    cands = [base / 2, base, base * 2]
    # Filtra valores extremos
    valid_cands = [c for c in cands if 40 <= c <= 240]
    if not valid_cands: valid_cands = [120.0]
    return valid_cands

def detectar_compasso_a3(y, sr, bpm_global):
    logger.log("\n==============================")
    logger.log("METER DETECTION A3")
    logger.log("==============================")
    
    total_len = len(y)
    positions = [0.20, 0.50, 0.80]
    window_samples = int(18.0 * sr)
    
    scores_by_seg = []
    winner_by_seg = []
    
    for idx, pos in enumerate(positions):
        logger.log(f"\n--- SEGMENTO {int(pos*100)}% ---")
        start_s = max(0, int(pos * total_len) - window_samples // 2)
        end_s = min(total_len, start_s + window_samples)
        y_seg = y[start_s:end_s]
        
        if len(y_seg) < sr * 3:
            continue
            
        onset_env = librosa.onset.onset_strength(y=y_seg, sr=sr)
        _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, bpm=bpm_global)
        
        if len(beats) < 8:
            continue
            
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        ac = np.mean(tempogram, axis=1)
        
        frames_per_beat = max(1, int((60.0 / bpm_global * sr) / 512))
        
        def get_ac(mult):
            lag = int(round(frames_per_beat * mult))
            return float(ac[lag]) if lag < len(ac) else 0.0
            
        e2, e3, e4 = get_ac(2.0), get_ac(3.0), get_ac(4.0)
        e_tot = e2 + e3 + e4 + 1e-8
        
        # Pulsos
        p2 = (e2 * 0.7 + e4 * 0.3) / e_tot
        p3 = e3 / e_tot
        p4 = e4 / e_tot
        
        logger.log(f"Periodicidade pulsos: 2={p2:.2f}, 3={p3:.2f}, 4={p4:.2f}")
        
        # Subdivisão (Ternária vs Binária)
        lag_tern = int(round(frames_per_beat / 3.0))
        lag_bin = int(round(frames_per_beat / 2.0))
        e_tern = float(ac[lag_tern]) if lag_tern < len(ac) else 0.0
        e_bin = float(ac[lag_bin]) if lag_bin < len(ac) else 0.0
        sub_ratio = e_tern / (e_tern + e_bin + 1e-8)
        
        logger.log(f"Subdivisão: Bin={e_bin:.2f}, Tern={e_tern:.2f} (Ratio Tern={sub_ratio:.2f})")
        
        # Templates
        score_24 = p2
        score_34 = p3 * (1.0 - sub_ratio)
        score_44 = p4
        score_68 = p2 * sub_ratio * 1.5 # 2 pulsos, mas subdivisão ternária
        score_32 = p3 * (1.0 - sub_ratio) * 0.8
        
        s_dict = {"2/4": score_24, "3/4": score_34, "4/4": score_44, "6/8": score_68, "3/2": score_32}
        total_s = sum(s_dict.values()) + 1e-8
        s_dict = {k: v/total_s for k, v in s_dict.items()}
        
        win = max(s_dict.items(), key=lambda x: x[1])[0]
        scores_by_seg.append(s_dict)
        winner_by_seg.append(win)
        logger.log(f"Líder Seg: {win}")

    logger.log("\n--- GLOBAL ---")
    if not scores_by_seg:
        logger.log("Falha na análise dos segmentos. Fallback 4/4.")
        return "4/4", "ambiguous", "baixa"

    candidates = ["2/4", "3/4", "4/4", "6/8", "3/2"]
    global_scores = {c: np.mean([s[c] for s in scores_by_seg]) for c in candidates}
    
    majority = max(set(winner_by_seg), key=winner_by_seg.count)
    consistency = winner_by_seg.count(majority) / len(winner_by_seg)
    
    global_scores[majority] += CONFIG_METER["w_segment_consistency"] * consistency
    
    g_sum = sum(global_scores.values()) + 1e-8
    global_scores = {k: v/g_sum for k, v in global_scores.items()}
    
    ranking = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)
    best, best_val = ranking[0]
    sec, sec_val = ranking[1]
    margin = best_val - sec_val
    
    logger.log(f"Candidatos: {global_scores}")
    logger.log(f"Melhor: {best} ({best_val:.3f}) | Seg: {sec} ({sec_val:.3f})")
    logger.log(f"Margem: {margin:.3f} | Consistência: {consistency:.2f}")
    
    # Evidence Quality e Limiar Adaptativo
    base_thresh = CONFIG_METER["base_margin_threshold"]
    eff_thresh = base_thresh - (consistency - 0.5) * 0.05
    eff_thresh = max(0.02, min(eff_thresh, 0.15))
    
    logger.log(f"Limiar Base: {base_thresh:.3f} | Limiar Efetivo: {eff_thresh:.3f}")
    
    if margin >= eff_thresh and consistency >= 0.66:
        conf = "alta"
        status = "detected"
        res = best
    elif margin >= (eff_thresh * 0.5):
        conf = "média"
        status = "detected"
        res = best
    else:
        conf = "inconclusiva"
        status = "ambiguous"
        res = "4/4"
        logger.log(f"Motivo recusa: Margem ({margin:.3f}) menor que limiar efetivo ({eff_thresh:.3f})")
        
    logger.log(f"Resultado final: {res} (Status: {status})")
    return res, status, conf


# ---------------------------------------------------------
# INTERFACE PRINCIPAL
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Métrica (A3)")
    compasso_selecionado = st.selectbox(
        "Fórmula de Compasso:",
        ["Automático", "2/4", "3/4", "4/4", "6/8", "3/2"],
        index=0
    )

uploaded_file = st.file_uploader("Escolha um arquivo de áudio", type=["mp3", "wav", "m4a", "flac", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file.getvalue())
    
    if st.button("✨ Gerar Partitura A3", type="primary"):
        for k in ["result_ready", "midi_bytes", "xml_bytes", "pdf_bytes", "page_images", "info_text", "log_text"]:
            st.session_state.pop(k, None)

        session_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(uploaded_file.name)[1].lower() or ".wav"
        safe_input_path = UPLOAD_DIR / f"raw_{session_id}{ext}"
        session_dir = UPLOAD_DIR / f"session_{session_id}"
        session_dir.mkdir(exist_ok=True)

        logger.log("===================================")
        logger.log(f"PARTITURA.AI A3 — LOG TÉCNICO")
        logger.log(f"Data: {datetime.datetime.now().isoformat()}")
        logger.log(f"Arquivo: {uploaded_file.name}")
        logger.log(f"Session ID: {session_id}")
        logger.log("===================================")

        with st.spinner("⚡ Processamento A3 (Análise de Grade, Melodia e LilyPond)..."):
            try:
                ai_model = get_inference_model()
                with open(safe_input_path, "wb") as f: f.write(uploaded_file.getbuffer())

                y_full, sr = librosa.load(str(safe_input_path), sr=22050, mono=False)
                if y_full.ndim > 1:
                    logger.log("Áudio Estéreo detectado. Convertendo para MID (L+R)/2.")
                    y_full = np.mean(y_full, axis=0)
                else:
                    logger.log("Áudio Mono detectado.")
                    
                total_duration = len(y_full) / sr
                logger.log(f"Duração: {total_duration:.2f}s | Sample Rate: {sr}")

                onset_env = librosa.onset.onset_strength(y=y_full[:int(45.0*sr)], sr=sr)
                tempo_arr, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                
                cands = extract_tactus_candidates(np.asarray(tempo_arr).reshape(-1))
                bpm_global = round(cands[1] if len(cands)>1 else cands[0], 1)
                
                logger.log(f"Tactus Candidatos: {[round(c,1) for c in cands]}")
                logger.log(f"Tactus Selecionado: {bpm_global} BPM")

                beat_times = librosa.frames_to_time(beats, sr=sr)

                if compasso_selecionado != "Automático":
                    time_sig = compasso_selecionado
                    compasso_exibicao = f"{time_sig} (Manual)"
                    logger.log(f"Modo: Manual -> {time_sig}")
                else:
                    logger.log("Modo: Automático")
                    time_sig, m_status, m_conf = detectar_compasso_a3(y_full, sr, bpm_global)
                    if m_status == "detected":
                        compasso_exibicao = f"{time_sig} (Auto - Conf: {m_conf})"
                    else:
                        compasso_exibicao = f"Não determinado (Fallback {time_sig})"

                num, den = map(int, time_sig.split("/"))
                bar_length_ql = num * (4.0 / den)

                logger.log("\n--- INFERÊNCIA BASIC PITCH COM OVERLAP ---")
                chunk_duration = CONFIG_SIGNAL["chunk_duration"]
                overlap = CONFIG_SIGNAL["chunk_overlap"]
                step = chunk_duration - overlap
                
                raw_notes = []
                start_time = 0.0
                chunk_idx = 0
                
                while start_time < total_duration:
                    end_time = min(start_time + chunk_duration, total_duration)
                    if end_time - start_time < 1.5: break
                    
                    y_chunk = y_full[int(start_time*sr):int(end_time*sr)]
                    chunk_file = session_dir / f"chunk_{chunk_idx}.wav"
                    sf.write(str(chunk_file), y_chunk, sr, subtype='PCM_16')

                    _, midi_data_chunk, _ = predict(str(chunk_file.resolve()), ai_model)
                    
                    for inst in midi_data_chunk.instruments:
                        for n in inst.notes:
                            raw_notes.append(MockNote(n.start + start_time, n.end + start_time, n.pitch, n.velocity))

                    try: chunk_file.unlink()
                    except: pass
                    start_time += step
                    chunk_idx += 1

                logger.log(f"Notas brutas geradas: {len(raw_notes)}")
                
                # Deduplicação
                raw_notes.sort(key=lambda x: x.start_sec)
                dedup_notes = []
                tol = CONFIG_SIGNAL["dedup_tolerance_sec"]
                for n in raw_notes:
                    if not dedup_notes:
                        dedup_notes.append(n)
                        continue
                    last_n = dedup_notes[-1]
                    if n.pitch == last_n.pitch and abs(n.start_sec - last_n.start_sec) < tol:
                        last_n.end_sec = max(last_n.end_sec, n.end_sec)
                    else:
                        dedup_notes.append(n)
                
                logger.log(f"Notas após deduplicação de overlap: {len(dedup_notes)}")

                melodia_final, vel_eff = rastreamento_melodico_adaptativo(dedup_notes)
                
                if not melodia_final:
                    st.warning("Nenhuma nota sobreviveu à calibração adaptativa.")
                else:
                    midi_path = OUTPUT_DIR / f"partitura_{session_id}.mid"
                    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm_global)
                    inst = pretty_midi.Instrument(program=0)
                    for n in melodia_final:
                        inst.notes.append(pretty_midi.Note(
                            velocity=n.velocity, pitch=n.pitch, start=n.start_sec, end=n.end_sec
                        ))
                    pm.instruments.append(inst)
                    pm.write(str(midi_path))
                    logger.log("\nMIDI exportado com sucesso (tempo em segundos).")

                    # SCORE MUSIC21
                    score = music21.stream.Score()
                    score.metadata = music21.metadata.Metadata(title=os.path.splitext(uploaded_file.name)[0], composer="Partitura.ai A3")
                    part = music21.stream.Part()
                    part.insert(0, music21.clef.TrebleClef())
                    part.insert(0, music21.meter.TimeSignature(time_sig))
                    part.insert(0, music21.tempo.MetronomeMark(number=bpm_global))

                    logger.log("\n--- CONSTRUÇÃO DE GRADE MUSICAL E MUSIC21 ---")
                    for n in melodia_final:
                        start_ql = seconds_to_musical_position(n.start_sec, beat_times, time_sig)
                        end_ql = seconds_to_musical_position(n.end_sec, beat_times, time_sig)
                        dur_ql = end_ql - start_ql
                        
                        start_ql = round(start_ql * 4.0) / 4.0
                        dur_ql = round(dur_ql * 4.0) / 4.0
                        if dur_ql <= 0: dur_ql = 0.25
                        dur_ql = min(bar_length_ql, dur_ql)
                        
                        el = music21.note.Note(n.pitch)
                        el.quarterLength = dur_ql
                        part.insert(start_ql, el)

                    det_key, k_succ = detectar_centro_tonal_global(part)
                    part.insert(0, music21.key.KeySignature(det_key.sharps))
                    
                    part.makeMeasures(inPlace=True)
                    part.makeTies(inPlace=True)

                    # Análise Harmônica por Templates
                    w_pesos = [n.duration * (n.velocity/127.0) for n in dedup_notes]
                    peso_med = np.median(w_pesos) if w_pesos else 0
                    peso_eff = CONFIG_HARMONY["peso_base"] + (peso_med * CONFIG_HARMONY["peso_adapt_weight"])
                    logger.log(f"Peso Acorde Base: {CONFIG_HARMONY['peso_base']} | Efetivo: {peso_eff:.3f}")
                    
                    chords_by_time = analisar_harmonia_templates(dedup_notes, beat_times, peso_eff, bpm_global)
                    
                    # Ancoragem das cifras nas medidas correspondentes
                    measures = list(part.getElementsByClass(music21.stream.Measure))
                    for m in measures:
                        m_start_sec = librosa.frames_to_time(int(m.offset / (bpm_global/60.0) * (sr/512)), sr=sr)
                        # Busca o acorde temporal mais próximo para esta medida
                        closest_time = min(chords_by_time.keys(), key=lambda k: abs(k - m_start_sec)) if chords_by_time else None
                        if closest_time and abs(closest_time - m_start_sec) < 2.0:
                            best_chord = max(chords_by_time[closest_time], key=lambda x: x['score'])
                            xml_sym = best_chord['xml_sym']
                            mpb_sym = clean_enharmonics(xml_sym).replace('-', 'b')
                            try:
                                cs = music21.harmony.ChordSymbol(xml_sym)
                                cs.writeAsChord = False
                                m.insert(0, cs)
                                if m.notes: m.notes[0].addLyric(mpb_sym)
                            except Exception as e:
                                logger.log(f"Erro ChordSymbol '{xml_sym}': {e}")
                                
                    score.insert(0, part)
                    
                    pdf_bytes, page_images = None, []
                    try:
                        ly_base = OUTPUT_DIR / f"partitura_{session_id}"
                        gen_pdf = Path(score.write('lily.pdf', fp=str(ly_base)))
                        if gen_pdf.exists():
                            pdf_bytes = gen_pdf.read_bytes()
                            pdf_doc = pdfium.PdfDocument(pdf_bytes)
                            for i in range(len(pdf_doc)):
                                page_images.append(pdf_doc[i].render(scale=2).to_pil())
                    except Exception as e:
                        logger.log(f"PDF LilyPond falhou: {e}")
                        st.warning(f"PDF indisponível pontualmente: {e}")

                    xml_path = OUTPUT_DIR / f"partitura_{session_id}.musicxml"
                    try:
                        score.write('musicxml', fp=str(xml_path))
                        logger.log("MusicXML exportado com sucesso.")
                    except Exception as e:
                        logger.log(f"[ERRO CRÍTICO] Falha MusicXML: {e}")
                        st.error(f"Erro ao exportar MusicXML: {e}")

                    tom_str = sanitizar_nome_tom_pt(det_key.tonic.name) + (" Maior" if det_key.mode == "major" else " Menor") if k_succ else "Não determinado"

                    st.session_state["result_ready"] = True
                    st.session_state["midi_bytes"] = midi_path.read_bytes() if midi_path.exists() else None
                    st.session_state["xml_bytes"] = xml_path.read_bytes() if xml_path.exists() else None
                    st.session_state["pdf_bytes"] = pdf_bytes
                    st.session_state["page_images"] = [io.BytesIO().getvalue() for img in page_images] # needs proper PIL to bytes mapping normally, simplifying for space
                    
                    # Proper PIL to bytes mapping
                    encoded_images = []
                    for img in page_images:
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        encoded_images.append(buf.getvalue())
                    st.session_state["page_images"] = encoded_images

                    st.session_state["info_text"] = f"🎵 Andamento: {bpm_global} BPM | Compasso: {compasso_exibicao} | Tom: {tom_str}"
                    
                    logger.log("\n===================================")
                    logger.log("PARTITURA.AI — RESUMO SUCESSO")
                    logger.log("===================================")
                    st.session_state["log_text"] = logger.get_log_text()

            except Exception as e:
                logger.log(f"[ERRO GLOBAL FATAL]: {e}")
                st.session_state["log_text"] = logger.get_log_text()
                st.error(f"Falha: {e}")
            finally:
                for p in [safe_input_path, session_dir] + list(OUTPUT_DIR.glob(f"*{session_id}*")):
                    if p.is_file(): p.unlink(missing_ok=True)
                    elif p.is_dir(): shutil.rmtree(p, ignore_errors=True)
                gc.collect()

    if st.session_state.get("result_ready"):
        st.success("✨ Partitura gerada com sucesso!")
        st.info(st.session_state["info_text"])

        c1, c2, c3, c4 = st.columns(4)
        if st.session_state.get("pdf_bytes"):
            c1.download_button("📄 Baixar PDF", st.session_state["pdf_bytes"], "partitura.pdf", "application/pdf", use_container_width=True)
        if st.session_state.get("xml_bytes"):
            c2.download_button("🎼 Baixar MusicXML", st.session_state["xml_bytes"], "partitura.musicxml", "application/vnd.recordare.musicxml+xml", use_container_width=True)
        if st.session_state.get("midi_bytes"):
            c3.download_button("📥 Baixar MIDI", st.session_state["midi_bytes"], "partitura.mid", "audio/midi", use_container_width=True)
        if st.session_state.get("log_text"):
            c4.download_button("📋 Baixar Log Técnico", st.session_state["log_text"], "partitura_ai_log.txt", "text/plain", use_container_width=True)

        st.markdown("---")
        if st.session_state.get("page_images"):
            st.markdown("### 🎼 Visualização")
            for idx, img in enumerate(st.session_state["page_images"]):
                st.image(img, caption=f"Página {idx + 1}", use_column_width=True)
