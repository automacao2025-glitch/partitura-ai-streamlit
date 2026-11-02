import sys
import os
import io
import gc
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
# 1. PAINEL CENTRAL DE CALIBRAÇÃO (CONFIG_CALIBRACAO)
# ---------------------------------------------------------
CONFIG_CALIBRACAO = {
    "CORTE_PITCH_MELODIA": 52,             
    "MAX_NOTAS_CLUSTER_VISUAL": 2,         
    "VELOCITY_MIN_MELODIA": 42,            
    "VELOCITY_MIN_HARMONIA": 35,           
    "DURACAO_MIN_NOTA_SEG": 0.09,          
    "SUBDIVISAO_MINIMA_QL": 0.25,          
    "PESO_MINIMO_ACORDE": 0.22,            
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
    def __init__(self, start, end, pitch, velocity):
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity

# ---------------------------------------------------------
# 2. FUNÇÕES MODULARES ISOLADAS
# ---------------------------------------------------------
def format_root_pt_br(pitch_name: str) -> str:
    if not pitch_name:
        return "C"
    if len(pitch_name) > 1 and pitch_name[1] == '-':
        return pitch_name[0] + 'b' + pitch_name[2:]
    return pitch_name

def sanitize_chord_figure(ch: music21.chord.Chord, detected_key=None) -> tuple:
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
    sec_per_quarter = 60.0 / bpm
    beats_per_bar = int(time_sig.split("/")[0])
    grid_step = config["SUBDIVISAO_MINIMA_QL"] * sec_per_quarter

    melody_clusters = defaultdict(list)
    harmonic_stream_by_bar = defaultdict(list)

    for n in all_notes:
        dur = n.end - n.start
        if dur >= config["DURACAO_MIN_NOTA_SEG"]:
            q_start = round(n.start / grid_step) * grid_step
            q_end = round(n.end / grid_step) * grid_step
            if q_end <= q_start:
                q_end = q_start + grid_step

            start_ql_raw = round((q_start / sec_per_quarter) * 4) / 4.0
            bar_idx = int(start_ql_raw // beats_per_bar)

            if n.velocity >= config["VELOCITY_MIN_HARMONIA"]:
                weight = dur * (n.velocity / 127.0)
                harmonic_stream_by_bar[bar_idx].append({
                    'pitch': int(n.pitch),
                    'weight': weight
                })

            if n.velocity >= config["VELOCITY_MIN_MELODIA"] and n.pitch >= config["CORTE_PITCH_MELODIA"]:
                melody_pitch = int(n.pitch)
                while melody_pitch < 55: melody_pitch += 12
                while melody_pitch > 88: melody_pitch -= 12

                melody_clusters[q_start].append(MockNote(q_start, q_end, melody_pitch, n.velocity))

    return melody_clusters, harmonic_stream_by_bar

def detectar_centro_tonal_global(part):
    ks = music21.analysis.discrete.KrumhanslSchmuckler()
    return ks.getSolution(part)


# ---------------------------------------------------------
# CABEÇALHO E INTERFACE WEB
# ---------------------------------------------------------
st.title("🎵 Partitura.ai")
st.markdown("### Sua Música em Partitura com Melodia e Cifras Harmônicas")
st.write("Envie um arquivo de áudio para transcrever a melodia, extrair o campo harmônico avançado (cifras e acordes base), visualizar a partitura e exportar em **PDF**, **MusicXML** e **MIDI**.")

uploaded_file = st.file_uploader("Escolha um arquivo de áudio", type=["mp3", "wav", "m4a", "flac", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file.getvalue())
    
    if st.button("✨ Gerar Partitura Completa com Cifras", type="primary"):
        session_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(uploaded_file.name)[1].lower() or ".wav"
        safe_input_path = UPLOAD_DIR / f"raw_{session_id}{ext}"
        session_dir = UPLOAD_DIR / f"session_{session_id}"
        session_dir.mkdir(exist_ok=True)

        with st.spinner("⚡ Processando áudio no Render (compilando LilyPond e IA)... Por favor, aguarde alguns instantes."):
            try:
                ai_model = get_inference_model()

                with open(safe_input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                y_full, sr = librosa.load(str(safe_input_path), sr=22050, mono=True)
                total_duration = len(y_full) / sr

                sample_limit = min(len(y_full), int(45.0 * sr))
                onset_env = librosa.onset.onset_strength(y=y_full[:sample_limit], sr=sr)
                tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                bpm = float(np.ndim(tempo) and tempo[0] or tempo)
                if bpm <= 0: bpm = 120.0
                if bpm > 165: bpm /= 2.0
                elif bpm < 58: bpm *= 2.0
                bpm = round(bpm, 1)
                time_sig = "2/4" if bpm >= 128 else "4/4"

                del onset_env
                gc.collect()

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
                            all_notes.append(MockNote(n.start + start_time, n.end + start_time, n.pitch, n.velocity))

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
                    sec_per_quarter = 60.0 / bpm
                    beats_per_bar = int(time_sig.split("/")[0])

                    melody_clusters, harmonic_stream_by_bar = quantizar_e_filtrar_fluxos(
                        all_notes, bpm, time_sig, CONFIG_CALIBRACAO
                    )

                    processed_notes = []
                    for q_start, cluster in melody_clusters.items():
                        for note_obj in cluster:
                            processed_notes.append(note_obj)

                    midi_path = OUTPUT_DIR / f"partitura_{session_id}.mid"
                    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
                    inst = pretty_midi.Instrument(program=0)
                    for n in processed_notes:
                        inst.notes.append(pretty_midi.Note(velocity=int(n.velocity), pitch=int(n.pitch), start=float(n.start), end=float(n.end)))
                    pm.instruments.append(inst)
                    pm.write(str(midi_path))

                    score = music21.stream.Score()
                    score.insert(0, music21.metadata.Metadata())
                    score.metadata.title = os.path.splitext(uploaded_file.name)[0]
                    score.metadata.composer = "Transcrito por Partitura.ai"

                    part = music21.stream.Part()
                    part.insert(0, music21.clef.TrebleClef())
                    part.insert(0, music21.meter.TimeSignature(time_sig))
                    part.insert(0, music21.tempo.MetronomeMark(number=bpm))

                    sorted_offsets = sorted(melody_clusters.keys())
                    for idx, start_ql in enumerate(sorted_offsets):
                        cluster = melody_clusters[start_ql]
                        raw_dur_ql = max((c.end - c.start) / sec_per_quarter for c in cluster)
                        
                        # SANITIZAÇÃO DE DURAÇÃO: Arredonda rigorosamente para múltiplos válidos (0.25, 0.5, 1.0, etc.)
                        dur_ql = round(raw_dur_ql * 4) / 4.0
                        if dur_ql <= 0:
                            dur_ql = 0.25

                        if idx + 1 < len(sorted_offsets):
                            max_allowed = sorted_offsets[idx + 1] - start_ql
                            if dur_ql > max_allowed:
                                dur_ql = max(0.25, round(max_allowed * 4) / 4.0)

                        dur_ql = max(0.25, min(float(beats_per_bar), dur_ql))
                        pitches_in_cluster = sorted(list({int(c.pitch) for c in cluster}))[-CONFIG_CALIBRACAO["MAX_NOTAS_CLUSTER_VISUAL"]:]

                        if len(pitches_in_cluster) == 1:
                            el = music21.note.Note(pitches_in_cluster[0])
                        else:
                            el = music21.chord.Chord(pitches_in_cluster)

                        el.quarterLength = dur_ql
                        part.insert(start_ql, el)

                    detected_key = detectar_centro_tonal_global(part)
                    part.insert(0, music21.key.KeySignature(detected_key.sharps))
                    
                    # BLINDAGEM DE COMPASSOS: Garante que o music21 não gere durações inexpressíveis
                    try:
                        part.makeMeasures(inPlace=True, strictPadding=False)
                    except Exception:
                        part.makeMeasures(inPlace=True)

                    measures = list(part.getElementsByClass(music21.stream.Measure))
                    xml_chords_to_insert = []
                    last_mpb_sym = None

                    for idx_m, m in enumerate(measures):
                        bar_notes_data = harmonic_stream_by_bar.get(idx_m, [])
                        if bar_notes_data:
                            valid_pitches = [item['pitch'] for item in bar_notes_data if item['weight'] >= CONFIG_CALIBRACAO["PESO_MINIMO_ACORDE"]]
                            if not valid_pitches:
                                valid_pitches = [item['pitch'] for item in bar_notes_data]

                            if valid_pitches:
                                ch = music21.chord.Chord(valid_pitches)
                                mpb_sym, xml_sym = sanitize_chord_figure(ch, detected_key)

                                if mpb_sym and (mpb_sym != last_mpb_sym or idx_m % 4 == 0):
                                    last_mpb_sym = mpb_sym
                                    first_notes = list(m.notes)
                                    if first_notes:
                                        first_notes[0].addLyric(mpb_sym)
                                        xml_chords_to_insert.append((m, xml_sym))

                    score.insert(0, part)

                    # ETAPA 1: Renderização Gráfica em PDF via LilyPond com Fallback Seguro
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
                        st.warning(f"O motor LilyPond no Render encontrou restrição pontual: {pdf_err}. Os ficheiros MIDI e MusicXML foram gerados com sucesso.")

                    # ETAPA 2: Inserção tardia de ChordSymbol e Exportação MusicXML Blindada
                    for m_obj, xml_sym in xml_chords_to_insert:
                        try:
                            cs_obj = music21.harmony.ChordSymbol(xml_sym)
                            cs_obj.writeAsChord = False
                            m_obj.insert(0, cs_obj)
                        except Exception:
                            pass

                    xml_path = OUTPUT_DIR / f"partitura_{session_id}.musicxml"
                    try:
                        score.write('musicxml', fp=str(xml_path))
                    except Exception:
                        # Fallback seguro caso ocorra incompatibilidade extrema de oitava no XML
                        part.write('musicxml', fp=str(xml_path))

                    tonica_map = {'C': 'Dó', 'C#': 'Dó#', 'D': 'Ré', 'E': 'Mi', 'F': 'Fá', 'G': 'Sol', 'A': 'Lá', 'B': 'Si'}
                    tonic_pt = tonica_map.get(detected_key.tonic.name.replace('-', ''), detected_key.tonic.name)
                    mode_pt = "Maior" if detected_key.mode == "major" else "Menor"

                    midi_bytes = midi_path.read_bytes()
                    xml_bytes = xml_path.read_bytes()

                    midi_path.unlink(missing_ok=True)
                    xml_path.unlink(missing_ok=True)

                    st.session_state["result_ready"] = True
                    st.session_state["midi_bytes"] = midi_bytes
                    st.session_state["xml_bytes"] = xml_bytes
                    st.session_state["pdf_bytes"] = pdf_bytes
                    st.session_state["page_images"] = page_images
                    st.session_state["info_text"] = f"🎵 Andamento: {bpm} BPM | Compasso: {time_sig} | Tom: {tonic_pt} {mode_pt} | Clave de Sol + Cifras MPB"

            except Exception as e:
                st.error(f"Erro crítico durante o processamento do arquivo: {str(e)}")
            finally:
                if safe_input_path.exists():
                    try: safe_input_path.unlink()
                    except Exception: pass
                if session_dir.exists():
                    try: shutil.rmtree(session_dir)
                    except Exception: pass
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
                st.info("PDF em processamento gráfico.")
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
