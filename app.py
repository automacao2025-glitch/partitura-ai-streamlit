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
# CARREGAMENTO OTIMIZADO DO MODELO VIA ONNX RUNTIME
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
# MOTOR DE RECONHECIMENTO DE ACORDES / CIFRAS (ALTA ASSERTIVIDADE)
# ---------------------------------------------------------
NOTE_NAMES_PT = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

def build_chord_templates():
    """Gera perfis harmônicos (templates) para tríades e tétrades no padrão MPB."""
    templates = {}
    for root_pc in range(12):
        r_name = NOTE_NAMES_PT[root_pc]
        # Tríade Maior
        v = np.zeros(12); v[[root_pc, (root_pc+4)%12, (root_pc+7)%12]] = [1.0, 0.8, 0.9]
        templates[(f"{r_name}", f"{r_name}")] = v / np.linalg.norm(v)
        # Tríade Menor
        v = np.zeros(12); v[[root_pc, (root_pc+3)%12, (root_pc+7)%12]] = [1.0, 0.8, 0.9]
        templates[(f"{r_name}m", f"{r_name}m")] = v / np.linalg.norm(v)
        # Sétima da Dominante (7)
        v = np.zeros(12); v[[root_pc, (root_pc+4)%12, (root_pc+7)%12, (root_pc+10)%12]] = [1.0, 0.8, 0.7, 0.85]
        templates[(f"{r_name}7", f"{r_name}7")] = v / np.linalg.norm(v)
        # Sétima Maior (7M)
        v = np.zeros(12); v[[root_pc, (root_pc+4)%12, (root_pc+7)%12, (root_pc+11)%12]] = [1.0, 0.8, 0.7, 0.85]
        templates[(f"{r_name}7M", f"{r_name}maj7")] = v / np.linalg.norm(v)
        # Menor com Sétima (m7)
        v = np.zeros(12); v[[root_pc, (root_pc+3)%12, (root_pc+7)%12, (root_pc+10)%12]] = [1.0, 0.8, 0.7, 0.85]
        templates[(f"{r_name}m7", f"{r_name}m7")] = v / np.linalg.norm(v)
        # Meio-diminuto (m7(b5))
        v = np.zeros(12); v[[root_pc, (root_pc+3)%12, (root_pc+6)%12, (root_pc+10)%12]] = [1.0, 0.8, 0.8, 0.85]
        templates[(f"{r_name}m7(b5)", f"{r_name}m7b5")] = v / np.linalg.norm(v)
    return templates

CHORD_TEMPLATES = build_chord_templates()

def detect_bar_chord(chroma_slice: np.ndarray, midi_pitches: list, key_scale_pcs: set):
    """
    Combina a energia espectral do áudio (Chroma CQT) com as notas detectadas pela IA
    no compasso e o campo harmônico do tom para máxima assertividade na cifra.
    """
    vec = np.zeros(12, dtype=float)
    if chroma_slice is not None and chroma_slice.size > 0:
        vec += np.mean(chroma_slice, axis=1)

    for p in midi_pitches:
        vec[p % 12] += 0.6

    norm = np.linalg.norm(vec)
    if norm <= 1e-6:
        return None, None
    vec = vec / norm

    best_score = -1.0
    best_pair = ("C", "C")
    for (mpb_sym, xml_sym), tmpl in CHORD_TEMPLATES.items():
        score = float(np.dot(vec, tmpl))
        # Bônus suave para acordes diatônicos ao tom da música (evita cifras espúrias)
        active_pcs = np.where(tmpl > 0.1)[0]
        if all(pc in key_scale_pcs for pc in active_pcs):
            score *= 1.12
        if score > best_score:
            best_score = score
            best_pair = (mpb_sym, xml_sym)

    return best_pair


# ---------------------------------------------------------
# CABEÇALHO E INTERFACE WEB
# ---------------------------------------------------------
st.title("🎵 Partitura.ai")
st.markdown("### Sua Música em Partitura com Apenas um Clique")
st.write("Envie um arquivo de áudio para transcrever a melodia, extrair as cifras harmônicas, visualizar a partitura na tela e baixar o arquivo em **PDF**, **MusicXML** e **MIDI**.")

uploaded_file = st.file_uploader("Escolha um arquivo de áudio", type=["mp3", "wav", "m4a", "flac", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file.getvalue())
    
    if st.button("✨ Gerar Partitura Completa", type="primary"):
        session_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(uploaded_file.name)[1].lower() or ".wav"
        safe_input_path = UPLOAD_DIR / f"raw_{session_id}{ext}"
        session_dir = UPLOAD_DIR / f"session_{session_id}"
        session_dir.mkdir(exist_ok=True)

        with st.spinner("⚡ Processando o arquivo de áudio com IA e gerando a partitura com cifras... Por favor, aguarde."):
            try:
                ai_model = get_inference_model()

                with open(safe_input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 1. Carregamento rápido em 22050 Hz mono
                y_full, sr = librosa.load(str(safe_input_path), sr=22050, mono=True)
                total_duration = len(y_full) / sr

                # 2. Detecção ultra-rápida de BPM (analisa até os primeiros 45s sem HPSS pesado)
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

                # 3. Extração leve de Cromagrama para precisão harmônica das Cifras
                hop_length = 512
                chroma_full = librosa.feature.chroma_stft(y=y_full, sr=sr, hop_length=hop_length)

                # 4. Inferência IA em blocos de 60s direto no áudio (2x mais rápido e mais assertivo)
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
                    grid_step = sec_per_quarter / 4.0

                    harmony_pitches_by_bar = defaultdict(list)
                    processed_notes = []

                    for n in all_notes:
                        if (n.end - n.start) >= 0.08 and n.velocity >= 38:
                            q_start = round(n.start / grid_step) * grid_step
                            q_end = round(n.end / grid_step) * grid_step
                            if q_end <= q_start:
                                q_end = q_start + grid_step

                            start_ql_raw = round((q_start / sec_per_quarter) * 4) / 4.0
                            bar_idx = int(start_ql_raw // beats_per_bar)
                            harmony_pitches_by_bar[bar_idx].append(int(n.pitch))

                            melody_pitch = int(n.pitch)
                            while melody_pitch < 55: melody_pitch += 12
                            while melody_pitch > 88: melody_pitch -= 12

                            processed_notes.append(MockNote(q_start, q_end, melody_pitch, n.velocity))

                    # 5. Exportação do arquivo MIDI
                    midi_path = OUTPUT_DIR / f"partitura_{session_id}.mid"
                    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
                    inst = pretty_midi.Instrument(program=0)
                    for n in processed_notes:
                        inst.notes.append(pretty_midi.Note(velocity=int(n.velocity), pitch=int(n.pitch), start=float(n.start), end=float(n.end)))
                    pm.instruments.append(inst)
                    pm.write(str(midi_path))

                    # 6. Construção da Partitura Estruturada (music21)
                    clusters_by_ql = defaultdict(list)
                    for n in processed_notes:
                        start_ql = round((n.start / sec_per_quarter) * 4) / 4.0
                        clusters_by_ql[start_ql].append(n)

                    sorted_offsets = sorted(clusters_by_ql.keys())
                    score = music21.stream.Score()
                    score.insert(0, music21.metadata.Metadata())
                    score.metadata.title = os.path.splitext(uploaded_file.name)[0]
                    score.metadata.composer = "Transcrito por Partitura.ai"

                    part = music21.stream.Part()
                    part.insert(0, music21.clef.TrebleClef())
                    part.insert(0, music21.meter.TimeSignature(time_sig))
                    part.insert(0, music21.tempo.MetronomeMark(number=bpm))

                    for idx, start_ql in enumerate(sorted_offsets):
                        cluster = clusters_by_ql[start_ql]
                        raw_dur_ql = max((c.end - c.start) / sec_per_quarter for c in cluster)
                        dur_ql = round(raw_dur_ql * 4) / 4.0

                        if idx + 1 < len(sorted_offsets):
                            max_allowed = sorted_offsets[idx + 1] - start_ql
                            if dur_ql > max_allowed:
                                dur_ql = max_allowed

                        dur_ql = max(0.25, min(float(beats_per_bar), dur_ql))
                        pitches_in_cluster = sorted(list({int(c.pitch) for c in cluster}))[-2:]

                        if len(pitches_in_cluster) == 1:
                            el = music21.note.Note(pitches_in_cluster[0])
                        else:
                            el = music21.chord.Chord(pitches_in_cluster)

                        el.quarterLength = dur_ql
                        part.insert(start_ql, el)

                    ks = music21.analysis.discrete.KrumhanslSchmuckler()
                    detected_key = ks.getSolution(part)
                    key_scale_pcs = set(p.pitchClass for p in detected_key.getPitches())

                    part.insert(0, music21.key.KeySignature(detected_key.sharps))
                    part.makeMeasures(inPlace=True)

                    # ---------------------------------------------------------
                    # 7. INJEÇÃO SEGURA DE CIFRAS (SEPARADA PARA LILYPOND E MUSICXML)
                    # ---------------------------------------------------------
                    measures = list(part.getElementsByClass(music21.stream.Measure))
                    xml_chords_to_insert = []
                    last_mpb_sym = None
                    bar_duration_sec = beats_per_bar * sec_per_quarter

                    for idx_m, m in enumerate(measures):
                        bar_start_sec = idx_m * bar_duration_sec
                        bar_end_sec = (idx_m + 1) * bar_duration_sec
                        f_start = int((bar_start_sec * sr) / hop_length)
                        f_end = min(chroma_full.shape[1], max(f_start + 1, int((bar_end_sec * sr) / hop_length)))
                        chroma_slice = chroma_full[:, f_start:f_end] if f_start < chroma_full.shape[1] else None

                        bar_pitches = list(harmony_pitches_by_bar.get(idx_m, []))
                        for el in m.notes:
                            for p in el.pitches:
                                bar_pitches.append(p.midi)

                        mpb_sym, xml_sym = detect_bar_chord(chroma_slice, bar_pitches, key_scale_pcs)
                        if mpb_sym and (mpb_sym != last_mpb_sym or idx_m % 4 == 0):
                            last_mpb_sym = mpb_sym
                            first_notes = list(m.notes)
                            if first_notes:
                                # Ancora a cifra diretamente na 1ª nota do compasso (possui duração > 0, nunca quebra o LilyPond!)
                                first_notes[0].addLyric(mpb_sym)
                                xml_chords_to_insert.append((m, xml_sym))

                    del chroma_full
                    gc.collect()

                    score.insert(0, part)

                    # 8. PRIMEIRO: GERA O PDF VIA LILYPOND (Sem objetos de duração 0.0!)
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
                        st.warning(f"Nota na renderização visual do arquivo PDF: {pdf_err}")

                    # 9. SEGUNDO: INSERE OS OBJETOS ChordSymbol OFICIAIS E EXPORTA O MUSICXML
                    for m_obj, xml_sym in xml_chords_to_insert:
                        try:
                            m_obj.insert(0, music21.harmony.ChordSymbol(xml_sym))
                        except Exception:
                            pass

                    xml_path = OUTPUT_DIR / f"partitura_{session_id}.musicxml"
                    score.write('musicxml', fp=str(xml_path))

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
                    st.session_state["info_text"] = f"🎵 Andamento: {bpm} BPM | Compasso: {time_sig} | Tom: {tonic_pt} {mode_pt} | Clave de Sol + Cifras"

            except Exception as e:
                st.error(f"Erro durante o processamento do arquivo: {str(e)}")
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
                    file_name="partitura_clave_de_sol.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        with col2:
            st.download_button(
                label="🎼 Baixar Arquivo MusicXML",
                data=st.session_state["xml_bytes"],
                file_name="partitura_clave_de_sol.musicxml",
                mime="application/vnd.recordare.musicxml+xml",
                use_container_width=True
            )
        with col3:
            st.download_button(
                label="📥 Baixar Arquivo MIDI",
                data=st.session_state["midi_bytes"],
                file_name="partitura_completa.mid",
                mime="audio/midi",
                use_container_width=True
            )

        st.markdown("---")
        if st.button("🔄 Enviar Outro Arquivo de Áudio"):
            st.session_state.clear()
            st.rerun()

        if st.session_state.get("page_images"):
            st.markdown("### 🎼 Visualização da Partitura")
            for idx, img_bytes in enumerate(st.session_state["page_images"]):
                st.image(img_bytes, caption=f"Página {idx + 1}", use_column_width=True)
