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
# 1. PAINEL CENTRAL DE CALIBRAÇÃO (CONFIG_CALIBRACAO)
# ---------------------------------------------------------
CONFIG_CALIBRACAO = {
    "CORTE_PITCH_MELODIA": 52,             # Limite refinado para separar baixos/harmonia da melodia visual
    "MAX_NOTAS_CLUSTER_VISUAL": 2,         # Mantém a pauta limpa na Clave de Sol (Lead Sheet)
    "VELOCITY_MIN_MELODIA": 42,            # Filtro rigoroso contra ruídos e ressonâncias indesejadas
    "VELOCITY_MIN_HARMONIA": 35,           # Sensibilidade ideal para capturar o campo harmônico polifônico
    "DURACAO_MIN_NOTA_SEG": 0.09,          # Descarta micro-estalos e transientes de percussão
    "SUBDIVISAO_MINIMA_QL": 0.25,          # Trava na semicolcheia (filtro elástico anti-rubato)
    "PESO_MINIMO_ACORDE": 0.22,            # Exige peso real (duração x velocity) para consolidar a cifra do acorde
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
# 2. FUNÇÕES MODULARES ISOLADAS E CONVERSORES SEGUROS
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

            # Garante precisão binária limpa (múltiplos exatos de 0.25)
            q_start_ql = round(q_start_ql * 4.0) / 4.0
            q_end_ql = round(q_end_ql * 4.0) / 4.0

            bar_idx = int(q_start_ql // bar_length_ql)

            # Fluxo Harmônico Polifônico Real (mantém oitava original e graves)
            if n.velocity >= config["VELOCITY_MIN_HARMONIA"]:
                weight = dur_sec * (n.velocity / 127.0)
                harmonic_stream_by_bar[bar_idx].append({
                    'pitch': int(n.pitch),
                    'weight': weight
                })

            # Fluxo Melódico Visual (Clave de Sol limpa)
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
    Varre o Score/Part antes da exportação MusicXML, registando e corrigindo
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
# CABEÇALHO E INTERFACE WEB
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
        help="No modo Automático, o sistema estima o compasso. Se conhecer a métrica exata da obra, selecione manualmente."
    )

uploaded_file = st.file_uploader("Escolha um arquivo de áudio", type=["mp3", "wav", "m4a", "flac", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file.getvalue())
    
    if st.button("✨ Gerar Partitura Completa com Cifras", type="primary"):
        # PONTO 4: Limpeza preventiva do session_state antes de iniciar nova transcrição
        for k in ["result_ready", "midi_bytes", "xml_bytes", "pdf_bytes", "page_images", "info_text"]:
            if k in st.session_state:
                del st.session_state[k]

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

                # PONTO 9: Detecção e normalização robusta de BPM
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

                # Definição de compasso (Manual vs Automático com indicação transparente)
                if compasso_selecionado != "Automático":
                    time_sig = compasso_selecionado
                    compasso_exibicao = f"{time_sig} (Manual)"
                else:
                    time_sig = "2/4" if bpm >= 128.0 else "4/4"
                    compasso_exibicao = f"{time_sig} (Automático)"

                del onset_env
                gc.collect()

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
                    # Executa a separação Dual-Stream com unidades desacopladas
                    melody_clusters, harmonic_stream_by_bar, bar_length_ql = quantizar_e_filtrar_fluxos(
                        all_notes, bpm, time_sig, CONFIG_CALIBRACAO
                    )

                    # PONTO 7: Verifica se melody_clusters ficou vazio após o filtro visual
                    if not melody_clusters:
                        st.warning(
                            "Foram detectadas notas no áudio, mas nenhuma atingiu os limiares do filtro melódico principal "
                            "(altura ou intensidade mínimas). Tente enviar uma gravação com a melodia mais destacada."
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

                            # PONTO 2: Comparação de max_allowed_ql 100% em quarterLength
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
                            # PONTO 10: Mensagem precisa sobre a continuidade do fluxo
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
                                # PONTO 6: Registo detalhado de falha de ChordSymbol
                                print(
                                    f"[Auditoria ChordSymbol] Compasso {m_obj.number}: "
                                    f"mpb='{mpb_sym}', xml='{xml_sym}' falhou -> {cs_err}"
                                )

                        auditar_e_sanitizar_duracoes_musicxml(score)

                        xml_path = OUTPUT_DIR / f"partitura_{session_id}.musicxml"
                        try:
                            score.write('musicxml', fp=str(xml_path))
                        except Exception as xml_err:
                            # PONTO 11: Regista o erro original antes de tentar o fallback
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
                # PONTO 5: Limpeza garantida de todos os ficheiros temporários da sessão
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
