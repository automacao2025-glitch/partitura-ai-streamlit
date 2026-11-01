import os
import gc
import uuid
from pathlib import Path
from collections import defaultdict
import numpy as np
import librosa
import soundfile as sf
import music21
import streamlit as st
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

# Configuração da página
st.set_page_config(page_title="Partitura.ai", page_icon="🎵", layout="centered")

UPLOAD_DIR = Path("temp_uploads")
OUTPUT_DIR = Path("temp_outputs")
for d in [UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

st.title("🎵 Partitura.ai - Motor Universal")
st.markdown("### Transcrição Musical Inteligente e Completa por IA")
st.write("Selecione ou arraste um ficheiro de áudio (MP3, WAV, M4A, FLAC, OGG) para gerar a partitura completa e o ficheiro MIDI.")

uploaded_file = st.file_uploader("Escolha um ficheiro de áudio", type=["mp3", "wav", "m4a", "flac", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    
    if st.button("✨ Gerar Partitura Completa", type="primary"):
        session_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(uploaded_file.name)[1].lower() or ".wav"
        safe_input_path = UPLOAD_DIR / f"raw_{session_id}{ext}"
        session_dir = UPLOAD_DIR / f"session_{session_id}"
        session_dir.mkdir(exist_ok=True)

        with st.spinner("⚡ A processar a música inteira com IA em blocos otimizados... Por favor, aguarde."):
            try:
                # Salva o ficheiro enviado
                with open(safe_input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 1. Carrega o áudio completo em memória de forma limpa
                y_full, sr = librosa.load(str(safe_input_path), sr=22050, mono=True)
                total_duration = len(y_full) / sr

                # Deteção global de BPM
                y_harmonic_full, _ = librosa.effects.hpss(y_full)
                onset_env = librosa.onset.onset_strength(y=y_harmonic_full, sr=sr)
                tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                bpm = float(np.ndim(tempo) and tempo[0] or tempo)
                if bpm <= 0: bpm = 120.0
                if bpm > 165: bpm /= 2.0
                elif bpm < 58: bpm *= 2.0
                time_sig = "2/4" if bpm >= 128 else "4/4"

                del y_harmonic_full, onset_env
                gc.collect()

                # Processamento em Chunks (blocos de 30 segundos) para suportar músicas longas sem travar
                chunk_duration = 30.0
                all_notes = []

                class MockNote:
                    def __init__(self, start, end, pitch, velocity):
                        self.start = start
                        self.end = end
                        self.pitch = pitch
                        self.velocity = velocity

                class MockInstrument:
                    def __init__(self, notes):
                        self.notes = notes

                class MockMidiData:
                    def __init__(self, instruments, notes):
                        self.instruments = instruments
                        self.notes = notes
                        
                    def write(self, path):
                        import pretty_midi
                        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
                        inst = pretty_midi.Instrument(program=0)
                        for n in self.notes:
                            inst.notes.append(pretty_midi.Note(velocity=int(n.velocity), pitch=int(n.pitch), start=float(n.start), end=float(n.end)))
                        pm.instruments.append(inst)
                        pm.write(path)

                start_time = 0.0
                chunk_idx = 0
                while start_time < total_duration:
                    end_time = min(start_time + chunk_duration, total_duration)
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    y_chunk = y_full[start_sample:end_sample]
                    if len(y_chunk) < sr * 2:
                        break

                    y_harm_chunk, _ = librosa.effects.hpss(y_chunk)
                    chunk_file = session_dir / f"chunk_{chunk_idx}.wav"
                    sf.write(str(chunk_file), y_harm_chunk, sr, subtype='PCM_16')

                    _, midi_data_chunk, _ = predict(str(chunk_file.resolve()), ICASSP_2022_MODEL_PATH)
                    
                    for inst in midi_data_chunk.instruments:
                        for n in inst.notes:
                            real_start = n.start + start_time
                            real_end = n.end + start_time
                            all_notes.append(MockNote(real_start, real_end, n.pitch, n.velocity))

                    try:
                        chunk_file.unlink()
                    except Exception:
                        pass
                    del y_chunk, y_harm_chunk, midi_data_chunk
                    gc.collect()

                    start_time += chunk_duration
                    chunk_idx += 1

                del y_full
                gc.collect()

                if not all_notes:
                    st.error("Nenhuma nota musical clara detetada neste áudio.")
                else:
                    sec_per_quarter = 60.0 / bpm
                    beats_per_bar = int(time_sig.split("/")[0])
                    grid_step = sec_per_quarter / 4.0

                    processed_notes = []
                    for n in all_notes:
                        if (n.end - n.start) >= 0.08 and n.velocity >= 38:
                            while n.pitch < 55: n.pitch += 12
                            while n.pitch > 88: n.pitch -= 12

                            n.start = round(n.start / grid_step) * grid_step
                            n.end = round(n.end / grid_step) * grid_step
                            if n.end <= n.start:
                                n.end = n.start + grid_step
                            processed_notes.append(n)

                    midi_filename = f"partitura_{session_id}.mid"
                    midi_path = OUTPUT_DIR / midi_filename
                    final_midi = MockMidiData([MockInstrument(processed_notes)], processed_notes)
                    final_midi.write(str(midi_path))

                    # Construção da partitura estruturada com music21
                    clusters_by_ql = defaultdict(list)
                    for n in processed_notes:
                        start_ql = round((n.start / sec_per_quarter) * 4) / 4.0
                        clusters_by_ql[start_ql].append(n)

                    sorted_offsets = sorted(clusters_by_ql.keys())
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
                    part.insert(0, music21.key.KeySignature(detected_key.sharps))

                    tonica_map = {'C': 'Dó', 'C#': 'Dó#', 'D': 'Ré', 'E': 'Mi', 'F': 'Fá', 'G': 'Sol', 'A': 'Lá', 'B': 'Si'}
                    tonic_pt = tonica_map.get(detected_key.tonic.name.replace('-', ''), detected_key.tonic.name)
                    mode_pt = "Maior" if detected_key.mode == "major" else "Menor"
                    
                    st.success("✨ Partitura da música inteira gerada com sucesso!")
                    st.info(f"🎵 Ritmo: {bpm} BPM | Compasso: {time_sig} | Tom: {tonic_pt} {mode_pt} | Clave de Sol")

                    # Botão para download do ficheiro MIDI
                    with open(midi_path, "rb") as mf:
                        st.download_button(
                            label="📥 Baixar Ficheiro MIDI Completo",
                            data=mf,
                            file_name="partitura_completa.mid",
                            mime="audio/midi"
                        )

            except Exception as e:
                st.error(f"Ocorreu um erro no processamento: {str(e)}")
            finally:
                if safe_input_path.exists():
                    try: safe_input_path.unlink()
                    except: pass
                import shutil
                if session_dir.exists():
                    try: shutil.rmtree(session_dir)
                    except: pass
                gc.collect()