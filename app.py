import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
import os
import tempfile

# Direktori dasar proyek
BASE_DIR = os.path.dirname(__file__)

# Path Model
MODEL_PATH = 'model/model_genre_musik.keras'

# Parameter untuk preprocessing
SAMPLE_RATE = 22050
DURASI_TRACK = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURASI_TRACK
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS_TRAINING = 6

# Parameter untuk jumlah frame Mel-spectrogram per segmen
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS_TRAINING)
NUM_FRAMES_PER_SEGMENT_TRAIN = math.ceil(SAMPLES_PER_SEGMENT / HOP_LENGTH)

# Genre mapping sesuai dataset
GENRE_MAPPING = [
    "blues",
    "classical",
    "country",
    "disco",
    "edm",
    "hiphop",
    "jazz",
    "lofi",
    "metal",
    "pop",
    "reggae",
    "rock"
]


def extract_mel_spec_from_segment(signal, sample_rate, start_sample, end_sample, n_mels, n_fft, hop_length):
    """Mengekstrak Mel-spectrogram dari segmen sinyal audio yang ditentukan."""
    segment_signal = signal[start_sample:end_sample]
    mel_spec = librosa.feature.melspectrogram(y=segment_signal, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

@st.cache_resource
def load_ml_model(model_path):
    """Memuat model Keras yang sudah dilatih."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}. Pastikan path model benar dan file model tidak rusak.")
        return None

def preprocess_and_predict_genre_streamlit(model, audio_file_path, mapping_genres,
                                             sample_rate, n_mels, n_fft, hop_length,
                                             samples_per_segment_train, num_frames_per_segment_train):
    """Memproses file audio, mengekstrak Mel-spectrogram, dan memprediksi genre untuk aplikasi Streamlit."""

    if model is None:
        return None, None

    try:
        signal, sr = librosa.load(audio_file_path, sr=sample_rate)
    except Exception as e:
        st.error(f"Error memuat file audio: {e}. Pastikan format audio valid (mis. WAV, MP3, OGG).")
        return None, None

    total_samples = len(signal)
    num_segments_in_file = math.ceil(total_samples / samples_per_segment_train)

    segment_predictions_probs = []

    for i in range(num_segments_in_file):
        start_sample = i * samples_per_segment_train
        end_sample = start_sample + samples_per_segment_train

        segment_mel_spec_extracted = extract_mel_spec_from_segment(signal, sr, start_sample, end_sample, n_mels, n_fft, hop_length)

        if segment_mel_spec_extracted.shape[1] < num_frames_per_segment_train:
            padding_needed = num_frames_per_segment_train - segment_mel_spec_extracted.shape[1]
            segment_mel_spec_extracted = np.pad(segment_mel_spec_extracted, ((0, 0), (0, padding_needed)), mode='constant')
        elif segment_mel_spec_extracted.shape[1] > num_frames_per_segment_train:
            segment_mel_spec_extracted = segment_mel_spec_extracted[:, :num_frames_per_segment_train]

        if segment_mel_spec_extracted.shape[1] != num_frames_per_segment_train:
             if segment_mel_spec_extracted.shape[1] < num_frames_per_segment_train:
                 padding_needed = num_frames_per_segment_train - segment_mel_spec_extracted.shape[1]
                 segment_mel_spec_extracted = np.pad(segment_mel_spec_extracted, ((0, 0), (0, padding_needed)), mode='constant')
             elif segment_mel_spec_extracted.shape[1] > num_frames_per_segment_train:
                 segment_mel_spec_extracted = segment_mel_spec_extracted[:, :num_frames_per_segment_train]

        mel_spec_input = segment_mel_spec_extracted[np.newaxis, ..., np.newaxis]

        prediction_probs = model.predict(mel_spec_input, verbose=0)
        segment_predictions_probs.append(prediction_probs[0])

    final_prediction_probs = np.sum(segment_predictions_probs, axis=0)

    if np.sum(final_prediction_probs) > 0:
        final_prediction_probs = final_prediction_probs / np.sum(final_prediction_probs)
    else:
        final_prediction_probs = np.zeros_like(final_prediction_probs)

    predicted_index = np.argmax(final_prediction_probs)
    predicted_genre = mapping_genres[predicted_index]

    return predicted_genre, final_prediction_probs

# --- STREAMLIT APP ---
st.set_page_config(page_title="Deteksi Genre Musik", layout="centered")

st.title("🎶 Klasifikasi Genre Musik")

st.markdown("""
**Peneliti:** Muhammad Zharfan Alfanso (51421100)
<br><br>
Website ini adalah sistem implementasi menggunakan Streamlit untuk melakukan klasifikasi genre musik pada file audio menggunakan model Keras yang dibuat menggunakan Convolutional Neural Network (CNN) dan Recurrent Neural Network (RNN) yang memanfaatkan fitur Mel-spectrogram.
<br>
""", unsafe_allow_html=True)

# Hapus panggilan load_genre_mapping dan sesuaikan logika
with st.spinner("Memuat model yang diperlukan..."):
    model = load_ml_model(MODEL_PATH)
    genre_mapping = GENRE_MAPPING

    if model is not None:
        st.markdown("---")
        st.subheader("Daftar Genre Musik yang Dapat Diprediksi:")
        genre_emojis = {
            'blues': '🎷',
            'classical': '🎻',
            'country': '🤠',
            'disco': '🕺',
            'edm': '🎧',
            'hiphop': '🎤',
            'jazz': '🎺',
            'lofi': '☕',
            'metal': '🤘',
            'pop': '✨',
            'reggae': '🌴',
            'rock': '🎸'
        }
        cols = st.columns(4)
        genres = sorted(genre_mapping)
        for i, genre in enumerate(genres):
            emoji = genre_emojis.get(genre.lower(), '🎵')
            with cols[i % 4]:
                st.markdown(f"#### {emoji} {genre.capitalize()}")
    else:
        st.error("Gagal memuat model. Pastikan path model benar dan file ada.")
        st.stop()

st.markdown("---")
st.subheader("Cara Menggunakan:")
st.markdown("""
1. Unggah file audio Anda dalam format **.wav**, **.mp3**, atau **.ogg** di bawah.
2. Tunggu beberapa saat hingga proses analisis audio selesai.
3. Hasil prediksi genre akan ditampilkan di bawah, dilengkapi dengan persentase keyakinan model.
""")

st.markdown("---")

uploaded_file = st.file_uploader("Pilih file audio Anda", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.subheader("File Audio Anda:")
        st.audio(tmp_file_path, format=uploaded_file.type)

        st.subheader("Hasil Prediksi:")
        with st.spinner("Menganalisis audio, mohon tunggu..."):
            predicted_genre_result, all_probs = preprocess_and_predict_genre_streamlit(
                model,
                tmp_file_path,
                genre_mapping, 
                SAMPLE_RATE,
                N_MELS,
                N_FFT,
                HOP_LENGTH,
                SAMPLES_PER_SEGMENT,
                NUM_FRAMES_PER_SEGMENT_TRAIN
            )

        if predicted_genre_result is not None:
            confidence_threshold = 0.50
            max_prob = np.max(all_probs)

            if max_prob < confidence_threshold:
                st.write(f"**Genre yang Diprediksi:** **<span style='font-size: 24px; color: #FFA500;'>Genre tidak dapat dipastikan</span>**", unsafe_allow_html=True)
                st.info(f"Model kurang yakin dengan prediksinya (keyakinan tertinggi: {max_prob*100:.2f}%). Input mungkin di luar genre yang dilatih atau bukan musik.")
            else:
                st.write(f"**Genre yang Diprediksi (Paling Mungkin):** **<span style='font-size: 24px; color: #4CAF50;'>{predicted_genre_result}</span>**", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Top 5 Persentase Prediksi:")

            top_indices = np.argsort(all_probs)[::-1]

            for i in range(min(5, len(genre_mapping))):
                genre_name = genre_mapping[top_indices[i]]
                probability = all_probs[top_indices[i]] * 100
                st.write(f"- {genre_name}: **{probability:.2f}%**")

        os.unlink(tmp_file_path)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        st.info("Pastikan file audio tidak rusak dan dalam format yang didukung (WAV, MP3, OGG).")
