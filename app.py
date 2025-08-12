import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import soundfile as sf
import tempfile
import os

# Load model dan encoder
model = tf.keras.models.load_model("model_genre_musik.h5")
label_encoder = joblib.load("label_encoder.pkl")

# Fungsi ekstraksi MFCC
def extract_mfcc(file_path, n_mfcc=130, n_fft=2048, hop_length=512, num_segments=1):
    signal, sr = librosa.load(file_path, sr=22050)
    samples_per_segment = int(len(signal) / num_segments)

    mfccs = []
    for s in range(num_segments):
        start_sample = samples_per_segment * s
        finish_sample = start_sample + samples_per_segment

        mfcc = librosa.feature.mfcc(
            y=signal[start_sample:finish_sample],
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mfcc = mfcc.T
        mfccs.append(mfcc)

    mfccs = np.array(mfccs)
    return mfccs

# UI Streamlit
st.title("🎵 Aplikasi Klasifikasi Genre Musik")
st.write("Unggah file audio untuk memprediksi genre musik.")

uploaded_file = st.file_uploader("Pilih file audio", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Simpan sementara file audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_path = tmp_file.name
        data, samplerate = sf.read(uploaded_file)
        sf.write(tmp_path, data, samplerate)

    # Putar audio
    st.audio(uploaded_file, format="audio/wav")

    # Ekstraksi fitur
    mfccs = extract_mfcc(tmp_path, num_segments=1)
    mfccs_scaled = mfccs[..., np.newaxis]  # Tambahkan dimensi channel

    # Prediksi
    prediction = model.predict(mfccs_scaled)
    avg_prediction = np.mean(prediction, axis=0)  # rata-rata prediksi per segmen
    predicted_index = np.argmax(avg_prediction)
    predicted_genre = label_encoder.inverse_transform([predicted_index])[0]
    max_prob = np.max(avg_prediction)

    confidence_threshold = 0.5  # 50%

    if max_prob < confidence_threshold:
        st.write(
            "**Genre yang Diprediksi:** "
            "<span style='font-size: 24px; color: #FFA500;'>Genre tidak dapat dipastikan</span>",
            unsafe_allow_html=True
        )
        st.info(f"Model kurang yakin dengan prediksinya (keyakinan tertinggi: {max_prob*100:.2f}%). "
                "Input mungkin di luar genre yang dilatih atau bukan musik.")
    else:
        st.write(
            f"**Genre yang Diprediksi:** "
            f"<span style='font-size: 24px; color: #00BFFF;'>{predicted_genre}</span>",
            unsafe_allow_html=True
        )

        # Tampilkan Top 5 Genre
        top_5_indices = np.argsort(avg_prediction)[::-1][:5]
        top_5_genres = label_encoder.inverse_transform(top_5_indices)
        top_5_probs = avg_prediction[top_5_indices] * 100

        st.subheader("Top 5 Genre Terprediksi:")
        for genre, prob in zip(top_5_genres, top_5_probs):
            st.write(f"{genre}: {prob:.2f}%")

    # Hapus file sementara
    os.remove(tmp_path)
