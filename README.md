# 🎵 Aplikasi Streamlit untuk Klasifikasi Genre Musik

Repository ini adalah bagian dari proyek penelitian skripsi oleh **Muhammad Zharfan Alfanso (51421100)**.  
Model dibangun menggunakan Tensorflow Python dengan algoritma *Convolutional Neural Network (CNN)* dan *Bidirectional Long Short-Term Memory (BiLSTM)* untuk melakukan klasifikasi genre musik berdasarkan fitur audio **Mel-Spectrogram** terhadap 12 genre musik, yaitu:

<table>
  <tr>
    <td>blues</td>
    <td>classical</td>
    <td>country</td>
    <td>disco</td>
  </tr>
  <tr>
    <td>edm</td>
    <td>hiphop</td>
    <td>jazz</td>
    <td>lofi</td>
  </tr>
  <tr>
    <td>metal</td>
    <td>pop</td>
    <td>reggae</td>
    <td>rock</td>
  </tr>
</table>

Aplikasi dibangun menggunakan *framework* Streamlit yang dapat diakses pada link berikut:
[Link Website Streamlit Klasifikasi Genre Musik](https://muhzarfan-klasifikasi-genre-musik.streamlit.app/)

---

## 🔗 Repository & Dataset

- Repo GitHub: [muhzarfan/klasifikasi-genre-musik](https://github.com/muhzarfan/klasifikasi-genre-musik)  
- Dataset Audio Musik: [Download zip dataset di Google Drive](https://drive.google.com/file/d/1RkBDSUPzerNXs4yWHO5JksE9baFod4W1/view?usp=drive_link)

---

## 🚀 Fitur Utama

- Upload file audio (*format: .wav, .mp3, .ogg*)  
- Ekstraksi fitur audio otomatis  
- Prediksi genre musik menggunakan model yang sudah dilatih  
- Menampilkan top 5 genre teratas  
- Pemutar audio bawaan di aplikasi  

---

## 📂 Struktur Repository

```plaintext
klasifikasi-genre-musik/
├── model/                 # Folder untuk menyimpan model 
├── README.md              # Dokumentasi reporsitori
├── app.py                 # Aplikasi utama Streamlit
├── crnn-genre-musik.ipynb # Notebook untuk membuat model klasifikasi
└── requirements.txt       # Daftar library Python yang digunakan
```

---

## ⚙️ Instalasi & Menjalankan Aplikasi

Langkah cara penggunaan aplikasi dapat dilihat sebagai berikut:

### 1. Clone repository

```bash
git clone https://github.com/muhzarfan/klasifikasi-genre-musik.git
cd klasifikasi-genre-musik
```
### 2. Instalasi Library (Sesuai dengan `requirements.txt`)

Pastikan Anda sudah berada di folder project, lalu jalankan perintah berikut untuk menginstal semua library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

Isi file `requirements.txt` adalah sebagai berikut:
```bash
streamlit==1.41.1
numpy==1.26.4
librosa==0.11.0
tensorflow==2.19.0
keras==3.10.0
```

### 3. Menjalankan Aplikasi

Setelah semua library terinstal, jalankan aplikasi Streamlit dengan perintah berikut:

```bash
streamlit run app.py
```

### 4. Membuat Model (Opsional)

Jika ingin mencoba membuat model sendiri, Anda dapat menggunakan file notebook **`crnn-genre-musik.ipynb`**.  

#### Instalasi Library
Sebelum menjalankan notebook, pastikan semua dependensi sudah terinstal. Jalankan perintah berikut di terminal:

```bash
pip install tensorflow
pip install keras
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install librosa
```

#### Download Dataset
Unduh dataset audio musik terlebih dahulu melalui link berikut:  
[📥 Download Dataset (Google Drive)](https://drive.google.com/file/d/1RkBDSUPzerNXs4yWHO5JksE9baFod4W1/view?usp=drive_link)

#### Struktur Folder
Setelah dataset berhasil diunduh dan diekstrak, pastikan struktur folder sesuai berikut:

```plaintext
klasifikasi-genre-musik/
├── audio/                 # Folder untuk dataset audio
├── model/                 # Folder untuk menyimpan model 
└── crnn-genre-musik.ipynb # Notebook untuk membuat model klasifikasi
```

#### Menjalankan Notebook
1. Buka file **`crnn-genre-musik.ipynb`** di Jupyter Notebook atau JupyterLab.  
2. Jalankan cell secara berurutan dari atas hingga bawah.  
3. Model yang telah dilatih akan tersimpan otomatis ke dalam folder **`model/`**.  

---
