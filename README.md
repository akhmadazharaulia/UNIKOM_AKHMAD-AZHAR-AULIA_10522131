# 🍜☕ Sedjenak Mie & Kopi — Aplikasi Prediksi Penjualan SARIMA

Aplikasi Streamlit untuk implementasi data mining menggunakan metode **SARIMA (Seasonal
AutoRegressive Integrated Moving Average)** dalam memprediksi penjualan harian, serta
mengonversi hasil prediksi menjadi estimasi kebutuhan bahan baku berdasarkan struktur
**Bill of Materials (BOM)**.

Dibangun berdasarkan skripsi:
*"Implementation of Data Mining Using Seasonal AutoRegressive Integrated Moving Average
(SARIMA) Method for Sales Prediction at Sedjenak Mie & Kopi"* — Akhmad Azhar Aulia, UNIKOM 2026.

---

## 📁 Struktur Aplikasi

```
sedjenak_app/
├── Home.py                              # Halaman utama + upload data
├── pages/
│   ├── 1_📊_Prediksi_Penjualan.py       # Pipeline CRISP-DM lengkap + SARIMA
│   ├── 2_🧾_Kebutuhan_Bahan_Baku.py     # Integrasi BOM
│   └── 3_📈_Evaluasi_Model.py            # Dashboard MAE/RMSE/MAPE
├── utils.py                              # Fungsi bersama
└── requirements.txt
```

## 🚀 Cara Menjalankan

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Jalankan aplikasi:
   ```bash
   streamlit run Home.py
   ```

3. Buka browser ke `http://localhost:8501`

## 📋 Cara Pakai

### 1. Home — Upload Data
- Upload **data transaksi POS** (CSV/XLSX) dari Loyverse atau sistem POS lain.
  Wajib ada kolom yang merepresentasikan: **Tanggal**, **SKU**, **Nama Item**, **Jumlah Terjual**.
- Upload **data BOM** (CSV/XLSX). Wajib ada kolom: **SKU**, **Bahan Baku**, **Qty Resep**, (opsional: Satuan, Menu).
- Aplikasi otomatis mendeteksi nama kolom yang umum (Date/Tanggal, SKU/Kode, dst), namun
  bisa disesuaikan manual jika deteksi salah.

### 2. 📊 Prediksi Penjualan
- Jalankan **Data Preparation** (cleaning → agregasi harian → train/test split 80:20).
- Lihat **Uji Stasioneritas (ADF Test)** dan proses **differencing** otomatis.
- Lihat plot **ACF & PACF** untuk membantu menentukan parameter model.
- Atur parameter **SARIMA(p,d,q)(P,D,Q,s)** — default sesuai skripsi: (1,1,1)(1,1,1,7).
- Latih model, lihat **summary statistik**, **diagnostik residual**.
- Lihat perbandingan **aktual vs prediksi** pada data testing.
- Generate **prediksi masa depan** (1–90 hari ke depan) dan download hasilnya.

### 3. 🧾 Kebutuhan Bahan Baku
- Pilih SKU/menu yang ingin dikonversi ke kebutuhan bahan baku.
- Atur proporsi kontribusi tiap SKU terhadap total prediksi penjualan harian.
- Lihat tabel detail kebutuhan bahan baku per hari per SKU.
- Lihat ringkasan total kebutuhan per jenis bahan baku (dengan grafik top 10).
- Lihat tren kebutuhan harian per bahan baku tertentu.
- Download semua hasil dalam format CSV.

### 4. 📈 Evaluasi Model
- Lihat metrik **MAE, RMSE, MAPE** terpisah untuk **weekday** dan **weekend**.
- Lihat breakdown evaluasi per minggu.
- Lihat tren MAPE sepanjang periode testing.
- Baca interpretasi otomatis hasil evaluasi.

## 🛠️ Catatan Teknis

- Model SARIMA menggunakan `statsmodels.tsa.statespace.sarimax.SARIMAX`.
- Parameter default mengikuti hasil penelitian skripsi: **SARIMA(1,1,1)(1,1,1)₇**,
  namun dapat diubah langsung di antarmuka untuk eksperimen dengan dataset lain.
- Deteksi kolom otomatis (case-insensitive, partial match) untuk mengakomodasi variasi
  format ekspor dari berbagai sistem POS (Loyverse, Moka, dll).
- Untuk dataset sangat besar (>100,000 baris), proses fitting model SARIMA dapat memakan
  waktu beberapa detik hingga menit tergantung kompleksitas parameter musiman.
