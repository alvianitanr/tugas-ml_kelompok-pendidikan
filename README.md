# Prediksi Performa Siswa

Web prediktif untuk memprediksi kelulusan siswa pada mata pelajaran Matematika dari atribut/data baru.

## Sumber dataset

- **UCI Machine Learning Repository — Student Performance Data Set**
- https://archive.ics.uci.edu/dataset/320/student+performance
- File yang dipakai: `student-mat.csv` (395 baris, 33 kolom), disalin ke `data/raw/`.
- Target: `G3` (nilai akhir 0–20). Dikonversi menjadi label biner: **Lulus** jika `G3 >= 10`, sebaliknya **Tidak Lulus**.

## Struktur

```
student-performance/
├── app.py                 # web server (Python stdlib)
├── templates/index.html   # tampilan form + hasil
├── src/train.py           # training dan ekspor model
├── data/raw/              # dataset UCI + student.txt (deskripsi atribut)
├── models/                # model.joblib, metadata.json, metrics.json
└── requirements.txt
```

## Cara menjalankan

```bash
pip install -r requirements.txt
python src/train.py
python app.py
```

Buka http://localhost:8000. Endpoint `GET /health` tersedia untuk pengecekan server.

## Fitur input

15 fitur numerik: `age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`, `G1`, `G2`.

17 fitur kategorikal: `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`.

`G3` **tidak** dijadikan input agar tidak terjadi data leakage. `G1` dan `G2` adalah nilai ujian sebelumnya yang sudah tersedia sebelum nilai akhir, sehingga sah dipakai.

## Model dan evaluasi

Pipeline: `StandardScaler` untuk numerik + `OneHotEncoder(handle_unknown="ignore")` untuk kategorikal, lalu `RandomForestClassifier` (300 pohon, `min_samples_leaf=2`, `class_weight="balanced"`). Seluruh preprocessing dan model disimpan dalam satu file `models/model.joblib`.

Split data 80/20 dengan stratifikasi; 5-fold cross-validation pada data latih. Metrik terbaru ada di `models/metrics.json` dan ditampilkan di header web.
