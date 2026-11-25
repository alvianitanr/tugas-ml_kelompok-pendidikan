import json
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"

model = joblib.load(MODEL_DIR / "model.joblib")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
FEATURES = metadata["numeric"] + metadata["categorical"]

LABELS = {
    "school": "Sekolah",
    "sex": "Jenis kelamin",
    "age": "Usia",
    "address": "Alamat tempat tinggal",
    "famsize": "Ukuran keluarga",
    "Pstatus": "Status tinggal orang tua",
    "Medu": "Pendidikan ibu",
    "Fedu": "Pendidikan ayah",
    "Mjob": "Pekerjaan ibu",
    "Fjob": "Pekerjaan ayah",
    "reason": "Alasan memilih sekolah",
    "guardian": "Wali siswa",
    "traveltime": "Waktu tempuh ke sekolah",
    "studytime": "Waktu belajar per minggu",
    "failures": "Jumlah kelas gagal",
    "schoolsup": "Bantuan belajar dari sekolah",
    "famsup": "Bantuan belajar dari keluarga",
    "paid": "Les tambahan",
    "activities": "Kegiatan ekstrakurikuler",
    "nursery": "Pernah masuk TK",
    "higher": "Rencana melanjutkan kuliah",
    "internet": "Internet di rumah",
    "romantic": "Sedang menjalin hubungan",
    "famrel": "Kualitas hubungan keluarga",
    "freetime": "Jumlah waktu luang",
    "goout": "Frekuensi keluar rumah",
    "Dalc": "Alkohol pada hari kerja",
    "Walc": "Alkohol pada akhir pekan",
    "health": "Kondisi kesehatan",
    "absences": "Jumlah ketidakhadiran",
    "G1": "Nilai ujian pertama",
    "G2": "Nilai ujian kedua",
}

VALUE_LABELS = {
    "GP": "Sekolah Gabriel Pereira",
    "MS": "Sekolah Mousinho da Silveira",
    "M": "Laki-laki",
    "F": "Perempuan",
    "U": "Perkotaan",
    "R": "Perdesaan",
    "LE3": "3 orang atau kurang",
    "GT3": "Lebih dari 3 orang",
    "A": "Tinggal bersama",
    "T": "Tinggal terpisah",
    "teacher": "Guru",
    "health": "Tenaga kesehatan",
    "services": "Pekerjaan jasa",
    "at_home": "Ibu rumah tangga",
    "other": "Lainnya",
    "home": "Dekat rumah",
    "reputation": "Reputasi sekolah",
    "course": "Program studi diminati",
    "mother": "Ibu",
    "father": "Ayah",
    "yes": "Ya",
    "no": "Tidak",
}

HINTS = {
    "Medu": "0=tidak sekolah, 1=SD, 2=SMP, 3=SMA, 4=kuliah",
    "Fedu": "0=tidak sekolah, 1=SD, 2=SMP, 3=SMA, 4=kuliah",
    "studytime": "1=kurang dari 2 jam, 2=2-5 jam, 3=5-10 jam, 4=lebih dari 10 jam",
    "traveltime": "1=kurang dari 15 menit, 4=lebih dari 30 menit",
    "famrel": "1=sangat buruk, 5=sangat baik",
    "freetime": "1=sangat sedikit, 5=sangat banyak",
    "goout": "1=sangat jarang, 5=sangat sering",
    "Dalc": "1=sangat jarang, 5=sangat sering",
    "Walc": "1=sangat jarang, 5=sangat sering",
    "health": "1=sangat buruk, 5=sangat baik",
    "G1": "Rentang 0-20",
    "G2": "Rentang 0-20",
}


def render_field(name, values):
    submitted = values.get(name, "")
    label = LABELS.get(name, name)
    hint = f'<small class="hint">{escape(HINTS[name])}</small>' if name in HINTS else ""
    if name in metadata["categorical"]:
        options = []
        for category in metadata["categories"][name]:
            selected = " selected" if category == submitted else ""
            text = VALUE_LABELS.get(category, category)
            options.append(
                f'<option value="{escape(category)}"{selected}>{escape(text)}</option>'
            )
        control = f"<select name=\"{name}\">{''.join(options)}</select>"
    else:
        info = metadata["ranges"][name]
        value = submitted if submitted else round(info["mean"])
        control = (
            f'<input type="number" name="{name}" value="{escape(str(value))}" '
            f'min="{info["min"]}" max="{info["max"]}" step="1" required>'
        )
    return f'<label><span>{escape(label)}</span>{control}{hint}</label>'


def render_page(result="", error="", values=None):
    values = values or {}
    fields = "".join(render_field(name, values) for name in FEATURES)
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        html.replace("{{FIELDS}}", fields)
        .replace("{{RESULT}}", result)
        .replace("{{ERROR}}", error)
        .replace("{{ACCURACY}}", str(metrics["accuracy"]))
        .replace("{{F1}}", str(metrics["f1"]))
        .replace("{{ROWS}}", str(metadata["dataset"]["rows"]))
        .replace("{{SOURCE}}", escape(metadata["dataset"]["source"]))
    )


def parse_payload(raw_body):
    parsed = parse_qs(raw_body, keep_blank_values=True)
    row = {}
    for name in FEATURES:
        value = parsed.get(name, [""])[0]
        if name in metadata["categorical"]:
            if value not in metadata["categories"][name]:
                raise ValueError(f"Nilai {name} tidak valid: {value}")
            row[name] = value
        else:
            number = int(value)
            info = metadata["ranges"][name]
            if not info["min"] <= number <= info["max"]:
                raise ValueError(f"{name} harus antara {info['min']} dan {info['max']}")
            row[name] = number
    return row


def predict(row):
    frame = pd.DataFrame([row])[FEATURES]
    probability = float(model.predict_proba(frame)[0, 1])
    passed = probability >= 0.5
    label = metadata["positive_label"] if passed else metadata["negative_label"]
    return (
        f'<div class="result {"pass" if passed else "fail"}">'
        f"<h2>{label}</h2>"
        f"<p>Probabilitas lulus: <strong>{probability * 100:.2f}%</strong></p>"
        f'<div class="bar"><span style="width:{probability * 100:.2f}%"></span></div>'
        f"<small>Batas keputusan: 50%</small>"
        f"</div>"
    )


class Handler(BaseHTTPRequestHandler):
    def send_html(self, body, status=200):
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            payload = json.dumps({"status": "ok", "model": "ready"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_html(render_page())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8")
        try:
            row = parse_payload(raw_body)
            result = predict(row)
            error = ""
        except (ValueError, TypeError) as exc:
            result = ""
            error = f'<div class="error">Input tidak valid: {escape(str(exc))}</div>'
        submitted = {key: items[0] for key, items in parse_qs(raw_body, keep_blank_values=True).items()}
        self.send_html(render_page(result, error, submitted))


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
    print("Web prediktif berjalan di http://localhost:8000")
    server.serve_forever()
