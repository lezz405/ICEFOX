import json
import os
import re
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

# ================= CONFIG =================
API_KEY = os.getenv("GEMINI_API_KEY")

# URUTAN MODEL: Utama -> Cadangan
# Kalo yang atas 503 / penuh, otomatis pindah ke bawahnya
MODEL_LIST = [
    "gemini-3.5-flash-lite",  # Utama lo
    "gemini-2.5-flash-lite",  # Cadangan 1 - paling stabil, jarang 503
    "gemini-2.0-flash",       # Cadangan 2
    "gemini-2.5-flash",       # Cadangan 3
]

MAX_BODY_SIZE = 5_000_000
DEFAULT_CITY = "Jakarta"  # <-- Lo udah di Jakarta, bukan Pulogebang lagi

SYSTEM_PROMPT = """
Kamu adalah ICEFOX AI, teman nongkrong pintar yang dibuat oleh anak SMA.
Gaya bicara: gaul, santai, panggil user "bro", pake "gue" dan "lo".
Jawab langsung ke intinya, jangan formal.
Kalau user kirim foto, jelaskan isi foto itu.
Jangan pernah ngaku kamu ChatGPT / Gemini. Kamu ICEFOX AI.
"""

# ================= CUACA =================
def get_weather(city=DEFAULT_CITY):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=id&format=json"
        with urllib.request.urlopen(geo_url, timeout=10) as r:
            geo = json.loads(r.read().decode())
        if not geo.get("results"):
            return None
        loc = geo["results"][0]
        lat, lon, nama = loc["latitude"], loc["longitude"], loc["name"]

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=Asia%2FJakarta&forecast_days=2"
        with urllib.request.urlopen(weather_url, timeout=10) as r:
            data = json.loads(r.read().decode())

        cur = data["current"]
        daily = data["daily"]
        kode = cur["weather_code"]
        status = {0:"Cerah ☀️", 1:"Cerah Berawan 🌤️", 2:"Berawan ⛅", 3:"Mendung ☁️", 45:"Berkabut 🌫️", 51:"Gerimis 🌦️", 61:"Hujan Ringan 🌧️", 63:"Hujan 🌧️", 65:"Hujan Deras ⛈️", 95:"Hujan Badai ⛈️"}.get(kode, "Berawan")

        return (
            f"Cuaca di {nama} saat ini:\n"
            f"{status}\n"
            f"Suhu: {cur['temperature_2m']}°C | Kelembapan: {cur['relative_humidity_2m']}%\n"
            f"Hujan: {cur['precipitation']}mm | Angin: {cur['wind_speed_10m']} km/jam\n\n"
            f"Besok: {daily['temperature_2m_min'][1]}°C - {daily['temperature_2m_max'][1]}°C, Peluang Hujan {daily['precipitation_probability_max'][1]}%"
        )
    except Exception as e:
        print("WEATHER ERROR:", e)
        return None

def ambil_kota_dari_chat(pesan, kota_dropdown=DEFAULT_CITY):
    pesan = pesan.lower().strip()
    m = re.search(r"cuaca(?:.*?) di ([a-z\s]+?)(?: gimana| bagaimana| dong| bro| ya| \?|$)", pesan)
    if m:
        kota = m.group(1).strip()
        if kota:
            return kota.title()
    m = re.search(r"cuaca\s+([a-z\s]+?)(?: gimana| bagaimana| dong| bro| \?|$)", pesan)
    if m:
        kota = m.group(1).strip()
        if kota not in ["", "gimana", "hari ini", "besok", "saat ini", "sekarang"]:
            return kota.title()
    m = re.search(r"(?:di\s+)?([a-z\s]+?)\s+(hujan|panas|dingin|mendung)", pesan)
    if m:
        kota = m.group(1).strip()
        if kota not in ["", "cuaca"]:
            return kota.title()
    return kota_dropdown

# ================= HELPERS FOTO & CHAT =================
def extract_image(data):
    url = data.get("image")
    if not isinstance(url, str) or not url.startswith("data:image"):
        return None
    try:
        header, b64 = url.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        return {"inlineData": {"mimeType": mime, "data": b64}}
    except Exception:
        return None

def build_contents(messages, image_part):
    contents = []
    for msg in messages[-12:]:
        if not isinstance(msg, dict):
            continue
        role, content = msg.get("role"), msg.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        parts = [{"text": content[:4000]}]
        if image_part and msg is messages[-1] and role == "user":
            parts.append(image_part)
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    return contents

# ================= GEMINI - VERSI FIX 503 =================
def call_gemini_single_model(model_name, contents):
    """Coba 1 model, kalo 503 lempar error biar bisa pindah model."""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1500}
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": API_KEY},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=90) as res:
        result = json.loads(res.read().decode())
    parts = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()

def call_gemini(contents):
    """
    Coba semua model satu-satu.
    Kalo 503 / 429 / 500 -> langsung pindah model berikutnya, jangan nyerah.
    Kalo 400 / 401 / 403 -> itu salah API Key / format, langsung stop.
    """
    last_error = None
    last_code = 502

    for idx, model_name in enumerate(MODEL_LIST):
        print(f"[GEMINI] Coba model {idx+1}/{len(MODEL_LIST)}: {model_name}")

        # Retry 2x untuk model yang sama, dengan jeda makin lama
        # Percobaan 1: langsung, Percobaan 2: tunggu 2 detik
        for attempt in range(2):
            try:
                answer = call_gemini_single_model(model_name, contents)
                if answer:
                    print(f"[GEMINI] Berhasil pake {model_name}")
                    return answer
                else:
                    print(f"[GEMINI] {model_name} jawab kosong, coba lagi...")
                    continue

            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode()[:1000]
                except:
                    pass

                print(f"[GEMINI] {model_name} ERROR {e.code}: {body[:500]}")
                last_error = body or f"HTTP {e.code}"
                last_code = e.code

                # Error yang GAK USAH di-retry / pindah model, karena percuma
                if e.code in (400, 401, 403, 404):
                    # 400 = format salah, 401/403 = API Key salah, 404 = nama model salah
                    # Kalo nama model salah, lanjut ke model berikutnya aja
                    if e.code == 404:
                        print(f"[GEMINI] Model {model_name} gak ketemu, pindah model...")
                        break
                    # Kalo API Key salah, langsung lempar
                    raise Exception(f"API Key salah / tidak valid ({e.code}): {body[:300]}")

                # 429 = limit, 500 / 502 / 503 = server penuh
                # Kasih jeda dulu sebelum retry
                if attempt == 0:
                    wait = 2
                    print(f"[GEMINI] Tunggu {wait} detik sebelum retry {model_name}...")
                    time.sleep(wait)
                    continue
                else:
                    # Udah 2x gagal di model ini, pindah ke model berikutnya
                    print(f"[GEMINI] {model_name} nyerah, pindah model...")
                    break

            except Exception as e:
                print(f"[GEMINI] {model_name} Exception: {e}")
                last_error = str(e)
                last_code = 500
                break

        # Jeda dikit sebelum pindah model, biar gak langsung kena limit lagi
        if idx < len(MODEL_LIST) - 1:
            time.sleep(1)

    # Kalo semua model udah dicoba dan gagal semua
    raise Exception(f"Semua model Gemini gagal. Terakhir: {last_code} - {last_error}")

# ================= SERVER =================
class FrostServer(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/":
            filepath = "index.html"
            content_type = "text/html; charset=utf-8"
        elif self.path == "/logo.png":
            filepath = "logo.png"
            content_type = "image/png"
        elif self.path == "/logo.jpg":
            filepath = "logo.jpg"
            content_type = "image/jpeg"
        elif self.path == "/favicon.ico":
            filepath = "logo.png"
            content_type = "image/png"
        else:
            return self.send_error(404)

        try:
            with open(filepath, "rb") as f:
                page = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        except FileNotFoundError:
            self.send_error(404, f"{filepath} tidak ditemukan")

    def do_POST(self):
        if self.path != "/api/chat":
            return self.send_json({"error": "Endpoint tidak ditemukan"}, 404)
        if not API_KEY:
            return self.send_json({"error": "GEMINI_API_KEY belum diatur."}, 401)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_BODY_SIZE:
                return self.send_json({"error": "Ukuran pesan tidak valid."}, 400)
            data = json.loads(self.rfile.read(length))
            messages = data.get("messages", [])
            if not isinstance(messages, list) or not messages:
                return self.send_json({"error": "Format pesan salah."}, 400)

            last_msg = messages[-1].get("content", "").lower() if messages else ""
            if any(k in last_msg for k in ["cuaca", "hujan", "panas", "dingin", "weather", "mendung"]):
                city = ambil_kota_dari_chat(last_msg, data.get("city") or DEFAULT_CITY)
                hasil = get_weather(city)
                if hasil:
                    return self.send_json({"reply": f"Gue cek langsung dari satelit ya bro.\n\n{hasil}"})
                else:
                    return self.send_json({"reply": f"Waduh bro, kota '{city}' gak ketemu di satelit. Coba ketik nama kota lain yang lebih umum ya."})

            image_part = extract_image(data)
            contents = build_contents(messages, image_part)
            if not contents or contents[-1]["role"] != "user":
                return self.send_json({"error": "Pesan pengguna tidak ditemukan."}, 400)

            answer = call_gemini(contents)
            if not answer:
                return self.send_json({"error": "Jawaban Gemini kosong."}, 502)
            self.send_json({"reply": answer})

        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode()[:1000]
            except:
                body = ""
            print("GEMINI ERROR:", e.code, body)
            # Terusin kode aslinya ke frontend, jangan dibikin 502 terus
            # Biar frontend tau kalo itu 503 dan bisa retry
            self.send_json({"error": f"Gemini error {e.code}: {body[:300]}"}, e.code if e.code in (429, 500, 502, 503) else 502)
        except Exception as e:
            print("SERVER ERROR:", e)
            # Kalo pesannya udah jelas (misal API Key salah), kirim apa adanya
            msg = str(e)
            if "API Key" in msg:
                return self.send_json({"error": msg}, 401)
            if "Semua model" in msg:
                return self.send_json({"error": "Gemini lagi penuh semua bro (503). Tunggu 10-20 detik, coba lagi ya."}, 503)
            self.send_json({"error": "Terjadi kesalahan pada server: " + msg[:300]}, 500)

if __name__ == "__main__":
   if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8000))
    HOST = "0.0.0.0"

    print(f"ICEFOX AI jalan di http://{HOST}:{PORT}")
    print(f"Model utama: {MODEL_LIST[0]} | Cadangan: {', '.join(MODEL_LIST[1:])}")
    print(f"Kota default: {DEFAULT_CITY}")
    if not API_KEY:
        print("⚠️  WARNING: GEMINI_API_KEY belum diatur!")

    HTTPServer((HOST, PORT), FrostServer).serve_forever()