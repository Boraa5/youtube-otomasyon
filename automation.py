import os
import re
import json
import time
import random
import subprocess
from pathlib import Path

import requests
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


# ============================================================
# KAYIP HİKAYELER - ÜNLÜLER VE FENOMENLER OTOMASYONU
# Ana video: yaklaşık 7 dakika
# Short: yaklaşık 45-60 saniye
# ============================================================

OUT = Path("work")
OUT.mkdir(exist_ok=True)

GEMINI = os.environ["GEMINI_API_KEY"]
PEXELS = os.environ["PEXELS_API_KEY"]
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

client = genai.Client(api_key=GEMINI)

PIPER = OUT / "tr_TR-dfki-medium.onnx"
PIPER_CFG = OUT / "tr_TR-dfki-medium.onnx.json"
PIPER_URL = (
    "https://huggingface.co/rhasspy/piper-voices/"
    "resolve/v1.0.0/tr/tr_TR/dfki/medium/"
)

MAIN_MIN_WORDS = 750
MAIN_MAX_WORDS = 1050
SHORT_MIN_WORDS = 90
SHORT_MAX_WORDS = 150


# ============================================================
# YARDIMCILAR
# ============================================================

def run(command, inp=None):
    print("$", " ".join(map(str, command)))
    subprocess.run(command, input=inp, text=True, check=True)


def wc(text):
    return len(re.findall(r"\b[\wÇĞİÖŞÜçğıöşü'-]+\b", str(text)))


def duration(file_path):
    result = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ],
        text=True
    )
    return float(result.strip())


def download(url, path, minimum=500):
    if path.exists() and path.stat().st_size >= minimum:
        return

    response = requests.get(url, timeout=180)
    response.raise_for_status()
    path.write_bytes(response.content)

    if path.stat().st_size < minimum:
        raise RuntimeError("Eksik veya bozuk dosya: " + str(path))


def clean_query(query):
    query = re.sub(r"[^A-Za-z0-9\s.'&-]", " ", str(query))
    return re.sub(r"\s+", " ", query).strip()


def normalize_queries(queries, count, fallback):
    cleaned = []
    if isinstance(queries, list):
        for query in queries:
            query = clean_query(query)
            if query:
                cleaned.append(query)

    cleaned = cleaned[:count]
    n = 1
    while len(cleaned) < count:
        cleaned.append(fallback if n == 1 else f"{fallback} {n}")
        n += 1
    return cleaned


def normalize_tags(tags):
    cleaned = []
    if isinstance(tags, list):
        for tag in tags:
            tag = str(tag).strip()
            if tag and tag not in cleaned:
                cleaned.append(tag)

    fallback = [
        "Kayıp Hikâyeler",
        "Ünlüler",
        "Fenomenler",
        "Gerçek Hikâyeler",
        "Gizem",
        "İnternet Hikâyeleri",
    ]
    for tag in fallback:
        if len(cleaned) >= 15:
            break
        if tag not in cleaned:
            cleaned.append(tag)
    return cleaned[:15]


def is_quota_error(error_text):
    text = str(error_text).upper()
    return (
        "429" in text
        or "RESOURCE_EXHAUSTED" in text
        or "QUOTA" in text
        or "RATE LIMIT" in text
    )


# ============================================================
# GEMINI
# ============================================================

def gemini_json(prompt, schema, attempts=6):
    last_error = None

    for attempt in range(attempts):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.8
                )
            )
            if not response.text:
                raise RuntimeError("Gemini boş yanıt verdi")
            return json.loads(response.text)

        except Exception as error:
            last_error = str(error)
            if attempt == attempts - 1:
                break

            if is_quota_error(last_error):
                wait = min(300, 30 * (2 ** attempt)) + random.randint(0, 10)
                print(
                    f"Gemini kota/sınır hatası. "
                    f"{wait} saniye bekleniyor... "
                    f"Deneme {attempt + 1}/{attempts}"
                )
            else:
                wait = min(60, 5 * (attempt + 1))
                print(
                    f"Gemini hata: {last_error}. "
                    f"{wait} saniye sonra tekrar denenecek..."
                )

            time.sleep(wait)

    raise RuntimeError("Gemini içerik üretilemedi: " + str(last_error))


# ============================================================
# TÜRKÇE SES
# ============================================================

def setup():
    print("Türkçe ses modeli kontrol ediliyor...")

    download(
        PIPER_URL + "tr_TR-dfki-medium.onnx",
        PIPER,
        50_000_000
    )
    download(
        PIPER_URL + "tr_TR-dfki-medium.onnx.json",
        PIPER_CFG,
        500
    )
    print("Türkçe ses modeli hazır.")


# ============================================================
# HİKAYE PLANI
# ============================================================

PLAN_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "description": {"type": "STRING"},
        "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
        "narration": {"type": "STRING"},
        "short_title": {"type": "STRING"},
        "short_narration": {"type": "STRING"},
        "scene_queries": {"type": "ARRAY", "items": {"type": "STRING"}},
        "short_queries": {"type": "ARRAY", "items": {"type": "STRING"}},
        "thumbnail_query": {"type": "STRING"},
    },
    "required": [
        "title", "description", "tags", "narration",
        "short_title", "short_narration",
        "scene_queries", "short_queries", "thumbnail_query"
    ]
}


def plan():
    prompt = f"""
"Kayıp Hikâyeler" adlı Türkçe YouTube kanalı için TEK bir gerçek,
belgelenebilir ve güçlü merak uyandıran hikâye hazırla.

KANALIN KONUSU:
Türkiye'deki tanınmış sanatçılar, ünlüler, sosyal medya fenomenleri
ve kamuoyunca bilinen kişilerin hayatlarında yaşanmış gerçek,
şaşırtıcı, az bilinen veya yıllarca merak edilmiş olaylar.

ÖNEMLİ:
- İftira, uydurma skandal veya doğrulanmamış suçlama üretme.
- Bir kişi hakkında tartışmalı iddia varsa kesin gerçek gibi anlatma.
- Özel hayatı gereksiz biçimde hedef alma.
- Sadece kamuya açık, doğrulanabilir olayları anlat.
- Aynı "12 yıl kayboldu", "duvarın arkasında bulundu" kalıplarını
  kullanma.
- Her videoda farklı olay türü ve farklı başlık yapısı kullan.
- Arkeoloji veya tarih dersi gibi yazma.

ANA VİDEO:
Hedef yaklaşık 7 dakikalık doğal Türkçe seslendirme.
NARRATION en az {MAIN_MIN_WORDS}, en fazla {MAIN_MAX_WORDS} kelime olsun.
Gereksiz tekrar ve boş uzatma yapma.

İlk cümle doğrudan en şaşırtıcı ayrıntıyla başlasın.
Selamlama, kanal tanıtımı, "bugün sizlere" veya "bu videoda"
ifadeleri kullanma.

YAPI:
1. İlk 5 saniyede güçlü merak
2. Olayın kısa arka planı
3. Beklenmedik gelişme
4. Kamuoyunun gördüğü veya bilmediği ayrıntılar
5. Araştırılabilir gerçekler ve dengeli açıklama
6. Güçlü dönüm noktası
7. Sonucun ortaya çıkışı
8. Kısa ve etkileyici kapanış

SHORT:
Ana videodaki aynı gerçek olaya bağlı olsun.
Kopya özet olmasın; en çarpıcı noktayı anlatsın.
İlk 3 saniyede merak oluştursun.
Doğal olarak yaklaşık 45-60 saniye olacak şekilde
{SHORT_MIN_WORDS}-{SHORT_MAX_WORDS} kelime arasında olsun.

BAŞLIKLAR:
Her seferinde farklı yapı kullan.
Sürekli "X Yıl Boyunca..." veya "Yıllarca..." kalıbını kullanma.
Yalan clickbait yapma.

PEXELS:
Ana video için tam 12 İngilizce görsel sorgusu.
Short için tam 6 İngilizce görsel sorgusu.
Sorgular 2-6 kelime, genel ve gerçekçi olsun.
Gerçek kişiye ait sonuç bulmak zor olabileceği için gerektiğinde
olayın atmosferini ve bağlamını gösteren genel görseller kullan.

THUMBNAIL:
Tek İngilizce, güçlü, genel ve Pexels'te aranabilir sorgu.

Sadece geçerli JSON döndür.
"""

    data = gemini_json(prompt, PLAN_SCHEMA)

    required_text = [
        "title", "description", "narration",
        "short_title", "short_narration", "thumbnail_query"
    ]
    for field in required_text:
        if not str(data.get(field, "")).strip():
            raise RuntimeError("Boş alan: " + field)

    data["title"] = str(data["title"]).strip()
    data["description"] = str(data["description"]).strip()
    data["narration"] = str(data["narration"]).strip()
    data["short_title"] = str(data["short_title"]).strip()
    data["short_narration"] = str(data["short_narration"]).strip()
    data["thumbnail_query"] = clean_query(data["thumbnail_query"]) or "mysterious celebrity documentary"
    data["tags"] = normalize_tags(data.get("tags", []))
    data["scene_queries"] = normalize_queries(
        data.get("scene_queries", []), 12, "documentary investigation"
    )
    data["short_queries"] = normalize_queries(
        data.get("short_queries", []), 6, "mysterious documentary"
    )

    # KRİTİK DÜZELTME:
    # Kısa senaryo yüzünden yeni hikâye üretip konuyu değiştirme.
    # Aynı hikâyeyi tek bir devam çağrısıyla genişlet.
    words = wc(data["narration"])
    if words < MAIN_MIN_WORDS:
        print(
            f"Ana senaryo {words} kelime. "
            "Aynı hikâye korunarak otomatik genişletiliyor..."
        )
        data["narration"] = expand_main_story(data["narration"], data["title"])

    # Short kısa gelirse sadece aynı olayı genişlet.
    short_words = wc(data["short_narration"])
    if short_words < SHORT_MIN_WORDS:
        print(
            f"Short {short_words} kelime. "
            "Aynı olay korunarak genişletiliyor..."
        )
        data["short_narration"] = expand_short_story(
            data["short_narration"], data["title"], data["narration"]
        )

    # Son güvenlik: genişletme başarısızsa rastgele yeniden konu üretme.
    main_words = wc(data["narration"])
    short_words = wc(data["short_narration"])

    if main_words < MAIN_MIN_WORDS:
        raise RuntimeError(
            f"Ana senaryo yeterince uzatılamadı: {main_words} kelime."
        )
    if main_words > MAIN_MAX_WORDS:
        print(f"Uyarı: Ana senaryo {main_words} kelime, kabul ediliyor.")
    if short_words < SHORT_MIN_WORDS:
        raise RuntimeError(
            f"Short yeterince uzatılamadı: {short_words} kelime."
        )

    print("Plan kabul edildi.")
    print("Ana kelime:", main_words)
    print("Short kelime:", short_words)
    return data


def expand_main_story(narration, title):
    schema = {
        "type": "OBJECT",
        "properties": {"narration": {"type": "STRING"}},
        "required": ["narration"]
    }

    prompt = f"""
Aşağıdaki Türkçe YouTube senaryosunu AYNI GERÇEK HİKÂYE ve AYNI KİŞİ
üzerinden genişlet.

Başlık: {title}

KESİNLİKLE yeni konu seçme.
KESİNLİKLE hikâyeyi baştan farklı olayla yazma.
Mevcut girişteki merak duygusunu koru.
Eksik arka planı, olayların sırasını, doğrulanabilir bağlamı ve
önemli dönüm noktalarını doğal biçimde geliştir.

Uydurma bilgi, tarih, alıntı, suçlama veya olay ekleme.
Tartışmalı bilgi varsa kesin gerçek gibi yazma.

Sonuçta TEK PARÇA, doğal, akıcı ve yaklaşık 750-1050 kelimelik
bir senaryo ver.

Mevcut senaryo:
---
{narration}
---

Sadece JSON döndür.
"""
    data = gemini_json(prompt, schema)
    return str(data["narration"]).strip()


def expand_short_story(short_text, title, main_narration):
    schema = {
        "type": "OBJECT",
        "properties": {"short_narration": {"type": "STRING"}},
        "required": ["short_narration"]
    }

    prompt = f"""
Aşağıdaki kısa YouTube Shorts metnini, ana videodaki AYNI gerçek olayı
koruyarak 90-150 kelimeye tamamla.

Başlık: {title}

Yeni olay, yeni kişi veya uydurma bilgi ekleme.
İlk cümledeki merakı koru.
45-60 saniyelik doğal Türkçe konuşma temposuna uygun yaz.

Mevcut Short:
---
{short_text}
---

Ana hikâye bağlamı:
---
{main_narration[:7000]}
---

Sadece JSON döndür.
"""
    data = gemini_json(prompt, schema)
    return str(data["short_narration"]).strip()


# ============================================================
# PEXELS
# ============================================================

def pexels(query, orientation, used):
    query = clean_query(query) or "documentary"

    response = requests.get(
        "https://api.pexels.com/v1/search",
        headers={"Authorization": PEXELS},
        params={
            "query": query,
            "orientation": orientation,
            "per_page": 20
        },
        timeout=60
    )
    response.raise_for_status()

    photos = response.json().get("photos", [])
    if not photos:
        # Sorgu boş sonuç verirse otomasyon çökmek yerine genel arama.
        response = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS},
            params={
                "query": "documentary people",
                "orientation": orientation,
                "per_page": 20
            },
            timeout=60
        )
        response.raise_for_status()
        photos = response.json().get("photos", [])

    if not photos:
        raise RuntimeError("Pexels sonucu bulunamadı: " + query)

    photo = next((x for x in photos if x["id"] not in used), photos[0])
    used.add(photo["id"])

    src = photo["src"]
    image_url = (
        src.get("portrait") if orientation == "portrait"
        else src.get("landscape")
    )
    image_url = image_url or src.get("large2x") or src.get("original")
    return photo, image_url


def images(queries, prefix, orientation):
    files, credits = [], []
    used = set()

    for number, query in enumerate(queries, 1):
        print(f"Görsel {number}/{len(queries)}: {query}")
        photo, image_url = pexels(query, orientation, used)
        file_path = OUT / f"{prefix}_{number:02d}.jpg"

        response = requests.get(image_url, timeout=90)
        response.raise_for_status()
        file_path.write_bytes(response.content)

        if file_path.stat().st_size < 5000:
            raise RuntimeError("Bozuk Pexels görseli: " + str(file_path))

        files.append(file_path)
        credits.append(photo)

    return files, credits


# ============================================================
# TTS
# ============================================================

def tts(text, output):
    text = re.sub(r"\s+", " ", str(text)).strip()
    chunks = []

    while len(text) > 2200:
        split = text.rfind(" ", 0, 2200)
        if split < 500:
            split = 2200
        chunks.append(text[:split])
        text = text[split:].strip()

    if text:
        chunks.append(text)
    if not chunks:
        raise RuntimeError("Seslendirilecek metin boş")

    wavs = []
    for number, chunk in enumerate(chunks):
        wav = OUT / f"{output.stem}_{number:03d}.wav"
        print(f"Ses hazırlanıyor {number + 1}/{len(chunks)}")

        run(
            [
                "python", "-m", "piper",
                "--model", str(PIPER),
                "--output_file", str(wav),
                "--sentence-silence", "0.18",
                "--length-scale", "0.88",
                "--noise-scale", "0.667",
                "--noise-w", "0.8"
            ],
            chunk
        )
        wavs.append(wav)

    concat = OUT / f"{output.stem}_concat.txt"
    concat.write_text(
        "\n".join(
            "file '" + str(w.resolve()).replace("'", "'\\''") + "'"
            for w in wavs
        ),
        encoding="utf-8"
    )

    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat),
        "-c:a", "libmp3lame", "-b:a", "160k",
        str(output)
    ])


# ============================================================
# VİDEO
# ============================================================

def make_video(audio, image_files, output, vertical=False):
    if not image_files:
        raise RuntimeError("Video için görsel bulunamadı")

    total = duration(audio)
    each = max(1.0, total / len(image_files))

    slides = OUT / f"{output.stem}_slides.txt"
    lines = []

    for image_file in image_files:
        escaped = str(image_file.resolve()).replace("'", "'\\''")
        lines.append("file '" + escaped + "'")
        lines.append(f"duration {each:.3f}")

    lines.append(
        "file '" +
        str(image_files[-1].resolve()).replace("'", "'\\''") +
        "'"
    )
    slides.write_text("\n".join(lines), encoding="utf-8")

    if vertical:
        vf = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,format=yuv420p"
        )
    else:
        vf = (
            "scale=1280:720:force_original_aspect_ratio=increase,"
            "crop=1280:720,format=yuv420p"
        )

    run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(slides),
        "-i", str(audio),
        "-vf", vf,
        "-r", "25",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "160k",
        "-shortest",
        "-movflags", "+faststart",
        str(output)
    ])


# ============================================================
# THUMBNAIL
# ============================================================

def thumbnail(source, output):
    run([
        "ffmpeg", "-y", "-i", str(source),
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=increase,"
        "crop=1280:720",
        "-q:v", "7",
        str(output)
    ])

    if output.stat().st_size > 1_900_000:
        run([
            "ffmpeg", "-y", "-i", str(output),
            "-q:v", "10", str(output)
        ])


# ============================================================
# YOUTUBE
# ============================================================

def youtube():
    credentials = Credentials(
        token=None,
        refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"]
    )

    return build(
        "youtube",
        "v3",
        credentials=credentials,
        cache_discovery=False
    )


def upload(api, file_path, title, description, tags, thumbnail_file=None):
    body = {
        "snippet": {
            "title": str(title)[:100],
            "description": str(description)[:5000],
            "tags": [str(tag) for tag in tags[:30]],
            "categoryId": "24"
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
            "containsSyntheticMedia": True
        }
    }

    media = MediaFileUpload(
        str(file_path),
        mimetype="video/mp4",
        resumable=True,
        chunksize=8 * 1024 * 1024
    )

    request = api.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print("YouTube " + str(int(status.progress() * 100)) + "%")

    video_id = response["id"]

    if thumbnail_file and thumbnail_file.exists():
        api.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(
                str(thumbnail_file),
                mimetype="image/jpeg"
            )
        ).execute()

    print("YouTube'a yüklendi:", video_id)
    return video_id


# ============================================================
# ANA PROGRAM
# ============================================================

def main():
    print("=" * 60)
    print("KAYIP HİKÂYELER - ÜNLÜLER VE FENOMENLER")
    print("Hedef ana video: yaklaşık 7 dakika")
    print("Hedef Short: yaklaşık 45-60 saniye")
    print("=" * 60)

    setup()
    data = plan()

    print("=" * 60)
    print("KONU:", data["title"])
    print("Ana kelime:", wc(data["narration"]))
    print("Short kelime:", wc(data["short_narration"]))
    print("=" * 60)

    long_audio = OUT / "long.mp3"
    short_audio = OUT / "short.mp3"

    tts(data["narration"], long_audio)
    tts(data["short_narration"], short_audio)

    long_duration = duration(long_audio)
    short_duration = duration(short_audio)

    print("Ana video süresi:", round(long_duration, 1), "saniye")
    print("Short süresi:", round(short_duration, 1), "saniye")

    # Süre güvenliği: 7 dakika hedef, ama TTS hızına göre doğal fark olabilir.
    if long_duration < 300:
        raise RuntimeError(
            f"Ana video beklenenden kısa: {long_duration:.1f} saniye. "
            "Senaryo 7 dakika hedefini karşılamadı."
        )

    if short_duration < 35:
        raise RuntimeError(
            f"Short beklenenden kısa: {short_duration:.1f} saniye."
        )

    print("Ana video görselleri indiriliyor...")
    long_images, long_credits = images(
        data["scene_queries"], "long", "landscape"
    )

    print("Short görselleri indiriliyor...")
    short_images, short_credits = images(
        data["short_queries"], "short", "portrait"
    )

    long_video = OUT / "long.mp4"
    short_video = OUT / "short.mp4"

    print("Ana video hazırlanıyor...")
    make_video(long_audio, long_images, long_video)

    print("Short hazırlanıyor...")
    make_video(
        short_audio, short_images, short_video, vertical=True
    )

    print("Kapak hazırlanıyor...")
    thumbnail_images, _ = images(
        [data["thumbnail_query"]], "thumbnail", "landscape"
    )

    thumbnail_file = OUT / "thumbnail.jpg"
    thumbnail(thumbnail_images[0], thumbnail_file)

    credits = []
    for photo in long_credits + short_credits:
        photo_url = photo.get("url", "")
        photographer = photo.get("photographer", "Pexels photographer")
        if photo_url:
            credits.append(
                f"Photo by {photographer} on Pexels: {photo_url}"
            )

    unique_credits = list(dict.fromkeys(credits))

    description = (
        data["description"].strip()
        + "\n\nBu video özgün senaryo, yapay zekâ destekli "
          "seslendirme ve görseller kullanılarak hazırlanmıştır."
        + "\n\nPexels kaynakları:\n"
        + "\n".join(unique_credits)
        + "\n\n#KayıpHikayeler #Ünlüler #Fenomenler "
          "#GerçekHikayeler #Gizem"
    )

    print("YouTube bağlantısı hazırlanıyor...")
    api = youtube()

    print("=" * 60)
    print("ANA VİDEO YÜKLENİYOR")
    print("=" * 60)
    upload(
        api,
        long_video,
        data["title"],
        description,
        data["tags"],
        thumbnail_file
    )

    print("=" * 60)
    print("SHORT YÜKLENİYOR")
    print("=" * 60)
    upload(
        api,
        short_video,
        data["short_title"],
        description,
        data["tags"]
    )

    print("=" * 60)
    print("OTOMASYON BAŞARIYLA TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()
