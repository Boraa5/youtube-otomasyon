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
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
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

YOUTUBE_SCOPE = "https://www.googleapis.com/auth/youtube.upload"


# ============================================================
# YARDIMCILAR
# ============================================================

def run(command, inp=None):
    print("$", " ".join(map(str, command)))
    subprocess.run(
        command,
        input=inp,
        text=True,
        check=True
    )


def wc(text):
    return len(
        re.findall(
            r"\b[\wÇĞİÖŞÜçğıöşü'-]+\b",
            str(text)
        )
    )


def duration(file_path):
    result = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
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
        raise RuntimeError(
            "Eksik veya bozuk dosya: " + str(path)
        )


def clean_query(query):
    query = re.sub(
        r"[^A-Za-z0-9\s.'&-]",
        " ",
        str(query)
    )
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
        if n == 1:
            cleaned.append(fallback)
        else:
            cleaned.append(f"{fallback} {n}")
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
        "İnternet Hikâyeleri"
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
                    temperature=0.6,
                    max_output_tokens=8192
                )
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini boş yanıt verdi"
                )

            return json.loads(response.text)

        except Exception as error:
            last_error = str(error)

            if attempt == attempts - 1:
                break

            if is_quota_error(last_error):
                wait = min(
                    300,
                    30 * (2 ** attempt)
                ) + random.randint(0, 10)

                print(
                    f"Gemini kota/sınır hatası. "
                    f"{wait} saniye bekleniyor... "
                    f"Deneme {attempt + 1}/{attempts}"
                )
            else:
                wait = min(
                    60,
                    5 * (attempt + 1)
                )

                print(
                    f"Gemini hata: {last_error}. "
                    f"{wait} saniye sonra tekrar denenecek..."
                )

            time.sleep(wait)

    raise RuntimeError(
        "Gemini içerik üretilemedi: "
        + str(last_error)
    )


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
        "title": {
            "type": "STRING"
        },
        "description": {
            "type": "STRING"
        },
        "tags": {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            }
        },
        "narration": {
            "type": "STRING"
        },
        "short_title": {
            "type": "STRING"
        },
        "short_narration": {
            "type": "STRING"
        },
        "scene_queries": {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            }
        },
        "short_queries": {
            "type": "ARRAY",
            "items": {
                "type": "STRING"
            }
        },
        "thumbnail_query": {
            "type": "STRING"
        }
    },
    "required": [
        "title",
        "description",
        "tags",
        "narration",
        "short_title",
        "short_narration",
        "scene_queries",
        "short_queries",
        "thumbnail_query"
    ]
}


def _continuation_json(prompt, field):
    schema = {
        "type": "OBJECT",
        "properties": {
            field: {
                "type": "STRING"
            }
        },
        "required": [field]
    }

    data = gemini_json(prompt, schema)

    text = str(
        data.get(field, "")
    ).strip()

    if not text:
        raise RuntimeError(
            "Gemini devam metni boş döndü"
        )

    return text


def expand_main_story(narration, title):
    text = str(narration).strip()

    for round_no in range(1, 4):
        words = wc(text)

        if words >= MAIN_MIN_WORDS:
            return text

        remaining = MAIN_MIN_WORDS - words
        target_add = max(
            remaining + 80,
            280
        )

        print(
            f"Ana senaryo genişletme "
            f"{round_no}/3: "
            f"{words} kelime, "
            f"en az {remaining} kelime eksik..."
        )

        prompt = f"""
Aşağıdaki Türkçe YouTube senaryosu kısa kaldı.

AYNI gerçek hikâyeyi, AYNI kişiyi ve AYNI olay zincirini koru.

Başlık:
{title}

Sadece mevcut metnin BİTTİĞİ YERDEN DEVAMINI yaz.

KESİNLİKLE girişten yeniden başlama.
KESİNLİKLE önceki cümleleri tekrar etme.
KESİNLİKLE yeni konu veya yeni kişi seçme.
Uydurma bilgi, tarih, alıntı, suçlama veya olay ekleme.

Doğrulanabilir kamuya açık bağlamı,
olayların sırasını,
dönüm noktalarını
ve sonucun anlamını
doğal biçimde ayrıntılandır.

BU TURDA EN AZ {target_add} kelimelik
YENİ devam üret.

Başlık, numara, "devam" sözü
veya açıklama yazma.

Mevcut senaryo:

---
{text}
---

Sadece JSON döndür.
"""

        addition = _continuation_json(
            prompt,
            "narration"
        ).strip()

        if addition in text or wc(addition) < 40:
            print(
                "Geçersiz/kısa devam geldi, "
                "yeniden deneniyor..."
            )
            continue

        text = (
            text.rstrip()
            + "\n\n"
            + addition
        )

    return text.strip()


def expand_short_story(
    short_text,
    title,
    main_narration
):
    text = str(short_text).strip()

    for round_no in range(1, 3):
        words = wc(text)

        if words >= SHORT_MIN_WORDS:
            return text

        remaining = SHORT_MIN_WORDS - words
        target_add = max(
            remaining + 15,
            45
        )

        print(
            f"Short genişletme "
            f"{round_no}/2: "
            f"{words} kelime, "
            f"en az {remaining} kelime eksik..."
        )

        prompt = f"""
Aşağıdaki YouTube Shorts metni kısa kaldı.

AYNI gerçek olayı ve AYNI kişiyi koru.

Başlık:
{title}

Sadece mevcut Short'un BİTTİĞİ YERDEN DEVAMINI yaz.

Önceki cümleleri tekrar etme.
Yeni olay, yeni kişi veya uydurma bilgi ekleme.

BU TURDA EN AZ {target_add} kelimelik
doğal yeni devam üret.

45-60 saniyelik meraklı belgesel
anlatım tonunu koru.

Mevcut Short:

---
{text}
---

Ana hikâye bağlamı:

---
{main_narration[:9000]}
---

Sadece JSON döndür.
"""

        addition = _continuation_json(
            prompt,
            "short_narration"
        ).strip()

        if addition in text or wc(addition) < 10:
            print(
                "Geçersiz/kısa Short devamı geldi, "
                "yeniden deneniyor..."
            )
            continue

        text = (
            text.rstrip()
            + "\n\n"
            + addition
        )

    return text.strip()


def plan():
    prompt = f"""
"Kayıp Hikâyeler" adlı Türkçe YouTube kanalı için
TEK bir gerçek, belgelenebilir ve güçlü
merak uyandıran hikâye hazırla.

KANALIN KONUSU:

Türkiye'deki tanınmış sanatçılar,
ünlüler,
sosyal medya fenomenleri
ve kamuoyunca bilinen kişilerin
hayatlarında yaşanmış gerçek,
şaşırtıcı,
az bilinen
veya yıllarca merak edilmiş olaylar.

ÖNEMLİ:

- İftira, uydurma skandal veya doğrulanmamış suçlama üretme.
- Tartışmalı iddiaları kesin gerçek gibi anlatma.
- Özel hayatı gereksiz biçimde hedef alma.
- Sadece kamuya açık, doğrulanabilir olayları kullan.
- Aynı "12 yıl kayboldu", "duvarın arkasında bulundu"
  gibi kalıpları kullanma.
- Her videoda farklı olay türü ve farklı başlık yapısı kullan.
- Arkeoloji veya tarih dersi gibi yazma.

ANA VİDEO:

Hedef yaklaşık 7 dakikalık doğal Türkçe seslendirme.

NARRATION 850-1000 kelime olsun ve
KESİNLİKLE 750 kelimenin altına düşme.

İlk cümle doğrudan
en şaşırtıcı ayrıntıyla başlasın.

Selamlama,
kanal tanıtımı,
"bugün sizlere"
veya
"bu videoda"
ifadelerini kullanma.

Gereksiz tekrar ve boş uzatma yapma.

SHORT:

Ana videodaki aynı gerçek olaya bağlı olsun.

Kopya özet olmasın;
en çarpıcı noktayı anlatsın.

İlk 3 saniyede merak oluştursun.

90-150 kelime arasında,
yaklaşık 45-60 saniyelik
doğal konuşma temposunda olsun.

BAŞLIKLAR:

Her seferinde farklı yapı kullan.

Sürekli
"X Yıl Boyunca..."
veya
"Yıllarca..."
kalıbını kullanma.

Yalan clickbait yapma.

PEXELS:

Ana video için tam 12
İngilizce görsel sorgusu.

Short için tam 6
İngilizce görsel sorgusu.

Sorgular 2-6 kelime,
genel ve gerçekçi olsun.

THUMBNAIL:

Tek İngilizce,
güçlü,
genel
ve Pexels'te aranabilir sorgu.

Sadece geçerli JSON döndür.
"""

    data = gemini_json(
        prompt,
        PLAN_SCHEMA
    )

    required_text = [
        "title",
        "description",
        "narration",
        "short_title",
        "short_narration",
        "thumbnail_query"
    ]

    for field in required_text:
        if not str(
            data.get(field, "")
        ).strip():
            raise RuntimeError(
                "Boş alan: " + field
            )

    data["title"] = str(
        data["title"]
    ).strip()

    data["description"] = str(
        data["description"]
    ).strip()

    data["narration"] = str(
        data["narration"]
    ).strip()

    data["short_title"] = str(
        data["short_title"]
    ).strip()

    data["short_narration"] = str(
        data["short_narration"]
    ).strip()

    data["thumbnail_query"] = (
        clean_query(
            data["thumbnail_query"]
        )
        or "celebrity documentary"
    )

    data["tags"] = normalize_tags(
        data.get("tags", [])
    )

    data["scene_queries"] = normalize_queries(
        data.get("scene_queries", []),
        12,
        "documentary investigation"
    )

    data["short_queries"] = normalize_queries(
        data.get("short_queries", []),
        6,
        "mysterious documentary"
    )

    main_words = wc(
        data["narration"]
    )

    if main_words < MAIN_MIN_WORDS:
        print(
            f"Ana senaryo {main_words} kelime. "
            "Aynı hikâye tamamlanıyor..."
        )

        data["narration"] = expand_main_story(
            data["narration"],
            data["title"]
        )

    short_words = wc(
        data["short_narration"]
    )

    if short_words < SHORT_MIN_WORDS:
        print(
            f"Short {short_words} kelime. "
            "Aynı olay tamamlanıyor..."
        )

        data["short_narration"] = expand_short_story(
            data["short_narration"],
            data["title"],
            data["narration"]
        )

    main_words = wc(
        data["narration"]
    )

    short_words = wc(
        data["short_narration"]
    )

    if main_words < MAIN_MIN_WORDS:
        raise RuntimeError(
            f"Ana senaryo hedefe ulaşamadı: "
            f"{main_words} kelime."
        )

    if main_words > MAIN_MAX_WORDS:
        print(
            f"Ana senaryo {main_words} kelime; "
            "üst sınır aşıldı ama kabul edildi."
        )

    if short_words < SHORT_MIN_WORDS:
        raise RuntimeError(
            f"Short hedefe ulaşamadı: "
            f"{short_words} kelime."
        )

    if short_words > SHORT_MAX_WORDS:
        print(
            f"Short {short_words} kelime; "
            "üst sınır aşıldı ama kabul edildi."
        )

    print("Plan kabul edildi.")
    print("Ana kelime:", main_words)
    print("Short kelime:", short_words)

    return data


# ============================================================
# PEXELS
# ============================================================

def pexels(query, orientation, used):
    query = clean_query(query) or "documentary"

    response = requests.get(
        "https://api.pexels.com/v1/search",
        headers={
            "Authorization": PEXELS
        },
        params={
            "query": query,
            "orientation": orientation,
            "per_page": 20
        },
        timeout=60
    )

    response.raise_for_status()

    photos = response.json().get(
        "photos",
        []
    )

    if not photos:
        response = requests.get(
            "https://api.pexels.com/v1/search",
            headers={
                "Authorization": PEXELS
            },
            params={
                "query": "documentary people",
                "orientation": orientation,
                "per_page": 20
            },
            timeout=60
        )

        response.raise_for_status()

        photos = response.json().get(
            "photos",
            []
        )

    if not photos:
        raise RuntimeError(
            "Pexels sonucu bulunamadı: "
            + query
        )

    photo = next(
        (
            x for x in photos
            if x["id"] not in used
        ),
        photos[0]
    )

    used.add(photo["id"])

    src = photo["src"]

    if orientation == "portrait":
        image_url = src.get("portrait")
    else:
        image_url = src.get("landscape")

    image_url = (
        image_url
        or src.get("large2x")
        or src.get("original")
    )

    return photo, image_url


def images(
    queries,
    prefix,
    orientation
):
    files = []
    credits = []
    used = set()

    for number, query in enumerate(
        queries,
        1
    ):
        print(
            f"Görsel {number}/{len(queries)}: "
            f"{query}"
        )

        photo, image_url = pexels(
            query,
            orientation,
            used
        )

        file_path = OUT / (
            f"{prefix}_{number:02d}.jpg"
        )

        response = requests.get(
            image_url,
            timeout=90
        )

        response.raise_for_status()

        file_path.write_bytes(
            response.content
        )

        if file_path.stat().st_size < 5000:
            raise RuntimeError(
                "Bozuk Pexels görseli: "
                + str(file_path)
            )

        files.append(file_path)
        credits.append(photo)

    return files, credits


# ============================================================
# TTS
# ============================================================

def tts(text, output):
    text = re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()

    chunks = []

    while len(text) > 2200:
        split = text.rfind(
            " ",
            0,
            2200
        )

        if split < 500:
            split = 2200

        chunks.append(
            text[:split]
        )

        text = text[split:].strip()

    if text:
        chunks.append(text)

    if not chunks:
        raise RuntimeError(
            "Seslendirilecek metin boş"
        )

    wavs = []

    for number, chunk in enumerate(chunks):
        wav = OUT / (
            f"{output.stem}_{number:03d}.wav"
        )

        print(
            f"Ses hazırlanıyor "
            f"{number + 1}/{len(chunks)}"
        )

        run(
            [
                "python",
                "-m",
                "piper",
                "--model",
                str(PIPER),
                "--output_file",
                str(wav),
                "--sentence-silence",
                "0.18",
                "--length-scale",
                "0.88",
                "--noise-scale",
                "0.667",
                "--noise-w",
                "0.8"
            ],
            chunk
        )

        wavs.append(wav)

    concat = OUT / (
        f"{output.stem}_concat.txt"
    )

    concat.write_text(
        "\n".join(
            "file '"
            + str(
                wav.resolve()
            ).replace(
                "'",
                "'\\''"
            )
            + "'"
            for wav in wavs
        ),
        encoding="utf-8"
    )

    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "160k",
            str(output)
        ]
    )


# ============================================================
# VİDEO
# ============================================================

def make_video(
    audio,
    image_files,
    output,
    vertical=False
):
    if not image_files:
        raise RuntimeError(
            "Video için görsel bulunamadı"
        )

    total = duration(audio)
    each = max(
        1.0,
        total / len(image_files)
    )

    slides = OUT / (
        f"{output.stem}_slides.txt"
    )

    lines = []

    for image_file in image_files:
        escaped = str(
            image_file.resolve()
        ).replace(
            "'",
            "'\\''"
        )

        lines.append(
            "file '" + escaped + "'"
        )

        lines.append(
            f"duration {each:.3f}"
        )

    lines.append(
        "file '"
        + str(
            image_files[-1].resolve()
        ).replace(
            "'",
            "'\\''"
        )
        + "'"
    )

    slides.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    if vertical:
        vf = (
            "scale=1080:1920:"
            "force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "format=yuv420p"
        )
    else:
        vf = (
            "scale=1280:720:"
            "force_original_aspect_ratio=increase,"
            "crop=1280:720,"
            "format=yuv420p"
        )

    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(slides),
            "-i",
            str(audio),
            "-vf",
            vf,
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output)
        ]
    )


# ============================================================
# THUMBNAIL
# ============================================================

def thumbnail(source, output):
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vf",
            (
                "scale=1280:720:"
                "force_original_aspect_ratio=increase,"
                "crop=1280:720"
            ),
            "-q:v",
            "7",
            str(output)
        ]
    )

    if output.stat().st_size > 1_900_000:
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(output),
                "-q:v",
                "10",
                str(output)
            ]
        )


# ============================================================
# YOUTUBE
# ============================================================

def youtube():
    refresh_token = os.getenv(
        "YOUTUBE_REFRESH_TOKEN",
        ""
    ).strip()

    client_id = os.getenv(
        "GOOGLE_CLIENT_ID",
        ""
    ).strip()

    client_secret = os.getenv(
        "GOOGLE_CLIENT_SECRET",
        ""
    ).strip()

    if not refresh_token:
        raise RuntimeError(
            "YOUTUBE_REFRESH_TOKEN bulunamadı."
        )

    if not client_id:
        raise RuntimeError(
            "GOOGLE_CLIENT_ID bulunamadı."
        )

    if not client_secret:
        raise RuntimeError(
            "GOOGLE_CLIENT_SECRET bulunamadı."
        )

    credentials = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=(
            "https://oauth2.googleapis.com/token"
        ),
        client_id=client_id,
        client_secret=client_secret,
        scopes=[YOUTUBE_SCOPE]
    )

    print(
        "YouTube bağlantısı hazırlanıyor..."
    )

    try:
        credentials.refresh(Request())

    except RefreshError as error:
        error_text = str(error).lower()

        if "invalid_grant" in error_text:
            raise RuntimeError(
                "YOUTUBE_REFRESH_TOKEN geçersiz, "
                "iptal edilmiş veya süresi dolmuş. "
                "GitHub Secrets içindeki "
                "YOUTUBE_REFRESH_TOKEN değerini "
                "yenilemen gerekiyor."
            ) from error

        raise RuntimeError(
            "YouTube giriş hatası: "
            + str(error)
        ) from error

    print(
        "YouTube bağlantısı hazır."
    )

    return build(
        "youtube",
        "v3",
        credentials=credentials,
        cache_discovery=False
    )


def upload(
    api,
    file_path,
    title,
    description,
    tags,
    thumbnail_file=None
):
    body = {
        "snippet": {
            "title": str(
                title
            ).strip()[:100],
            "description": str(
                description
            ).strip()[:5000],
            "tags": [
                str(tag).strip()
                for tag in tags[:30]
                if str(tag).strip()
            ],
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
    retries = 0
    max_retries = 5

    print(
        "YouTube'a yükleme başladı..."
    )

    while response is None:
        try:
            status, response = request.next_chunk()

            if status:
                percent = int(
                    status.progress() * 100
                )

                print(
                    f"YouTube yükleme: %{percent}"
                )

        except Exception as error:
            retries += 1

            if retries >= max_retries:
                raise RuntimeError(
                    "YouTube yükleme başarısız oldu: "
                    + str(error)
                ) from error

            wait = min(
                60,
                retries * 10
            )

            print(
                f"YouTube yükleme hatası. "
                f"{wait} saniye bekleniyor... "
                f"Deneme {retries}/{max_retries}"
            )

            time.sleep(wait)

    video_id = response.get("id")

    if not video_id:
        raise RuntimeError(
            "YouTube video ID alınamadı."
        )

    print(
        "Video yüklendi: "
        + video_id
    )

    if (
        thumbnail_file
        and thumbnail_file.exists()
    ):
        print(
            "Kapak yükleniyor..."
        )

        try:
            api.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(
                    str(thumbnail_file),
                    mimetype="image/jpeg"
                )
            ).execute()

            print(
                "Kapak yüklendi."
            )

        except Exception as error:
            print(
                "UYARI: Video yüklendi fakat "
                "kapak yüklenemedi: "
                + str(error)
            )

    return video_id


# ============================================================
# ANA PROGRAM
# ============================================================

def main():
    print("=" * 60)
    print(
        "KAYIP HİKÂYELER - "
        "ÜNLÜLER VE FENOMENLER"
    )
    print(
        "Hedef ana video: yaklaşık 7 dakika"
    )
    print(
        "Hedef Short: yaklaşık 45-60 saniye"
    )
    print("=" * 60)

    setup()

    data = plan()

    print("=" * 60)
    print(
        "KONU:",
        data["title"]
    )
    print(
        "Ana kelime:",
        wc(data["narration"])
    )
    print(
        "Short kelime:",
        wc(data["short_narration"])
    )
    print("=" * 60)

    long_audio = OUT / "long.mp3"
    short_audio = OUT / "short.mp3"

    tts(
        data["narration"],
        long_audio
    )

    tts(
        data["short_narration"],
        short_audio
    )

    long_duration = duration(
        long_audio
    )

    short_duration = duration(
        short_audio
    )

    print(
        "Ana video süresi:",
        round(long_duration, 1),
        "saniye"
    )

    print(
        "Short süresi:",
        round(short_duration, 1),
        "saniye"
    )

    if long_duration < 300:
        raise RuntimeError(
            f"Ana video beklenenden kısa: "
            f"{long_duration:.1f} saniye."
        )

    if short_duration < 35:
        raise RuntimeError(
            f"Short beklenenden kısa: "
            f"{short_duration:.1f} saniye."
        )

    print(
        "Ana video görselleri indiriliyor..."
    )

    long_images, long_credits = images(
        data["scene_queries"],
        "long",
        "landscape"
    )

    print(
        "Short görselleri indiriliyor..."
    )

    short_images, short_credits = images(
        data["short_queries"],
        "short",
        "portrait"
    )

    long_video = OUT / "long.mp4"
    short_video = OUT / "short.mp4"

    print(
        "Ana video hazırlanıyor..."
    )

    make_video(
        long_audio,
        long_images,
        long_video
    )

    print(
        "Short hazırlanıyor..."
    )

    make_video(
        short_audio,
        short_images,
        short_video,
        vertical=True
    )

    print(
        "Kapak hazırlanıyor..."
    )

    thumbnail_images, _ = images(
        [data["thumbnail_query"]],
        "thumbnail",
        "landscape"
    )

    thumbnail_file = (
        OUT / "thumbnail.jpg"
    )

    thumbnail(
        thumbnail_images[0],
        thumbnail_file
    )

    credits = []

    for photo in (
        long_credits
        + short_credits
    ):
        photo_url = photo.get(
            "url",
            ""
        )

        photographer = photo.get(
            "photographer",
            "Pexels photographer"
        )

        if photo_url:
            credits.append(
                f"Photo by {photographer} "
                f"on Pexels: {photo_url}"
            )

    unique_credits = list(
        dict.fromkeys(credits)
    )

    description = (
        data["description"].strip()
        + "\n\n"
        + "Bu video özgün senaryo, "
        + "yapay zekâ destekli seslendirme "
        + "ve görseller kullanılarak hazırlanmıştır."
        + "\n\nPexels kaynakları:\n"
        + "\n".join(unique_credits)
        + "\n\n"
        + "#KayıpHikayeler "
        + "#Ünlüler "
        + "#Fenomenler "
        + "#GerçekHikayeler "
        + "#Gizem"
    )

    print(
        "YouTube bağlantısı hazırlanıyor..."
    )

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
    print(
        "OTOMASYON BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
