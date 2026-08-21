import os
import re
import json
import time
import subprocess
import random
from pathlib import Path

import requests
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


# ============================================================
# KAYIP HİKÂYELER - ÜNLÜLER / FENOMENLER / INFLUENCERLAR
# ============================================================

OUT = Path("work")
OUT.mkdir(exist_ok=True)

GEMINI = os.environ["GEMINI_API_KEY"]
PEXELS = os.environ["PEXELS_API_KEY"]

YOUTUBE_REFRESH_TOKEN = os.environ["YOUTUBE_REFRESH_TOKEN"]
GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]

MODEL = "gemini-3.1-flash-lite"

client = genai.Client(api_key=GEMINI)


# ============================================================
# TÜRKÇE SES
# ============================================================

PIPER = OUT / "tr_TR-dfki-medium.onnx"
PIPER_CFG = OUT / "tr_TR-dfki-medium.onnx.json"

PIPER_URL = (
    "https://huggingface.co/rhasspy/piper-voices/"
    "resolve/v1.0.0/tr/tr_TR/dfki/medium/"
)


# ============================================================
# DAHA ÖNCE KULLANILAN KONULAR
# ============================================================

HISTORY_FILE = OUT / "used_topics.json"


def load_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(
            HISTORY_FILE.read_text(encoding="utf-8")
        )

        if isinstance(data, list):
            return data[-50:]

    except Exception:
        pass

    return []


def save_topic(topic):
    history = load_history()

    topic = str(topic).strip()

    if topic and topic not in history:
        history.append(topic)

    history = history[-50:]

    HISTORY_FILE.write_text(
        json.dumps(
            history,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


# ============================================================
# YARDIMCI FONKSİYONLAR
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

    response = requests.get(
        url,
        timeout=180
    )

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

    query = re.sub(
        r"\s+",
        " ",
        query
    ).strip()

    return query


def normalize_queries(
    queries,
    count,
    fallback
):
    cleaned = []

    if isinstance(queries, list):
        for query in queries:
            query = clean_query(query)

            if query and query not in cleaned:
                cleaned.append(query)

    fallback_number = 0

    while len(cleaned) < count:
        fallback_number += 1

        candidate = (
            fallback
            if fallback_number == 1
            else fallback + " " + str(fallback_number)
        )

        if candidate not in cleaned:
            cleaned.append(candidate)

    return cleaned[:count]


def normalize_tags(tags):
    cleaned = []

    if isinstance(tags, list):
        for tag in tags:
            tag = str(tag).strip()

            if tag and tag not in cleaned:
                cleaned.append(tag)

    fallback_tags = [
        "Kayıp Hikâyeler",
        "Ünlüler",
        "Fenomenler",
        "Influencer",
        "Gizem",
        "Gerçek Hikâyeler",
        "Türkiye"
    ]

    for tag in fallback_tags:
        if len(cleaned) >= 15:
            break

        if tag not in cleaned:
            cleaned.append(tag)

    return cleaned[:15]


# ============================================================
# TÜRKÇE SES MODELİ
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
# GEMINI
# ============================================================

def gemini(prompt):

    last_error = None

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "OBJECT",
                        "properties": {

                            "topic": {
                                "type": "STRING"
                            },

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
                            "topic",
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
                    },
                    temperature=0.9
                )
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini boş cevap verdi"
                )

            return json.loads(response.text)

        except Exception as error:

            last_error = str(error)

            print(
                "Gemini hata:",
                last_error
            )

            # =================================================
            # KOTA BİTMİŞSE BOŞUNA TEKRAR TEKRAR İSTEK ATMA
            # =================================================

            if (
                "429" in last_error
                or "RESOURCE_EXHAUSTED" in last_error
                or "quota" in last_error.lower()
            ):
                raise RuntimeError(
                    "GEMINI API KOTASI BİTMİŞ. "
                    "Kodda bekleyerek çözülecek bir hata değil. "
                    "Google AI Studio / Gemini API kota veya "
                    "faturalandırma tarafı düzelmeden yeni istek "
                    "oluşturulamaz. "
                    "İşlem burada güvenli şekilde durduruldu."
                )

            if attempt < 2:
                wait = 8 * (attempt + 1)

                print(
                    f"Tekrar denenecek: "
                    f"{wait} saniye..."
                )

                time.sleep(wait)

    raise RuntimeError(
        "Gemini içerik üretilemedi: "
        + str(last_error)
    )


# ============================================================
# HİKÂYE PLANI
# ============================================================

def plan():

    used_topics = load_history()

    used_text = "\n".join(
        "- " + item
        for item in used_topics[-30:]
    )

    if not used_text:
        used_text = "Henüz kullanılmadı."

    prompt = f"""
"Kayıp Hikâyeler" adlı Türkçe YouTube kanalı için TEK bir güçlü,
gerçek ve merak uyandırıcı hikâye hazırla.

KANALIN YENİ KONUSU:

Türkiye'deki ünlüler, sosyal medya fenomenleri, influencerlar,
sanatçılar ve kamuoyunda geniş şekilde tanınan kişilerle ilgili
gerçek olaylar.

KONU TİPLERİ:

- Yıllarca saklı kalan şaşırtıcı gerçekler
- Kamuoyunun bilmediği sonradan ortaya çıkan olaylar
- Kayboluşlar ve ortadan kaybolma dönemleri
- Esrarengiz ayrılıklar
- Beklenmedik kariyer dönüşleri
- Herkesin yanlış anladığı olayların gerçek yüzü
- Yıllar sonra açıklanan sırlar
- Eski görüntü, röportaj veya belgelerle yeniden gündeme gelen olaylar
- Bir gecede değişen hayatlar
- Büyük tartışmaların perde arkası
- Sosyal medyada herkesin konuştuğu ama ayrıntısını bilmediği gerçek olaylar

ÇOK ÖNEMLİ:

Kişiler hakkında uydurma suçlama, iftira, dedikodu veya doğrulanmamış
iddia üretme.

Özel hayatla ilgili doğrulanmamış bilgi yazma.

Bir kişinin suç işlediğini, gizli ilişkisinin olduğunu, hastalığı
olduğunu veya başka hassas bir iddiayı kanıt yoksa gerçekmiş gibi anlatma.

Sadece kamuya açık, yaygın biçimde belgelenmiş veya kişinin kendisinin
açıkladığı gerçek olaylardan yola çık.

Belirsiz noktaları kesin bilgi gibi sunma.

AMA ANLATIM SIKICI OLMASIN.

İzleyici ilk saniyede:

"Ne olmuş olabilir?"
"Gerçekten böyle mi oldu?"
"Peki sonra ne ortaya çıktı?"

diye merak etmeli.

============================================================

DAHA ÖNCE KULLANILAN KONULAR:

{used_text}

BU KONULARI TEKRAR SEÇME.

Ayrıca aynı anlatım kalıbını tekrar tekrar kullanma.

Özellikle her videoyu:

"12 yıl boyunca kayboldu"
"Yıllarca bulunamadı"
"Polis onu aradı"

gibi aynı kalıpla başlatma.

Her yeni hikâyenin kendi özgün giriş açısı olsun.

============================================================

ANA VİDEO:

HEDEF SÜRE:

Yaklaşık 7 dakika.

Türkçe seslendirme hızını düşünerek narration yaklaşık
750 ile 1050 kelime arasında olsun.

750 kelimenin altına düşme.

1050 kelimeyi gereksiz tekrar için aşma.

Hikâye doğal, akıcı ve güçlü olsun.

YAPI:

1. İlk cümlede çok güçlü merak
2. Olayın en şaşırtıcı kısmına kısa giriş
3. Kişi veya olay hakkında gerekli arka plan
4. Herkesin gördüğü fakat çoğunun bilmediği önemli ayrıntı
5. Olayın gelişmesi
6. Beklenmedik dönüm noktası
7. Kamuoyunda oluşan soru veya yanlış anlaşılma
8. Gerçekte ortaya çıkan belgelenmiş bilgiler
9. En güçlü sonuç
10. Kısa ve etkileyici kapanış

İLK 15 SANİYE ÇOK ÖNEMLİ.

Selamlama yapma.

"Kanalımıza hoş geldiniz" yazma.

"Bugün sizlere" diye başlama.

"Bu videoda" diye başlama.

İlk cümle doğrudan hikâyenin en güçlü merak noktasına girsin.

Aynı bilgiyi tekrar tekrar anlatma.

Her 20-40 saniyelik anlatım hissinde yeni bir bilgi, soru veya
dönüm noktası oluştur.

============================================================

BAŞLIK:

Başlık 100 karakterden kısa olsun.

Kısa.

Güçlü.

Merak uyandırıcı.

Ama yalan clickbait olmasın.

Her videoda aynı kelimeleri kullanma.

Sürekli:

"Yıllarca"
"Kayboldu"
"Sonunda bulundu"
"Polis"

kelimelerine bağlı kalma.

Konuya göre farklı başlık açıları kullan.

Örnek başlık mantıkları:

- Herkes bunu konuştu ama gerçeği çok sonra öğrenildi
- Bir gecede ortadan kayboldu: Sonra ortaya çıkanlar
- Kameraların önünde yaşandı ama kimse o detayı fark etmedi
- Her şey tek bir açıklamayla değişti
- Yıllar sonra gelen itiraf herkesin bildiği hikâyeyi değiştirdi

Bunları aynen kullanma.

Yeni ve özgün başlık üret.

============================================================

SHORT:

Short ana videodaki AYNI gerçek olaya bağlı olsun.

Ama ana videonun özeti gibi bütün hikâyeyi anlatmasın.

Hikâyenin en çarpıcı anını seç.

İlk 2 saniyede güçlü merak oluştur.

Short yaklaşık 45 ile 60 saniye arasında olsun.

Short narration yaklaşık 100 ile 150 kelime arasında olsun.

25-30 kelimelik kısa metin üretme.

Short sonunda ana olayın tamamını merak ettirecek güçlü bir cümle olsun.

============================================================

PEXELS GÖRSELLERİ:

Ana video için TAM OLARAK 18 İngilizce sorgu üret.

Her sorgu:

- İngilizce
- 2 ile 7 kelime
- Gerçekçi
- Pexels'te sonuç bulabilecek kadar genel
- Olayın atmosferine uygun
- Birbirinden mümkün olduğunca farklı

Örnek türleri:

television studio lights
celebrity red carpet
smartphone social media
crowded press conference
empty backstage hallway
old newspaper archive
camera flash photographers
night city apartment

Bunlar sadece örnek.

Aynısını kullanma.

Short için TAM OLARAK 8 İngilizce sorgu üret.

============================================================

THUMBNAIL:

thumbnail_query alanına Pexels için TEK İngilizce sorgu yaz.

Kapakta güçlü ve merak uyandırıcı görüntü bulunabilecek kadar genel olsun.

============================================================

AÇIKLAMA:

description alanına kısa, doğal bir YouTube açıklaması yaz.

============================================================

ETİKETLER:

5 ile 15 arasında alakalı etiket üret.

============================================================

NARRATION:

- Başlık yazma
- Bölüm numarası yazma
- Kaynak listesi yazma
- "Kaynak:" yazma
- JSON yazma

Sadece doğal seslendirme metni olsun.

============================================================

SADECE GEÇERLİ JSON DÖNDÜR.

Zorunlu alanlar:

topic
title
description
tags
narration
short_title
short_narration
scene_queries
short_queries
thumbnail_query

JSON dışında hiçbir şey yazma.
"""

    last_error = "İlk deneme"

    for attempt in range(3):

        try:

            print(
                f"Hikâye hazırlanıyor... "
                f"{attempt + 1}/3"
            )

            data = gemini(prompt)

            required = [
                "topic",
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

            for field in required:
                if field not in data:
                    raise ValueError(
                        f"Eksik alan: {field}"
                    )

            for field in [
                "topic",
                "title",
                "description",
                "narration",
                "short_title",
                "short_narration",
                "thumbnail_query"
            ]:
                if not str(data[field]).strip():
                    raise ValueError(
                        f"Boş alan: {field}"
                    )

            data["topic"] = str(
                data["topic"]
            ).strip()

            data["title"] = str(
                data["title"]
            ).strip()[:100]

            data["description"] = str(
                data["description"]
            ).strip()

            data["narration"] = str(
                data["narration"]
            ).strip()

            data["short_title"] = str(
                data["short_title"]
            ).strip()[:100]

            data["short_narration"] = str(
                data["short_narration"]
            ).strip()

            data["thumbnail_query"] = clean_query(
                data["thumbnail_query"]
            )

            if not data["thumbnail_query"]:
                data["thumbnail_query"] = (
                    "celebrity mysterious portrait"
                )

            main_words = wc(
                data["narration"]
            )

            short_words = wc(
                data["short_narration"]
            )

            if main_words < 750:
                raise ValueError(
                    f"Ana video senaryosu kısa: "
                    f"{main_words} kelime. "
                    f"Hedef en az 750 kelime."
                )

            if main_words > 1100:
                raise ValueError(
                    f"Ana video senaryosu gereğinden uzun: "
                    f"{main_words} kelime."
                )

            if short_words < 100:
                raise ValueError(
                    f"Short senaryosu kısa: "
                    f"{short_words} kelime."
                )

            if short_words > 170:
                raise ValueError(
                    f"Short senaryosu fazla uzun: "
                    f"{short_words} kelime."
                )

            data["scene_queries"] = normalize_queries(
                data["scene_queries"],
                18,
                "celebrity documentary"
            )

            data["short_queries"] = normalize_queries(
                data["short_queries"],
                8,
                "celebrity social media"
            )

            data["tags"] = normalize_tags(
                data["tags"]
            )

            print("Plan kabul edildi.")
            print(
                "Ana kelime:",
                main_words
            )
            print(
                "Short kelime:",
                short_words
            )

            save_topic(
                data["topic"]
            )

            return data

        except Exception as error:

            last_error = str(error)

            print(
                f"Plan kontrolü "
                f"{attempt + 1}/3: "
                f"{last_error}"
            )

            if (
                "GEMINI API KOTASI BİTMİŞ"
                in last_error
            ):
                raise

            if attempt < 2:
                time.sleep(5)

    raise RuntimeError(
        "Geçerli plan üretilemedi. Son hata: "
        + last_error
    )


# ============================================================
# PEXELS
# ============================================================

def pexels(query, orientation, used):

    query = clean_query(query)

    if not query:
        query = "documentary"

    response = requests.get(
        "https://api.pexels.com/v1/search",
        headers={
            "Authorization": PEXELS
        },
        params={
            "query": query,
            "orientation": orientation,
            "per_page": 30
        },
        timeout=60
    )

    response.raise_for_status()

    photos = response.json().get(
        "photos",
        []
    )

    if not photos:

        fallback = (
            "documentary portrait"
            if orientation == "portrait"
            else "documentary mystery"
        )

        response = requests.get(
            "https://api.pexels.com/v1/search",
            headers={
                "Authorization": PEXELS
            },
            params={
                "query": fallback,
                "orientation": orientation,
                "per_page": 30
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

    unused = [
        item
        for item in photos
        if item["id"] not in used
    ]

    if unused:
        photo = random.choice(unused)
    else:
        photo = random.choice(photos)

    used.add(photo["id"])

    src = photo["src"]

    if orientation == "portrait":
        image_url = (
            src.get("portrait")
            or src.get("large2x")
            or src.get("original")
        )
    else:
        image_url = (
            src.get("landscape")
            or src.get("large2x")
            or src.get("original")
        )

    return photo, image_url


def images(queries, prefix, orientation):

    files = []
    credits = []
    used = set()

    for number, query in enumerate(
        queries,
        1
    ):

        print(
            f"Görsel {number}/"
            f"{len(queries)}: "
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

    lines = []

    for wav in wavs:

        escaped = str(
            wav.resolve()
        ).replace(
            "'",
            "'\\''"
        )

        lines.append(
            "file '" + escaped + "'"
        )

    concat.write_text(
        "\n".join(lines),
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

    total_duration = duration(audio)

    each_duration = (
        total_duration
        / len(image_files)
    )

    if each_duration < 1:
        each_duration = 1

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
            f"duration {each_duration:.3f}"
        )

    last_image = str(
        image_files[-1].resolve()
    ).replace(
        "'",
        "'\\''"
    )

    lines.append(
        "file '" + last_image + "'"
    )

    slides.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    if vertical:

        video_filter = (
            "scale=1080:1920:"
            "force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "format=yuv420p"
        )

    else:

        video_filter = (
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
            video_filter,
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
# YOUTUBE
# ============================================================

def youtube():

    credentials = Credentials(
        token=None,
        refresh_token=YOUTUBE_REFRESH_TOKEN,
        token_uri=(
            "https://oauth2.googleapis.com/token"
        ),
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=[
            "https://www.googleapis.com/"
            "auth/youtube.upload"
        ]
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
    thumbnail=None
):

    body = {
        "snippet": {
            "title": str(title)[:100],
            "description": str(description)[:5000],
            "tags": [
                str(tag)
                for tag in tags[:30]
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

    while response is None:

        status, response = request.next_chunk()

        if status:

            print(
                "YouTube "
                + str(
                    int(
                        status.progress() * 100
                    )
                )
                + "%"
            )

    video_id = response["id"]

    if (
        thumbnail
        and thumbnail.exists()
    ):

        api.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(
                str(thumbnail),
                mimetype="image/jpeg"
            )
        ).execute()

    print(
        "YouTube'a yüklendi:",
        video_id
    )

    return video_id


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
# ANA PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("KAYIP HİKÂYELER - ÜNLÜLER VE FENOMENLER")
    print("=" * 60)

    print(
        "Hedef ana video: yaklaşık 7 dakika"
    )

    print(
        "Short: yaklaşık 45-60 saniye"
    )

    setup()

    data = plan()

    print("=" * 60)
    print("KONU:")
    print(data["topic"])

    print("=" * 60)
    print("ANA BAŞLIK:")
    print(data["title"])

    print("=" * 60)
    print(
        "Ana kelime:",
        wc(data["narration"])
    )

    print(
        "Short kelime:",
        wc(data["short_narration"])
    )

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

    # ========================================================
    # ANA VİDEO SÜRE KONTROLÜ
    # ========================================================

    if long_duration < 360:

        raise RuntimeError(
            "Ana video 6 dakikadan kısa çıktı: "
            + f"{long_duration:.1f} saniye. "
            + "YouTube videosu oluşturulmadan işlem durduruldu."
        )

    if long_duration > 600:

        raise RuntimeError(
            "Ana video 10 dakikadan uzun çıktı: "
            + f"{long_duration:.1f} saniye. "
            + "YouTube videosu oluşturulmadan işlem durduruldu."
        )

    # ========================================================
    # SHORT SÜRE KONTROLÜ
    # ========================================================

    if short_duration < 35:

        raise RuntimeError(
            "Short olağan dışı kısa çıktı: "
            + f"{short_duration:.1f} saniye."
        )

    if short_duration > 75:

        raise RuntimeError(
            "Short olağan dışı uzun çıktı: "
            + f"{short_duration:.1f} saniye."
        )

    print("=" * 60)
    print("ANA VİDEO GÖRSELLERİ İNDİRİLİYOR")
    print("=" * 60)

    long_images, long_credits = images(
        data["scene_queries"],
        "long",
        "landscape"
    )

    print("=" * 60)
    print("SHORT GÖRSELLERİ İNDİRİLİYOR")
    print("=" * 60)

    short_images, short_credits = images(
        data["short_queries"],
        "short",
        "portrait"
    )

    long_video = OUT / "long.mp4"
    short_video = OUT / "short.mp4"

    print("=" * 60)
    print("ANA VİDEO HAZIRLANIYOR")
    print("=" * 60)

    make_video(
        long_audio,
        long_images,
        long_video
    )

    print("=" * 60)
    print("SHORT HAZIRLANIYOR")
    print("=" * 60)

    make_video(
        short_audio,
        short_images,
        short_video,
        vertical=True
    )

    print("=" * 60)
    print("KAPAK HAZIRLANIYOR")
    print("=" * 60)

    thumbnail_images, _ = images(
        [data["thumbnail_query"]],
        "thumbnail",
        "landscape"
    )

    thumbnail_file = OUT / "thumbnail.jpg"

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
        str(data["description"]).strip()
        + "\n\n"
        + "Bu video kamuya açık bilgilerden "
        + "yararlanılarak hazırlanmış özgün anlatım içerir."
        + "\n\n"
        + "Yapay zekâ destekli seslendirme ve "
        + "görsel araçlar kullanılmıştır."
        + "\n\n"
        + "Pexels kaynakları:\n"
        + "\n".join(unique_credits)
        + "\n\n"
        + "#KayıpHikayeler #Ünlüler "
        + "#Fenomenler #GerçekHikayeler "
        + "#Gizem"
    )

    print("=" * 60)
    print("YOUTUBE BAĞLANTISI HAZIRLANIYOR")
    print("=" * 60)

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
