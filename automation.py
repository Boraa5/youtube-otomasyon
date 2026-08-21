import os
import re
import json
import time
import subprocess
from pathlib import Path

import requests
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


# ============================================================
# KAYIP HİKAYELER - YENİ OTOMASYON
# NİŞ:
# Gerçek, inanılmaz ve merak uyandıran kaybolma / bulunma /
# yıllar sonra ortaya çıkan / gizli kalmış gerçek hikâyeler
# ============================================================


OUT = Path("work")
OUT.mkdir(exist_ok=True)

GEMINI = os.environ["GEMINI_API_KEY"]
PEXELS = os.environ["PEXELS_API_KEY"]

MODEL = "gemini-3.1-flash-lite"

client = genai.Client(api_key=GEMINI)


PIPER = OUT / "tr_TR-dfki-medium.onnx"
PIPER_CFG = OUT / "tr_TR-dfki-medium.onnx.json"

PIPER_URL = (
    "https://huggingface.co/rhasspy/piper-voices/"
    "resolve/v1.0.0/tr/tr_TR/dfki/medium/"
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
                    },
                    temperature=0.8
                )
            )

            return json.loads(response.text)

        except Exception as error:
            last_error = str(error)

            print(
                "Gemini hata:",
                last_error
            )

            if (
                "429" in last_error
                or "RESOURCE_EXHAUSTED" in last_error
            ):
                wait = 20 * (attempt + 1)

                print(
                    f"Gemini sınırı. "
                    f"{wait} saniye bekleniyor..."
                )

                time.sleep(wait)

            else:
                if attempt < 2:
                    time.sleep(5)

    raise RuntimeError(
        "Gemini içerik üretilemedi: "
        + str(last_error)
    )


# ============================================================
# SORGU DÜZENLEME
# ============================================================


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

            if query:
                cleaned.append(query)

    cleaned = cleaned[:count]

    fallback_number = 0

    while len(cleaned) < count:
        fallback_number += 1

        if fallback_number == 1:
            cleaned.append(fallback)
        else:
            cleaned.append(
                fallback + " " + str(fallback_number)
            )

    return cleaned


def normalize_tags(tags):
    cleaned = []

    if isinstance(tags, list):
        for tag in tags:
            tag = str(tag).strip()

            if tag and tag not in cleaned:
                cleaned.append(tag)

    fallback_tags = [
        "Kayıp Hikâyeler",
        "Gerçek Hikâyeler",
        "Gizem",
        "Belgesel",
        "İnanılmaz Hikâyeler",
        "Gizemli Olaylar"
    ]

    for tag in fallback_tags:
        if len(cleaned) >= 5:
            break

        if tag not in cleaned:
            cleaned.append(tag)

    return cleaned[:15]


# ============================================================
# HİKAYE PLANI
# ============================================================


def plan():

    prompt = r"""
"Kayıp Hikâyeler" adlı Türkçe YouTube kanalı için TEK bir gerçek
hikâye hazırla.

KANALIN YENİ NİŞİ:

Gerçek hayatta yaşanmış, inanılmaz ve güçlü merak uyandıran
hikâyeler.

Özellikle şu tür konulara öncelik ver:

- Yıllarca kayıp kalan insanların bulunması
- Polis veya aile tarafından yıllarca aranan kişilerin ortaya çıkması
- Herkesin gözünün önünde olup yıllarca fark edilmeyen gerçekler
- Kayıp bir kişinin veya nesnenin beklenmedik şekilde bulunması
- Yıllar sonra çözülen gerçek gizemler
- Gizli kalmış gerçek olayların sonradan ortaya çıkması
- Terk edilmiş bir yerde bulunan önemli bir gerçek
- Unutulmuş, kaybolmuş veya saklanmış bir şeyin ortaya çıkması
- Gerçek ve belgelenmiş inanılmaz olaylar

SADECE ARKEOLOJİ VE TARİH KANALI GİBİ DAVRANMA.

Ana amaç:
İzleyici ilk saniyede "Ne olmuş olabilir?" diye merak etmeli ve
hikâyenin sonunu öğrenmek için videoda kalmalı.

Hikâye GERÇEK ve doğrulanabilir bir olay olmalı.

Uydurma kişi, tarih, sayı, yer, kanıt veya olay kullanma.
Belirsiz bilgileri kesin gerçek gibi anlatma.
Eğer bir olayın bazı ayrıntıları tartışmalıysa bunu dengeli şekilde
belirt.

ANA VİDEO ANLATIMI:

Hikâyeyi DOĞAL uzunlukta yaz.

Kesin olarak 5, 7, 8 veya 9 dakika doldurmaya çalışma.

Kesin bir kelime sayısına ulaşmak için hikâyeyi gereksiz uzatma.

Olay 3-4 dakikada güçlü biçimde anlatılabiliyorsa öyle bırak.
Olay daha büyük ve detaylıysa doğal olarak daha uzun anlat.

Önemli olan süre değil, hikâyenin güçlü olmasıdır.

Ana anlatım:
- Olayı eksik bırakacak kadar kısa olmasın
- Gereksiz tekrarlarla şişirilmesin
- Doğal olarak ihtiyaç duyduğu kadar uzun olsun

ÇOK ÖNEMLİ BAŞLANGIÇ:

İlk cümle doğrudan olayın en şaşırtıcı veya gizemli kısmıyla başlasın.

Selamlama yapma.
Kanal tanıtımı yapma.
"Bugün sizlere" yazma.
"Bu videoda" diye başlama.

İLK 5 SANİYE:
Şok veya çok güçlü merak.

İLK 15 SANİYE:
İzleyicinin aklında büyük soru oluşmalı.

Örnek mantık:

"Polis bu evi defalarca aradı. Ama kayıp kişi 12 yıl boyunca
aslında aradıkları yere çok yakındı."

Bu sadece anlatım mantığı örneğidir.
Aynı cümleyi veya aynı olayı kullanma.

Yapı:

1. Çok güçlü ve merak uyandıran açılış
2. Olayın kısa arka planı
3. Kişinin veya olayın kaybolması / gizemin başlaması
4. Arama veya araştırma süreci
5. İnsanların gözden kaçırdığı önemli ayrıntılar
6. Hikâyenin beklenmedik dönüm noktası
7. En güçlü kanıtlar ve gerçekler
8. Sonucun nasıl ortaya çıktığı
9. Kısa, etkileyici ve dengeli kapanış

Anlatım doğal Türkçe konuşma diliyle olsun.
Gereksiz tekrar yapma.
Aynı bilgiyi farklı cümlelerle tekrar tekrar anlatma.
İzleyiciyi tutmak için hikâye boyunca yeni bilgi ve yeni soru ver.

SHORT:

Short, ana videodaki AYNI GERÇEK HİKÂYEYE bağlı olsun.

Ama ana anlatımın küçük bir kopyası olmasın.

Short doğal uzunlukta olsun.
Süreyi doldurmak için gereksiz cümle ekleme.

Short ilk cümlede doğrudan en güçlü merakı oluştursun.

İlk 3 saniye çok önemlidir.

Short mantığı:

"Polis yıllarca onu aradı...
Ama gerçek ortaya çıktığında cevap aslında hiç kimsenin beklemediği
bir yerdeydi."

Bu sadece mantık örneğidir.
Aynı cümleyi kullanma.

Short, hikâyenin en çarpıcı anını anlatsın ve izleyicide ana
hikâyenin tamamını merak etme isteği oluştursun.

BAŞLIK:

Ana video başlığı kısa, güçlü ve merak uyandırıcı olsun.

Clickbait olabilir ama yalan söyleme.

Başlıkta mümkünse şu duygulardan biri olsun:

- Kayboldu
- Yıllarca bulunamadı
- Herkesin gözünün önündeydi
- Yıllar sonra ortaya çıktı
- Polis onu bulamadı
- Sonunda gerçek ortaya çıktı

Ancak her videoda aynı kalıbı kullanma.

THUMBNAIL:

Kapak için olayın en güçlü görsel unsurunu tarif eden tek bir
İngilizce Pexels sorgusu üret.

PEXELS GÖRSEL SORGULARI:

Ana video için TAM OLARAK 12 İngilizce sorgu üret.

Short için TAM OLARAK 6 İngilizce sorgu üret.

Her sorgu:
- İngilizce olmalı
- Gerçekçi olmalı
- Pexels'te sonuç bulabilecek kadar genel olmalı
- Olayın atmosferine uygun olmalı
- Mümkünse 2 ile 6 kelime arasında olmalı

Karanlık atmosfer gerekiyorsa uygun sorgular kullan.
Ancak aynı görsel fikrini sürekli tekrar etme.

Örnek sorgu türleri:

empty house night
police investigation
missing person poster
old family photograph
forest search
locked basement door

Bunlar sadece örnektir.
Hikâyeye göre yeni sorgular üret.

AÇIKLAMA:

description alanına doğal ve kısa bir YouTube açıklaması yaz.

ETİKETLER:

tags alanında en az 5, en fazla 15 alakalı etiket olsun.

NARRATION içinde:
- Başlık yazma
- Kaynak listesi yazma
- JSON yazma
- Bölüm numarası yazma

SADECE GEÇERLİ JSON ÜRET.

Zorunlu alanlar:

title
description
tags
narration
short_title
short_narration
scene_queries
short_queries
thumbnail_query

KONTROL:

- Hiçbir alan boş olmayacak.
- narration doğal ve yeterli uzunlukta olacak.
- Kesin kelime sınırı nedeniyle hikâye reddedilmeyecek.
- short_narration doğal ve güçlü olacak.
- scene_queries tercihen 12 adet olacak.
- short_queries tercihen 6 adet olacak.
- thumbnail_query boş olmayacak.
- tags en az 5 adet olacak.
- Short ve ana video aynı gerçek hikâyeye bağlı olacak.
- Hikâye arkeoloji dersi veya sıradan tarih anlatımı gibi olmayacak.
- Hikâye gerçek, inanılmaz ve merak uyandırıcı olacak.

Sadece JSON döndür.
JSON dışında hiçbir açıklama yazma.
"""

    last_error = "İlk deneme"

    for attempt in range(5):
        try:
            print(
                f"Hikâye hazırlanıyor... "
                f"{attempt + 1}/5"
            )

            data = gemini(prompt)

            required = [
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
                "title",
                "description",
                "narration",
                "short_title",
                "short_narration",
                "thumbnail_query"
            ]:
                if not str(
                    data[field]
                ).strip():
                    raise ValueError(
                        f"Boş alan: {field}"
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

            data["thumbnail_query"] = clean_query(
                data["thumbnail_query"]
            )

            if not data["thumbnail_query"]:
                data["thumbnail_query"] = (
                    "mysterious investigation"
                )

            # Ana hikâye artık katı bir kelime sınırı
            # yüzünden reddedilmiyor.
            main_words = wc(
                data["narration"]
            )

            short_words = wc(
                data["short_narration"]
            )

            # Sadece gerçekten eksik / bozuk üretimleri reddet.
            # 165, 278 gibi doğal uzunluklar kesinlikle kabul edilir.
            if main_words < 120:
                raise ValueError(
                    f"Ana anlatım fazla kısa: "
                    f"{main_words} kelime"
                )

            if short_words < 25:
                raise ValueError(
                    f"Short fazla kısa: "
                    f"{short_words} kelime"
                )

            data["scene_queries"] = normalize_queries(
                data["scene_queries"],
                12,
                "mysterious investigation"
            )

            data["short_queries"] = normalize_queries(
                data["short_queries"],
                6,
                "mysterious discovery"
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

            return data

        except Exception as error:
            last_error = str(error)

            print(
                f"Plan kontrolü "
                f"{attempt + 1}/5: "
                f"{last_error}"
            )

            if attempt < 4:
                time.sleep(3)

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
            item
            for item in photos
            if item["id"] not in used
        ),
        photos[0]
    )

    used.add(photo["id"])

    src = photo["src"]

    if orientation == "portrait":
        image_url = src.get("portrait")
    else:
        image_url = src.get("landscape")

    if not image_url:
        image_url = (
            src.get("large2x")
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

    for number, chunk in enumerate(
        chunks
    ):
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
        refresh_token=os.environ[
            "YOUTUBE_REFRESH_TOKEN"
        ],
        token_uri=(
            "https://oauth2.googleapis.com/token"
        ),
        client_id=os.environ[
            "GOOGLE_CLIENT_ID"
        ],
        client_secret=os.environ[
            "GOOGLE_CLIENT_SECRET"
        ],
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
            "categoryId": "27"
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
    print("KAYIP HİKAYELER - YENİ OTOMASYON")
    print("=" * 60)

    print(
        "Niş: Gerçek, inanılmaz ve "
        "merak uyandıran hikâyeler"
    )

    setup()

    data = plan()

    print("=" * 60)
    print("KONU:")
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

    # Süre artık katı şekilde 5-9 dakikaya
    # veya 180 saniyeye zorlanmıyor.
    # Hikâye doğal uzunluğunda kabul edilir.
    if long_duration < 30:
        raise RuntimeError(
            "Ana video olağan dışı kısa: "
            + f"{long_duration:.1f} saniye"
        )

    if short_duration < 10:
        raise RuntimeError(
            "Short olağan dışı kısa: "
            + f"{short_duration:.1f} saniye"
        )

    print("Ana video görselleri indiriliyor...")

    long_images, long_credits = images(
        data["scene_queries"],
        "long",
        "landscape"
    )

    print("Short görselleri indiriliyor...")

    short_images, short_credits = images(
        data["short_queries"],
        "short",
        "portrait"
    )

    long_video = OUT / "long.mp4"
    short_video = OUT / "short.mp4"

    print("Ana video hazırlanıyor...")

    make_video(
        long_audio,
        long_images,
        long_video
    )

    print("Short hazırlanıyor...")

    make_video(
        short_audio,
        short_images,
        short_video,
        vertical=True
    )

    print("Kapak hazırlanıyor...")

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
        str(
            data["description"]
        ).strip()
        + "\n\n"
        + "Bu video özgün senaryo, yapay zekâ "
        + "destekli seslendirme ve görseller "
        + "kullanılarak hazırlanmıştır."
        + "\n\n"
        + "Pexels kaynakları:\n"
        + "\n".join(unique_credits)
        + "\n\n"
        + "#KayıpHikayeler #Gizem "
        + "#GerçekHikayeler #Belgesel"
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
