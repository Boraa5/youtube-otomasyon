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
# AYARLAR
# ============================================================

OUT = Path("work")
OUT.mkdir(exist_ok=True)

GEMINI = os.environ["GEMINI_API_KEY"]
PEXELS = os.environ["PEXELS_API_KEY"]

MODEL = "gemini-3.1-flash-lite"

client = genai.Client(api_key=GEMINI)


# ============================================================
# TÜRKÇE PIPER SESİ
# ============================================================

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


def dur(file_path):
    result = subprocess.check_output(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ],
        text=True
    )

    return float(result.strip())


def get(url, path, minimum=500):
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
            "Eksik dosya indirildi: " + str(path)
        )


# ============================================================
# PIPER KURULUMU
# ============================================================

def setup():
    print("Türkçe ses modeli kontrol ediliyor...")

    get(
        PIPER_URL + "tr_TR-dfki-medium.onnx",
        PIPER,
        50_000_000
    )

    get(
        PIPER_URL + "tr_TR-dfki-medium.onnx.json",
        PIPER_CFG,
        500
    )


# ============================================================
# GEMINI
# ============================================================

def gemini(prompt):
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
            message = str(error)

            print("Gemini hatası:", message)

            if (
                "429" in message
                or "RESOURCE_EXHAUSTED" in message
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
        "Gemini içerik üretilemedi."
    )


# ============================================================
# YENİ NİŞ:
# GERÇEK, İNANILMAZ, MERAK UYANDIRAN HİKÂYELER
# ============================================================

def plan():
    prompt = r"""
Türkçe YouTube kanalı "Kayıp Hikâyeler" için TEK BİR gerçek
ve çok merak uyandıran hikâye hazırla.

KANALIN YENİ NİŞİ:

Gerçek hayatta yaşanmış;
- yıllarca kayıp kalan insanların bulunması,
- kaybolan kişilerin inanılmaz şekilde ortaya çıkması,
- polisin veya ailenin uzun süre çözemediği gerçek olaylar,
- bir evde, odada, binada veya çok yakında olmasına rağmen
  yıllarca fark edilmeyen insanlar veya kanıtlar,
- kayıp vakalarının beklenmedik şekilde çözülmesi,
- yıllar sonra ortaya çıkan sırlar,
- gerçek ve doğrulanabilir inanılmaz insan hikâyeleri,
- herkesin yanlış düşündüğü ama sonradan gerçeğin ortaya çıktığı
  gerçek olaylar.

KONU ARKEOLOJİ OLMASIN.
TARİH BELGESELİ GİBİ GENEL BİR KONU OLMASIN.
UZAY, ANTİK MEDENİYET VE RASTGELE BİLİM KONUSU SEÇME.

Hikâye MUTLAKA gerçek, doğrulanabilir ve tek bir ana olay
etrafında ilerlemeli.

Uydurma kişi, tarih, sayı, polis bilgisi veya olay oluşturma.
Doğrulanamayan söylentileri gerçek gibi anlatma.

Mümkün olduğunca güvenilir kaynaklarda belgelenmiş gerçek
olaylardan yararlan.

------------------------------------------------------------

ANA VİDEO:

Uzunluk hedefi yaklaşık 7-9 dakika.

Narration TAM OLARAK 1050-1250 Türkçe kelime arasında olsun.

İLK 5 SANİYE ÇOK KRİTİK.

Video doğrudan olayın en inanılmaz yerinden başlamalı.

KESİNLİKLE ŞUNLARI KULLANMA:
"Merhaba"
"Kanalımıza hoş geldiniz"
"Bugün sizlere"
"Bu videoda"
"Şimdi anlatacağımız"

İlk cümle izleyiciyi doğrudan olayın içine atsın.

ÖRNEK MANTIK:
"Polis bu evi defalarca aradı. Ama aradıkları kişi
12 yıl boyunca aslında çok yakındaydı."

Bu örneği aynen kullanma. Sadece mantığını uygula.

İLK 15 SANİYEDE:
- inanılmaz olay açıkça hissettirilsin,
- büyük soru oluşturulsun,
- izleyici "nasıl oldu?" diye merak etsin.

İLK 30 SANİYEDE:
- hikâyenin neden inanılmaz olduğu netleşsin,
- ancak bütün cevap hemen verilmesin.

Hikâye boyunca yeni bilgiler kademeli ortaya çıksın.
Her 20-40 saniyede yeni bir merak, soru, ayrıntı veya
beklenmedik gelişme olsun.

Anlatım doğal Türkçe konuşma diliyle yazılsın.
Robot gibi, Wikipedia gibi veya ders anlatır gibi olmasın.

YAPI:

1. Çok güçlü ve şok edici açılış
2. Büyük soru
3. Olayın başlangıcı
4. Kayıp veya gizemin derinleşmesi
5. Aramalar ve başarısız girişimler
6. Yeni ipuçları
7. Beklenmedik gelişme
8. Gerçeğin ortaya çıkışı
9. Olayın en inanılmaz ayrıntısı
10. Kısa ama etkili sonuç

Sonuçta hikâyenin tamamı açıklansın.
Gereksiz uzun kapanış yapma.

Narration içinde:
- başlık yazma
- bölüm numarası yazma
- kaynak listesi yazma
- "giriş", "sonuç" gibi etiketler yazma

------------------------------------------------------------

BAŞLIK:

Başlık kısa, güçlü ve merak uyandırıcı olsun.

Clickbait olarak yalan söyleme.

Mümkünse şu mantıklardan birini kullan:
"12 Yıl Kayıptı... Ama Polis Onu İlk Aradığı Yerde Buldu"
"Herkes Onun Öldüğünü Sandı... Yıllar Sonra Gerçek Ortaya Çıktı"
"Polis Bu Odayı Defalarca Aradı... Ama Gerçeği Göremedi"

Bu örnekleri aynen kopyalama.

Başlık olayın en güçlü gizemini söylesin ama bütün cevabı
vermesin.

------------------------------------------------------------

SHORT:

Short aynı gerçek hikâyeye bağlı olsun.

Short, ana videodan tamamen kopuk başka bir konu olmasın.

TAM OLARAK 30-50 saniyelik anlatıma uygun olsun.

short_narration yaklaşık 75-115 Türkçe kelime olsun.

İlk 3 saniye çok güçlü olsun.

Short:
- olayın en inanılmaz anından başlamalı,
- bütün hikâyeyi anlatmamalı,
- ana videonun cevabını tamamen vermemeli,
- izleyicide devamını öğrenme isteği bırakmalı.

Short'un son cümlesi doğal biçimde merak bıraksın.
"Devamı kanalda" gibi zoraki reklam cümlesi kullanma.

------------------------------------------------------------

PEXELS:

ANA VİDEO İÇİN TAM OLARAK 12 İngilizce Pexels sorgusu üret.

SHORT İÇİN TAM OLARAK 6 İngilizce Pexels sorgusu üret.

Her sorgu:
- İngilizce
- 2 ile 6 kelime arasında
- gerçek insan, ev, polis, sokak, orman, bina,
  kayıp kişi atmosferi, belge, arama veya olayla ilgili olsun
- soyut ve anlamsız kelimeler kullanma

THUMBNAIL İÇİN TAM OLARAK 1 İngilizce Pexels sorgusu üret.

Thumbnail sorgusu çok güçlü görsel oluşturabilecek şekilde
hikâyenin en gizemli nesnesini, yeri veya atmosferini tarif etsin.

------------------------------------------------------------

JSON KURALLARI:

SADECE geçerli JSON döndür.
JSON dışında hiçbir açıklama yazma.

Alanlar TAM OLARAK şunlar:

title
description
tags
narration
short_title
short_narration
scene_queries
short_queries
thumbnail_query

ZORUNLU KURALLAR:

- Hiçbir alan boş olamaz.
- title boş olamaz.
- description boş olamaz.
- tags en az 5 adet olmalı.
- narration 1050-1250 Türkçe kelime olmalı.
- short_narration 75-115 Türkçe kelime olmalı.
- scene_queries TAM 12 adet olmalı.
- short_queries TAM 6 adet olmalı.
- thumbnail_query boş olamaz.
- Uydurma bilgi kullanma.
- Hikâye tek bir gerçek olay etrafında ilerlesin.
- Arkeoloji konusu seçme.
- Genel tarih belgeseli konusu seçme.
- Short aynı hikâyeden olmalı.
"""

    last_error = "Henüz üretim yapılmadı."

    for attempt in range(5):
        try:
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

            if not all(
                field in data
                for field in required
            ):
                raise ValueError(
                    "JSON alanlarından biri eksik."
                )

            if not str(data["title"]).strip():
                raise ValueError("Başlık boş.")

            if not str(data["description"]).strip():
                raise ValueError("Açıklama boş.")

            if len(data["tags"]) < 5:
                raise ValueError(
                    "En az 5 etiket gerekli."
                )

            narration_words = wc(
                data["narration"]
            )

            if not 1050 <= narration_words <= 1250:
                raise ValueError(
                    f"Ana anlatım kelime sayısı yanlış: "
                    f"{narration_words}"
                )

            short_words = wc(
                data["short_narration"]
            )

            if not 75 <= short_words <= 115:
                raise ValueError(
                    f"Short kelime sayısı yanlış: "
                    f"{short_words}"
                )

            if len(data["scene_queries"]) != 12:
                raise ValueError(
                    "Ana video için tam 12 sahne gerekli."
                )

            if len(data["short_queries"]) != 6:
                raise ValueError(
                    "Short için tam 6 sahne gerekli."
                )

            if not str(
                data["thumbnail_query"]
            ).strip():
                raise ValueError(
                    "Thumbnail sorgusu boş."
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
                time.sleep(4)

    raise RuntimeError(
        "Geçerli plan üretilemedi. "
        "Son hata: " + last_error
    )


# ============================================================
# PEXELS
# ============================================================

def pexels(query, orientation, used):
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

    if not query:
        query = "mystery documentary"

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
            "Pexels sonucu bulunamadı: " + query
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

    source = photo["src"]

    if orientation == "portrait":
        image_url = source.get("portrait")
    else:
        image_url = source.get("landscape")

    if not image_url:
        image_url = (
            source.get("large2x")
            or source.get("original")
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

    for index, query in enumerate(
        queries,
        1
    ):
        photo, image_url = pexels(
            query,
            orientation,
            used
        )

        file_path = (
            OUT /
            f"{prefix}_{index:02d}.jpg"
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
                "Bozuk Pexels görseli."
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
        split_at = text.rfind(
            " ",
            0,
            2200
        )

        if split_at < 500:
            split_at = 2200

        chunks.append(
            text[:split_at]
        )

        text = text[split_at:].strip()

    if text:
        chunks.append(text)

    wav_files = []

    for index, chunk in enumerate(chunks):
        wav_file = (
            OUT /
            f"{output.stem}_{index:03d}.wav"
        )

        run(
            [
                "python",
                "-m",
                "piper",
                "--model",
                str(PIPER),
                "--output_file",
                str(wav_file),
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

        wav_files.append(wav_file)

    concat_file = (
        OUT /
        f"{output.stem}_concat.txt"
    )

    concat_file.write_text(
        "\n".join(
            "file '" +
            str(wav.resolve()).replace(
                "'",
                "'\\''"
            ) +
            "'"
            for wav in wav_files
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
            str(concat_file),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "160k",
            str(output)
        ]
    )


# ============================================================
# VİDEO OLUŞTURMA
# ============================================================

def make_video(
    audio,
    image_files,
    output,
    vertical=False
):
    total = dur(audio)

    each = total / len(image_files)

    if each < 2:
        raise RuntimeError(
            "Sahne süresi çok kısa."
        )

    slide_list = (
        OUT /
        f"{output.stem}_slides.txt"
    )

    lines = []

    for image_file in image_files:
        safe_path = str(
            image_file.resolve()
        ).replace(
            "'",
            "'\\''"
        )

        lines.append(
            f"file '{safe_path}'"
        )

        lines.append(
            f"duration {each:.3f}"
        )

    last_path = str(
        image_files[-1].resolve()
    ).replace(
        "'",
        "'\\''"
    )

    lines.append(
        f"file '{last_path}'"
    )

    slide_list.write_text(
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
            str(slide_list),
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
# YOUTUBE BAĞLANTISI
# ============================================================

def yt():
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


# ============================================================
# YOUTUBE YÜKLEME
# ============================================================

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

    result = None

    while result is None:
        status, result = request.next_chunk()

        if status:
            print(
                "YouTube %d%%" %
                int(
                    status.progress() * 100
                )
            )

    video_id = result["id"]

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
        "Yüklendi:",
        video_id
    )

    return video_id


# ============================================================
# THUMBNAIL
# ============================================================

def thumb(source, output):
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
    print("=" * 55)
    print("KAYIP HİKÂYELER - YENİ OTOMASYON")
    print("=" * 55)

    print(
        "Niş: Gerçek, inanılmaz ve "
        "merak uyandıran hikâyeler"
    )

    setup()

    print("Hikâye hazırlanıyor...")
    data = plan()

    print()
    print("KONU:")
    print(data["title"])

    print()
    print(
        "Ana anlatım:",
        wc(data["narration"]),
        "kelime"
    )

    print(
        "Short anlatım:",
        wc(data["short_narration"]),
        "kelime"
    )

    long_audio = OUT / "long.mp3"
    short_audio = OUT / "short.mp3"

    print()
    print("Ana ses oluşturuluyor...")

    tts(
        data["narration"],
        long_audio
    )

    print("Short sesi oluşturuluyor...")

    tts(
        data["short_narration"],
        short_audio
    )

    long_duration = dur(
        long_audio
    )

    short_duration = dur(
        short_audio
    )

    print()
    print(
        "Ana video:",
        round(long_duration / 60, 2),
        "dakika"
    )

    print(
        "Short:",
        round(short_duration, 1),
        "saniye"
    )

    if not 360 <= long_duration <= 660:
        raise RuntimeError(
            f"Ana video süresi uygun değil: "
            f"{long_duration:.1f} saniye"
        )

    if not 25 <= short_duration <= 60:
        raise RuntimeError(
            f"Short süresi uygun değil: "
            f"{short_duration:.1f} saniye"
        )

    print()
    print("Ana video görselleri indiriliyor...")

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

    print()
    print("Ana video oluşturuluyor...")

    make_video(
        long_audio,
        long_images,
        long_video
    )

    print(
        "Short video oluşturuluyor..."
    )

    make_video(
        short_audio,
        short_images,
        short_video,
        vertical=True
    )

    print(
        "Thumbnail görseli indiriliyor..."
    )

    thumbnail_images, _ = images(
        [data["thumbnail_query"]],
        "thumbnail",
        "landscape"
    )

    thumbnail = (
        OUT /
        "thumbnail.jpg"
    )

    thumb(
        thumbnail_images[0],
        thumbnail
    )

    credits = []

    for photo in (
        long_credits +
        short_credits
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
        +
        "\n\n"
        "Bu video özgün senaryo, yapay zekâ destekli "
        "seslendirme ve Pexels görselleri kullanılarak "
        "hazırlanmıştır."
        +
        "\n\n"
        "Pexels kaynakları:\n"
        +
        "\n".join(unique_credits)
        +
        "\n\n"
        "#KayıpHikayeler #GerçekHikaye "
        "#Gizem #İnanılmazHikayeler"
    )

    print()
    print("YouTube bağlantısı kuruluyor...")

    api = yt()

    print()
    print("ANA VİDEO YÜKLENİYOR...")

    upload(
        api,
        long_video,
        data["title"],
        description,
        data["tags"],
        thumbnail
    )

    print()
    print("SHORT YÜKLENİYOR...")

    upload(
        api,
        short_video,
        data["short_title"],
        description,
        data["tags"]
    )

    print()
    print("=" * 55)
    print("OTOMASYON BAŞARIYLA TAMAMLANDI")
    print("=" * 55)


if __name__ == "__main__":
    main()
