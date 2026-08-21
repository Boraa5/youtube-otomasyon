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
#
# NİŞ:
# Gerçek, inanılmaz, merak uyandıran kaybolma / bulunma /
# yıllar sonra ortaya çıkma / gizli kalmış gerçek hikâyeler
#
# ANA VİDEO:
# Doğal uzunlukta gerçek hikâye
#
# SHORT:
# Aynı hikâyenin en güçlü ve merak uyandıran anı
#
# KAPAK:
# Konuya özel, güçlü, gizemli ve merak uyandıran görsel
# ============================================================


OUT = Path("work")
OUT.mkdir(exist_ok=True)

HISTORY_FILE = OUT / "history.json"

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


def load_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(
            HISTORY_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):
            return data[-30:]

    except Exception:
        pass

    return []


def save_history(title):
    history = load_history()

    title = str(title).strip()

    if title:
        history.append(title)

    history = history[-30:]

    HISTORY_FILE.write_text(
        json.dumps(
            history,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


def normalize_queries(
    queries,
    count,
    fallback
):
    cleaned = []

    if isinstance(queries, list):

        for query in queries:

            query = clean_query(query)

            if (
                query
                and query not in cleaned
            ):
                cleaned.append(query)

    cleaned = cleaned[:count]

    fallback_number = 0

    while len(cleaned) < count:

        fallback_number += 1

        if fallback_number == 1:
            candidate = fallback
        else:
            candidate = (
                fallback
                + " "
                + str(fallback_number)
            )

        if candidate not in cleaned:
            cleaned.append(candidate)

    return cleaned


def normalize_tags(tags):
    cleaned = []

    if isinstance(tags, list):

        for tag in tags:

            tag = str(tag).strip()

            if (
                tag
                and tag not in cleaned
            ):
                cleaned.append(tag)

    fallback_tags = [
        "Kayıp Hikâyeler",
        "Gerçek Hikâyeler",
        "Gizem",
        "Gerçek Olaylar",
        "Belgesel",
        "İnanılmaz Hikâyeler",
        "Gizemli Olaylar",
        "Çözülen Gizemler"
    ]

    for tag in fallback_tags:

        if len(cleaned) >= 5:
            break

        if tag not in cleaned:
            cleaned.append(tag)

    return cleaned[:15]


# ============================================================
# TÜRKÇE SES MODELİ
# ============================================================


def setup():

    print(
        "Türkçe ses modeli kontrol ediliyor..."
    )

    download(
        PIPER_URL
        + "tr_TR-dfki-medium.onnx",
        PIPER,
        50_000_000
    )

    download(
        PIPER_URL
        + "tr_TR-dfki-medium.onnx.json",
        PIPER_CFG,
        500
    )

    print(
        "Türkçe ses modeli hazır."
    )


# ============================================================
# GEMINI
# ============================================================


def gemini(prompt):

    last_error = None

    for attempt in range(5):

        try:

            response = (
                client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type=(
                            "application/json"
                        ),
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
                                },

                                "thumbnail_queries": {
                                    "type": "ARRAY",
                                    "items": {
                                        "type": "STRING"
                                    }
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
                                "thumbnail_query",
                                "thumbnail_queries"
                            ]
                        },

                        temperature=0.9
                    )
                )
            )

            return json.loads(
                response.text
            )

        except Exception as error:

            last_error = str(error)

            print(
                "Gemini hata:",
                last_error
            )

            if (
                "429" in last_error
                or "RESOURCE_EXHAUSTED"
                in last_error
            ):

                wait = 20 * (
                    attempt + 1
                )

                print(
                    f"Gemini sınırı. "
                    f"{wait} saniye bekleniyor..."
                )

                time.sleep(wait)

            else:

                if attempt < 4:
                    time.sleep(5)

    raise RuntimeError(
        "Gemini içerik üretilemedi: "
        + str(last_error)
    )


# ============================================================
# HİKAYE PLANI
# ============================================================


def plan():

    previous_titles = load_history()

    previous_text = "\n".join(
        "- " + title
        for title in previous_titles[-20:]
    )

    if not previous_text:
        previous_text = (
            "- Henüz geçmiş video yok"
        )

    prompt = f"""
"Kayıp Hikâyeler" adlı Türkçe YouTube kanalı için
TEK bir gerçek hikâye hazırla.

KANALIN NİŞİ:

Gerçek hayatta yaşanmış,
inanılmaz,
merak uyandıran,
gizemli ve doğrulanabilir olaylar.

ÖNCELİKLİ KONU TÜRLERİ:

- Yıllarca kayıp kalan insanların bulunması
- Beklenmedik şekilde ortaya çıkan kişiler
- Yıllarca çözülemeyen gerçek gizemler
- Herkesin gözünün önünde olup fark edilmeyen gerçekler
- Kayıp bir nesnenin veya önemli şeyin bulunması
- Gizli kalmış olayların yıllar sonra ortaya çıkması
- Terk edilmiş bir yerde bulunan önemli bir gerçek
- Kaybolmuş veya unutulmuş bir şeyin yeniden ortaya çıkması
- Gerçek ve belgelenmiş şaşırtıcı olaylar

ÖNEMLİ:

Sadece arkeoloji ve tarih anlatma.

Her video "12 yıl kayıptı",
"yıllarca kayboldu",
"polis onu bulamadı"
gibi aynı başlık ve olay kalıbına sıkışmasın.

Her videoda konu türünü,
olayın merkezindeki kişiyi,
olayın dönemini,
gizemin şeklini ve
başlık yapısını mümkün olduğunca değiştir.

Aşağıdaki daha önce kullanılan başlıklara
aynı veya çok benzer bir hikâye üretme:

{previous_text}

HİKÂYE GERÇEK OLMALI.

Uydurma kişi,
tarih,
sayı,
yer,
kanıt veya olay kullanma.

Belirsiz ayrıntıları kesin gerçek gibi anlatma.

Tartışmalı bilgiler varsa
bunu dengeli biçimde belirt.

============================================================

ANA VİDEO

Hikâyeyi doğal uzunlukta anlat.

Kesin olarak
5,
7,
8 veya
9 dakika doldurmaya çalışma.

Hikâyeyi gereksiz tekrarlarla uzatma.

Ama olayın önemli kısmını da
eksik bırakacak kadar kısa kesme.

İlk cümle doğrudan olayın
en şaşırtıcı veya gizemli kısmıyla başlasın.

Şunlarla başlama:

"Merhaba"
"Bugün sizlere"
"Bu videoda"
"Kanalımıza hoş geldiniz"

İLK 5 SANİYE:

İzleyiciyi durduracak kadar güçlü
bir merak oluştur.

İLK 15 SANİYE:

İzleyicinin zihninde
büyük bir soru oluşmalı.

Yapı doğal olarak şunları içerebilir:

- Güçlü açılış
- Kısa arka plan
- Gizemin başlaması
- Arama veya araştırma
- Gözden kaçan ayrıntılar
- Beklenmedik dönüm noktası
- Güçlü kanıtlar
- Gerçeğin nasıl ortaya çıktığı
- Kısa ve etkili kapanış

Aynı bilgiyi tekrar tekrar yazma.

Hikâye boyunca yeni bilgi,
yeni ayrıntı ve
yeni merak unsuru ver.

============================================================

SHORT

Short,
ANA VİDEODAKİ AYNI GERÇEK HİKÂYEYE bağlı olacak.

Ama ana videonun küçük bir özeti olmayacak.

Hikâyenin en güçlü,
en şaşırtıcı veya
en merak uyandırıcı bölümünü seç.

Short çok kısa kalmamalı.

Yaklaşık
30 ile 55 saniyelik
doğal bir anlatım hedefle.

Bunun için genellikle
yaklaşık 70 ile 140 Türkçe kelime uygundur.

İlk 3 saniye çok güçlü olmalı.

İlk cümle izleyicinin
hemen devamını merak etmesini sağlamalı.

Gereksiz cümlelerle süre doldurma.

Ama 10-15 saniyelik
çok kısa bir Short da üretme.

============================================================

BAŞLIK

Ana video başlığı:

- Kısa
- Güçlü
- Merak uyandırıcı
- Gerçeğe uygun

olmalı.

Clickbait olabilir,
ama yalan söyleme.

Her seferinde aynı kalıbı kullanma.

Özellikle sürekli:

"12 Yıl Kayıptı"
"Yıllarca Kayıptı"
"Polis Onu Bulamadı"

gibi benzer başlıklar üretme.

Olay neyi ilginç yapıyorsa
başlığı onun üzerinden kur.

Başlık türlerini değiştir.

Örneğin bazen:

- Beklenmedik bir keşif
- Gizli bir ayrıntı
- Yanlış anlaşılan bir olay
- Yıllar sonra bulunan bir kanıt
- Herkesin gözünden kaçan bir gerçek
- Ortaya çıkan gizli bağlantı

üzerinden merak oluştur.

Ama bunları da sürekli aynı şekilde kullanma.

============================================================

VİDEO KAPAK RESMİ - ÇOK ÖNEMLİ

Kapak görseli sıradan,
genel veya rastgele olmamalı.

Kapak,
hikâyenin EN GÜÇLÜ GÖRSEL ANI olmalı.

İzleyici kapağa baktığında:

"Burada ne olmuş?"
"Bu neyin resmi?"
"Devamında ne çıkacak?"

diye merak etmeli.

Kapak için sadece:

missing person
police investigation
old house

gibi sürekli kullanılan sıradan
ve genel sorgular üretme.

HİKÂYEYE ÖZEL BİR GÖRSEL DURUM seç.

Örneğin olayda:

- Kapalı bir kapının arkasında sır varsa
  kapak buna odaklansın.

- Ormanda bulunan bir kanıt varsa
  kapak buna odaklansın.

- Eski bir fotoğraf olayı çözdüyse
  kapak fotoğrafın gizemini yansıtsın.

- Terk edilmiş bir yerde bir şey bulunduysa
  o anın atmosferini yansıtsın.

- Önemli bir nesne varsa
  nesne kapakta merkezi unsur olsun.

- Bir araştırma sırasında beklenmedik
  bir ayrıntı ortaya çıktıysa
  kapak o ayrıntının etrafındaki
  gizemi yansıtsın.

KAPAKTA YAZI OLMAYACAK.

Kapak görselinde mümkünse:

- Tek güçlü ana unsur
- Net görsel odak
- Gizemli veya dramatik atmosfer
- İnsan varsa yüz veya hareketin anlaşılması
- Gereksiz kalabalık olmaması

sağlansın.

thumbnail_query:

Hikâyeye özel
EN GÜÇLÜ tek İngilizce Pexels sorgusu.

thumbnail_queries:

Aynı kapak fikrini destekleyen
TAM OLARAK 5 farklı
güçlü İngilizce alternatif sorgu.

Bunlar birbirinin aynısı olmasın.

Sorgular Pexels'te sonuç bulunabilecek kadar
gerçekçi ve genel olmalı.

============================================================

PEXELS SORGULARI

Ana video için
12 İngilizce sahne sorgusu üret.

Short için
6 İngilizce sahne sorgusu üret.

Her sorgu:

- İngilizce
- Gerçekçi
- Pexels'te aranabilir
- Hikâyeye uygun
- Mümkünse 2 ile 6 kelime

olmalı.

============================================================

AÇIKLAMA

description alanına kısa ve doğal
YouTube açıklaması yaz.

============================================================

ETİKETLER

tags alanında
en az 5,
en fazla 15
etiket olsun.

============================================================

NARRATION İÇİN

- Başlık yazma
- Kaynak listesi yazma
- Bölüm numarası yazma
- JSON yazma

Sadece doğal anlatım yaz.

============================================================

ZORUNLU ALANLAR:

title
description
tags
narration
short_title
short_narration
scene_queries
short_queries
thumbnail_query
thumbnail_queries

KONTROL:

- Hiçbir alan boş olmayacak
- Ana hikâye gerçek olacak
- Short aynı gerçek hikâyeye bağlı olacak
- Short yaklaşık 30-55 saniyeyi hedefleyecek
- scene_queries 12 adet olacak
- short_queries 6 adet olacak
- thumbnail_queries 5 adet olacak
- Başlıklar önceki başlıklarla çok benzer olmayacak
- Kapak sorguları konuya özel ve güçlü olacak
- Kapak sıradan rastgele bir görsel mantığıyla üretilmeyecek

SADECE GEÇERLİ JSON DÖNDÜR.

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
                "thumbnail_query",
                "thumbnail_queries"
            ]

            for field in required:

                if field not in data:
                    raise ValueError(
                        f"Eksik alan: {field}"
                    )

            text_fields = [
                "title",
                "description",
                "narration",
                "short_title",
                "short_narration",
                "thumbnail_query"
            ]

            for field in text_fields:

                value = str(
                    data[field]
                ).strip()

                if not value:
                    raise ValueError(
                        f"Boş alan: {field}"
                    )

                data[field] = value

            main_words = wc(
                data["narration"]
            )

            short_words = wc(
                data["short_narration"]
            )

            if main_words < 120:

                raise ValueError(
                    f"Ana anlatım fazla kısa: "
                    f"{main_words} kelime"
                )

            # Short artık 15 saniye gibi çok kısa çıkmasın.
            if short_words < 65:

                raise ValueError(
                    f"Short fazla kısa: "
                    f"{short_words} kelime"
                )

            data["scene_queries"] = (
                normalize_queries(
                    data["scene_queries"],
                    12,
                    "mysterious real discovery"
                )
            )

            data["short_queries"] = (
                normalize_queries(
                    data["short_queries"],
                    6,
                    "dramatic mysterious discovery"
                )
            )

            data["thumbnail_query"] = (
                clean_query(
                    data["thumbnail_query"]
                )
            )

            if not data["thumbnail_query"]:

                data["thumbnail_query"] = (
                    "mysterious hidden discovery"
                )

            data["thumbnail_queries"] = (
                normalize_queries(
                    data["thumbnail_queries"],
                    5,
                    "mysterious hidden discovery"
                )
            )

            data["tags"] = normalize_tags(
                data["tags"]
            )

            print(
                "Plan kabul edildi."
            )

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


def pexels_search(
    query,
    orientation,
    used=None
):

    query = clean_query(query)

    if not query:
        query = "mysterious documentary"

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
        return None, None

    if used is None:
        used = set()

    photo = next(
        (
            item
            for item in photos
            if item.get("id") not in used
        ),
        photos[0]
    )

    used.add(
        photo.get("id")
    )

    src = photo.get(
        "src",
        {}
    )

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


def pexels(
    query,
    orientation,
    used
):

    photo, image_url = pexels_search(
        query,
        orientation,
        used
    )

    if photo and image_url:
        return photo, image_url

    fallback_queries = [
        "mysterious discovery",
        "dark empty room",
        "old hidden object",
        "investigation evidence",
        "mysterious abandoned place"
    ]

    for fallback in fallback_queries:

        photo, image_url = pexels_search(
            fallback,
            orientation,
            used
        )

        if photo and image_url:

            print(
                "Pexels yedek sorgu kullanıldı:",
                fallback
            )

            return photo, image_url

    raise RuntimeError(
        "Pexels görseli bulunamadı: "
        + str(query)
    )


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
# THUMBNAIL GÖRSELİ
# ============================================================


def download_thumbnail(
    main_query,
    alternative_queries
):

    all_queries = []

    main_query = clean_query(
        main_query
    )

    if main_query:
        all_queries.append(
            main_query
        )

    if isinstance(
        alternative_queries,
        list
    ):

        for query in alternative_queries:

            query = clean_query(
                query
            )

            if (
                query
                and query not in all_queries
            ):
                all_queries.append(
                    query
                )

    fallback_queries = [
        "mysterious hidden discovery",
        "dramatic abandoned room",
        "dark secret door",
        "old mysterious evidence",
        "lonely person investigation"
    ]

    for query in fallback_queries:

        if query not in all_queries:
            all_queries.append(query)

    print("=" * 60)
    print("KAPAK GÖRSELİ ARANIYOR")
    print("=" * 60)

    for number, query in enumerate(
        all_queries,
        1
    ):

        print(
            f"Kapak sorgusu "
            f"{number}/{len(all_queries)}: "
            f"{query}"
        )

        try:

            photo, image_url = pexels_search(
                query,
                "landscape",
                set()
            )

            if not (
                photo
                and image_url
            ):
                continue

            file_path = OUT / (
                "thumbnail_source.jpg"
            )

            response = requests.get(
                image_url,
                timeout=90
            )

            response.raise_for_status()

            file_path.write_bytes(
                response.content
            )

            if file_path.stat().st_size >= 5000:

                print(
                    "Kapak görseli bulundu:",
                    query
                )

                return file_path, photo

        except Exception as error:

            print(
                "Kapak sorgusu başarısız:",
                error
            )

    raise RuntimeError(
        "Kapak için uygun görsel bulunamadı"
    )


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

        text = text[
            split:
        ].strip()

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
            f"{number + 1}/"
            f"{len(chunks)}"
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
            "file '"
            + escaped
            + "'"
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

    total_duration = duration(
        audio
    )

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
            "file '"
            + escaped
            + "'"
        )

        lines.append(
            f"duration "
            f"{each_duration:.3f}"
        )

    last_image = str(
        image_files[-1].resolve()
    ).replace(
        "'",
        "'\\''"
    )

    lines.append(
        "file '"
        + last_image
        + "'"
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

            "title": str(
                title
            )[:100],

            "description": str(
                description
            )[:5000],

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

    request = (
        api.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )
    )

    response = None

    while response is None:

        status, response = (
            request.next_chunk()
        )

        if status:

            print(
                "YouTube "
                + str(
                    int(
                        status.progress()
                        * 100
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


def thumbnail(
    source,
    output
):

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
            "5",
            str(output)
        ]
    )

    if (
        output.stat().st_size
        > 1_900_000
    ):

        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(output),
                "-q:v",
                "9",
                str(output)
            ]
        )


# ============================================================
# ANA PROGRAM
# ============================================================


def main():

    print("=" * 60)
    print(
        "KAYIP HİKAYELER - "
        "YENİ OTOMASYON"
    )
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
        wc(
            data["narration"]
        )
    )

    print(
        "Short kelime:",
        wc(
            data["short_narration"]
        )
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
        round(
            long_duration,
            1
        ),
        "saniye"
    )

    print(
        "Short süresi:",
        round(
            short_duration,
            1
        ),
        "saniye"
    )

    if long_duration < 30:

        raise RuntimeError(
            "Ana video olağan dışı kısa: "
            + f"{long_duration:.1f}"
            + " saniye"
        )

    # Short 15 saniye gibi kısa çıkmasın.
    if short_duration < 25:

        raise RuntimeError(
            "Short fazla kısa: "
            + f"{short_duration:.1f}"
            + " saniye"
        )

    print(
        "Ana video görselleri "
        "indiriliyor..."
    )

    long_images, long_credits = (
        images(
            data["scene_queries"],
            "long",
            "landscape"
        )
    )

    print(
        "Short görselleri "
        "indiriliyor..."
    )

    short_images, short_credits = (
        images(
            data["short_queries"],
            "short",
            "portrait"
        )
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

    thumbnail_source, thumbnail_credit = (
        download_thumbnail(
            data["thumbnail_query"],
            data["thumbnail_queries"]
        )
    )

    thumbnail_file = (
        OUT / "thumbnail.jpg"
    )

    thumbnail(
        thumbnail_source,
        thumbnail_file
    )

    credits = []

    for photo in (
        long_credits
        + short_credits
        + [thumbnail_credit]
    ):

        if not photo:
            continue

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
                f"Photo by "
                f"{photographer} "
                f"on Pexels: "
                f"{photo_url}"
            )

    unique_credits = list(
        dict.fromkeys(
            credits
        )
    )

    description = (
        str(
            data["description"]
        ).strip()
        + "\n\n"
        + "Bu video özgün senaryo, "
        + "yapay zekâ destekli seslendirme "
        + "ve görseller kullanılarak "
        + "hazırlanmıştır."
        + "\n\n"
        + "Pexels kaynakları:\n"
        + "\n".join(
            unique_credits
        )
        + "\n\n"
        + "#KayıpHikayeler "
        + "#Gizem "
        + "#GerçekHikayeler "
        + "#Belgesel"
    )

    print(
        "YouTube bağlantısı "
        "hazırlanıyor..."
    )

    api = youtube()

    print("=" * 60)
    print(
        "ANA VİDEO YÜKLENİYOR"
    )
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
    print(
        "SHORT YÜKLENİYOR"
    )
    print("=" * 60)

    upload(
        api,
        short_video,
        data["short_title"],
        description,
        data["tags"]
    )

    # Video başarıyla üretildi ve yüklendi.
    # Aynı başlıkların tekrarını önlemek için kaydet.
    save_history(
        data["title"]
    )

    print("=" * 60)
    print(
        "OTOMASYON BAŞARIYLA "
        "TAMAMLANDI"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
