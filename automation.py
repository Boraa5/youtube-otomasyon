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
# KAYIP HİKAYELER
# TÜRKİYE'DE KONUŞULAN ÜNLÜLER VE FENOMENLER
#
# ANA VİDEO:
# Gerçek, doğrulanabilir, merak uyandıran hikâyeler
# Hedef süre: yaklaşık 6-8 dakika
#
# SHORT:
# Aynı hikâyenin en güçlü merak anı
# Hedef süre: yaklaşık 25-45 saniye
#
# TEKRAR KONTROLÜ:
# Son kullanılan kişi, konu ve başlıklar kaydedilir.
# Aynı kişi ve aynı olay sürekli tekrar seçilmez.
# ============================================================


OUT = Path("work")
OUT.mkdir(exist_ok=True)

HISTORY_FILE = OUT / "topic_history.json"

GEMINI = os.environ["GEMINI_API_KEY"]
PEXELS = os.environ["PEXELS_API_KEY"]

MODEL = "gemini-3.1-flash-lite"

client = genai.Client(api_key=GEMINI)


# ============================================================
# TÜRKÇE SES MODELİ
# ============================================================


PIPER = OUT / "tr_TR-dfki-medium.onnx"
PIPER_CFG = OUT / "tr_TR-dfki-medium.onnx.json"

PIPER_URL = (
    "https://huggingface.co/rhasspy/piper-voices/"
    "resolve/v1.0.0/tr/tr_TR/dfki/medium/"
)


# ============================================================
# GENEL AYARLAR
# ============================================================


MAIN_MIN_WORDS = 850
MAIN_MAX_WORDS = 1250

SHORT_MIN_WORDS = 60
SHORT_MAX_WORDS = 115

MAIN_MIN_SECONDS = 300
MAIN_MAX_SECONDS = 540

SHORT_MIN_SECONDS = 22
SHORT_MAX_SECONDS = 55


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


def clean_text(text):
    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


def extract_json(text):
    text = str(text).strip()

    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.I
    )

    text = re.sub(
        r"```$",
        "",
        text
    ).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        return json.loads(
            text[start:end + 1]
        )

    raise ValueError(
        "Gemini geçerli JSON döndürmedi"
    )


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
            return data

    except Exception:
        pass

    return []


def save_history(item):
    history = load_history()

    history.append(item)

    # Son 60 kaydı sakla.
    history = history[-60:]

    HISTORY_FILE.write_text(
        json.dumps(
            history,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


def history_summary(history):
    if not history:
        return "Henüz kullanılmamış konu yok."

    lines = []

    for item in history[-30:]:
        person = clean_text(
            item.get("person", "")
        )

        topic = clean_text(
            item.get("topic", "")
        )

        title = clean_text(
            item.get("title", "")
        )

        if person or topic or title:
            lines.append(
                f"- Kişi: {person} | "
                f"Konu: {topic} | "
                f"Başlık: {title}"
            )

    if not lines:
        return "Henüz kullanılmamış konu yok."

    return "\n".join(lines)


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
            tag = clean_text(tag)

            if (
                tag
                and tag not in cleaned
            ):
                cleaned.append(tag)

    fallback_tags = [
        "Kayıp Hikâyeler",
        "Gerçek Hikâyeler",
        "Ünlülerin Hikâyeleri",
        "Türkiye",
        "Fenomenler",
        "Perde Arkası",
        "Belgesel"
    ]

    for tag in fallback_tags:
        if len(cleaned) >= 7:
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
# GEMINI İSTEKLERİ
# ============================================================


def gemini_json(
    prompt,
    schema,
    temperature=0.7
):
    last_error = None

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature
                )
            )

            return json.loads(
                response.text
            )

        except Exception as error:
            last_error = str(error)

            print(
                "Gemini JSON hata:",
                last_error
            )

            wait = 8 * (
                attempt + 1
            )

            print(
                f"{wait} saniye bekleniyor..."
            )

            time.sleep(wait)

    raise RuntimeError(
        "Gemini JSON üretilemedi: "
        + str(last_error)
    )


def grounded_research(prompt):
    last_error = None

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            google_search=types.GoogleSearch()
                        )
                    ],
                    temperature=0.35
                )
            )

            text = str(
                response.text
            ).strip()

            if not text:
                raise RuntimeError(
                    "Araştırma sonucu boş"
                )

            return text

        except Exception as error:
            last_error = str(error)

            print(
                "Araştırma hata:",
                last_error
            )

            time.sleep(
                8 * (attempt + 1)
            )

    raise RuntimeError(
        "Gerçek olay araştırılamadı: "
        + str(last_error)
    )


# ============================================================
# GERÇEK KONU ARAŞTIRMA
# ============================================================


def research_topic(history):

    previous = history_summary(
        history
    )

    prompt = f"""
Türkiye'de kamuoyunda konuşulmuş gerçek ve doğrulanabilir bir
ünlü, sanatçı, oyuncu, şarkıcı veya sosyal medya fenomeni hikâyesi
araştır.

AMAÇ:
"Kayıp Hikâyeler" YouTube kanalı için güçlü merak duygusu taşıyan,
belgesel gibi anlatılabilecek gerçek bir hikâye bulmak.

ÖNCELİK VER:
- Bir anda herkesin konuştuğu gerçek olaylar
- Şöhret yolunda yaşanmış şaşırtıcı gerçekler
- Kamuoyunda büyük merak uyandıran dönüm noktaları
- Yıllar sonra açıklığa kavuşan gerçek olaylar
- Kariyeri bir gecede değişen isimler
- Kayıp, bulunma, ortadan kaybolma gibi gerçek olaylar
- Herkesin bildiğini sandığı fakat perde arkasında başka gerçekler
- Bir açıklama, görüntü veya belgeyle ortaya çıkan gerçekler
- Türkiye'de geniş biçimde haber olmuş olaylar

KESİNLİKLE YAPMA:
- Söylentiyi gerçek gibi sunma
- Aldatma, suç, uyuşturucu, cinsel hayat veya benzeri hassas
  iddiaları doğrulanmadan anlatma
- Özel hayatı gereksiz şekilde kurcalama
- İftira niteliğinde iddia üretme
- Sadece magazin kavgasını konu seçme
- Aynı kişinin hayatını tekrar tekrar seçme

Aşağıdaki daha önce kullanılan konuları TEKRAR SEÇME:

{previous}

Google Search ile araştırma yap.

Sonra SADECE aşağıdaki yapıda JSON döndür:

{{
  "person": "kişinin adı",
  "topic": "tek cümlelik gerçek olay",
  "topic_key": "kişiyi ve olayı ayırt eden kısa anahtar",
  "why_interesting": "neden güçlü merak oluşturduğu",
  "verified_facts": [
    "doğrulanabilir gerçek 1",
    "doğrulanabilir gerçek 2",
    "doğrulanabilir gerçek 3",
    "doğrulanabilir gerçek 4"
  ],
  "timeline": [
    "önemli olay sırası 1",
    "önemli olay sırası 2",
    "önemli olay sırası 3"
  ],
  "avoid_claims": [
    "kanıtsız veya hassas şekilde anlatılmaması gereken iddia"
  ],
  "source_notes": [
    "araştırmada kullanılan güvenilir kaynak özeti"
  ]
}}

Eğer yeterince güvenilir gerçek bulamazsan başka kişi ve başka
olay araştır.

Sadece JSON döndür.
"""

    text = grounded_research(
        prompt
    )

    data = extract_json(text)

    required = [
        "person",
        "topic",
        "topic_key",
        "why_interesting",
        "verified_facts",
        "timeline",
        "avoid_claims",
        "source_notes"
    ]

    for field in required:
        if field not in data:
            raise ValueError(
                "Araştırmada eksik alan: "
                + field
            )

    data["person"] = clean_text(
        data["person"]
    )

    data["topic"] = clean_text(
        data["topic"]
    )

    data["topic_key"] = clean_text(
        data["topic_key"]
    )

    if (
        not data["person"]
        or not data["topic"]
        or not data["topic_key"]
    ):
        raise ValueError(
            "Araştırma konusu eksik"
        )

    return data


# ============================================================
# SENARYO PLANI
# ============================================================


def plan():

    history = load_history()

    last_error = None

    for attempt in range(8):

        try:
            print(
                f"Konu araştırılıyor... "
                f"{attempt + 1}/8"
            )

            research = research_topic(
                history
            )

            previous = history_summary(
                history
            )

            prompt = f"""
"Kayıp Hikâyeler" adlı Türkçe YouTube kanalı için
TEK bir ana video ve ona bağlı TEK bir Short hazırla.

KANALIN YENİ KONSEPTİ:

Türkiye'de kamuoyunda konuşulmuş ünlüler, sanatçılar, oyuncular,
şarkıcılar ve fenomenlerin GERÇEK, doğrulanabilir ve merak
uyandıran hikâyeleri.

Bu bir sıradan magazin kanalı değildir.

Anlatım tarzı:
Sinematik, güçlü, merak uyandırıcı, belgesel gibi ve doğal.

ARAŞTIRILMIŞ KONU:

Kişi:
{research["person"]}

Ana olay:
{research["topic"]}

Neden ilgi çekici:
{research["why_interesting"]}

Doğrulanabilir gerçekler:
{json.dumps(research["verified_facts"], ensure_ascii=False)}

Olay sırası:
{json.dumps(research["timeline"], ensure_ascii=False)}

Kaçınılacak iddialar:
{json.dumps(research["avoid_claims"], ensure_ascii=False)}

Kaynak notları:
{json.dumps(research["source_notes"], ensure_ascii=False)}

ÖNCEKİ VİDEOLAR:
Aşağıdaki kişi, olay, başlık ve konu kalıplarını tekrar etme:

{previous}

ÇOK ÖNEMLİ:

Sadece araştırmada verilen doğrulanabilir gerçeklere dayan.

Araştırmada olmayan bir bilgiyi kesin gerçek gibi ekleme.

Söylenti, dedikodu ve kanıtsız iddiaları gerçek gibi anlatma.

"İddia edildi" diyerek kanıtsız bir bilgiyi uzatma.

Hikâyenin gücü gerçek olayın kendisinden gelsin.

============================================================

ANA VİDEO ANLATIMI

Ana video yaklaşık 6-8 dakikalık doğal bir anlatım olacak.

NARRATION yaklaşık 850-1250 Türkçe kelime arasında olsun.

Kesin süre doldurmak için boş tekrar yapma.

Ancak 1-2 dakikalık kısa bir metin de üretme.

İlk 5 saniye:
Doğrudan en şaşırtıcı veya en merak uyandıran gerçekle başla.

Selamlama yok.
Kanal tanıtımı yok.
"Bugün sizlere" yok.
"Bu videoda" yok.

İLK CÜMLEDE KİŞİNİN ADINI HEMEN SÖYLEMEK ZORUNDA DEĞİLSİN.

Önce merak oluşturabilirsin.

Örnek mantık:

"Türkiye onu bir gecede konuşmaya başladı.
Ama herkesin gördüğü olayın arkasında, çok daha önce başlayan
başka bir hikâye vardı."

Bu sadece anlatım mantığıdır.
Aynısını kullanma.

YAPI:

1. Şok veya güçlü merakla açılış
2. İzleyicinin bilmediği veya unutmuş olabileceği arka plan
3. Olayın başlaması
4. Gerilim veya merakın yükselmesi
5. Herkesin dikkatini çeken dönüm noktası
6. Perde arkasındaki doğrulanabilir gerçekler
7. Olayın nasıl geliştiği
8. Sonuç
9. Kısa ve etkileyici kapanış

Her 20-40 saniyede yeni bir bilgi, soru veya gelişme gelsin.

Aynı bilgiyi farklı cümlelerle tekrar etme.

"Yıllarca sustu",
"kimse bilmiyordu",
"herkes şok oldu",
"12 yıl kayboldu"
gibi kalıpları her videoda otomatik kullanma.

Bu hikâyeye özel, özgün cümleler kur.

============================================================

SHORT

Short ana videodaki AYNI GERÇEK HİKÂYEYE bağlı olacak.

Ama ana videonun kopyası olmayacak.

SHORT NARRATION yaklaşık 60-115 kelime olsun.

Hedef:
Yaklaşık 25-45 saniye.

İlk 2 saniyede güçlü merak oluştur.

İlk cümle mümkün olduğunca güçlü olsun.

Short:
- Tek bir güçlü anı anlatabilir
- En şaşırtıcı dönüm noktasını kullanabilir
- Sonucu tamamen vermek zorunda değildir
- İzleyiciyi ana hikâyeyi merak etmeye yöneltebilir

Ancak yanıltıcı veya yalan bilgi kullanma.

============================================================

BAŞLIK KURALLARI

Ana video başlığı:
- 45-85 karakter arası tercih et
- Kişiye ve bu olaya özel olsun
- Önceki başlıklara benzemesin
- Sürekli sayı kullanma
- Sürekli "yıllarca", "sır", "neden sustu" kalıbını kullanma
- Clickbait olabilir ama yalan söyleme

Başlık çeşitliliği kullan:

Bazen:
- Bir olay üzerinden

Bazen:
- Beklenmedik dönüm noktası üzerinden

Bazen:
- Kişinin hayatındaki büyük değişim üzerinden

Bazen:
- Herkesin konuştuğu bir gece veya an üzerinden

Bazen:
- Ortaya çıkan gerçek üzerinden

Her videoda aynı başlık yapısını kullanma.

============================================================

PEXELS GÖRSELLERİ

Ana video için TAM OLARAK 12 İngilizce sorgu üret.

Short için TAM OLARAK 6 İngilizce sorgu üret.

Her sorgu:
- İngilizce
- 2-6 kelime
- Pexels'te bulunabilecek kadar genel
- Gerçekçi
- Hikâyeye ve sahneye uygun

Ünlünün gerçek fotoğrafını aramak zorunda değilsin.

Telifsiz ve genel atmosfer görüntüleri kullan.

Örnek:
concert crowd lights
microphone backstage
dark television studio
old newspaper closeup
social media phone screen
empty dressing room

Bunlar sadece örnektir.

Aynı sorguyu tekrar etme.

Thumbnail için tek bir İngilizce Pexels sorgusu üret.

============================================================

AÇIKLAMA

Kısa, doğal ve merak uyandırıcı olsun.

Etiket:
En az 5, en fazla 15 adet.

============================================================

NARRATION içinde:
- Başlık yazma
- Kaynak listesi yazma
- Bölüm numarası yazma
- "Kaynak:" yazma
- JSON yazma

SADECE aşağıdaki geçerli JSON yapısını üret:

title
description
tags
narration
short_title
short_narration
scene_queries
short_queries
thumbnail_query
person
topic
topic_key

Sadece JSON döndür.
"""

            schema = {
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
                    "person": {
                        "type": "STRING"
                    },
                    "topic": {
                        "type": "STRING"
                    },
                    "topic_key": {
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
                    "thumbnail_query",
                    "person",
                    "topic",
                    "topic_key"
                ]
            }

            data = gemini_json(
                prompt,
                schema,
                temperature=0.72
            )

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
                "person",
                "topic",
                "topic_key"
            ]

            for field in required:
                if field not in data:
                    raise ValueError(
                        "Eksik alan: "
                        + field
                    )

            for field in [
                "title",
                "description",
                "narration",
                "short_title",
                "short_narration",
                "thumbnail_query",
                "person",
                "topic",
                "topic_key"
            ]:
                data[field] = clean_text(
                    data[field]
                )

                if not data[field]:
                    raise ValueError(
                        "Boş alan: "
                        + field
                    )

            main_words = wc(
                data["narration"]
            )

            short_words = wc(
                data["short_narration"]
            )

            if main_words < MAIN_MIN_WORDS:
                raise ValueError(
                    f"Ana anlatım kısa: "
                    f"{main_words} kelime"
                )

            if main_words > MAIN_MAX_WORDS:
                raise ValueError(
                    f"Ana anlatım fazla uzun: "
                    f"{main_words} kelime"
                )

            if short_words < SHORT_MIN_WORDS:
                raise ValueError(
                    f"Short kısa: "
                    f"{short_words} kelime"
                )

            if short_words > SHORT_MAX_WORDS:
                raise ValueError(
                    f"Short fazla uzun: "
                    f"{short_words} kelime"
                )

            old_keys = {
                clean_text(
                    item.get(
                        "topic_key",
                        ""
                    )
                ).lower()
                for item in history[-30:]
            }

            if (
                data["topic_key"].lower()
                in old_keys
            ):
                raise ValueError(
                    "Aynı konu tekrar seçildi"
                )

            data["scene_queries"] = (
                normalize_queries(
                    data["scene_queries"],
                    12,
                    "documentary investigation"
                )
            )

            data["short_queries"] = (
                normalize_queries(
                    data["short_queries"],
                    6,
                    "mysterious documentary"
                )
            )

            data["thumbnail_query"] = (
                clean_query(
                    data["thumbnail_query"]
                )
            )

            if not data["thumbnail_query"]:
                data["thumbnail_query"] = (
                    "dramatic documentary portrait"
                )

            data["tags"] = normalize_tags(
                data["tags"]
            )

            data["_research"] = research

            print("=" * 60)
            print("PLAN KABUL EDİLDİ")
            print("=" * 60)
            print(
                "Kişi:",
                data["person"]
            )
            print(
                "Konu:",
                data["topic"]
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
                f"Plan reddedildi "
                f"{attempt + 1}/8: "
                f"{last_error}"
            )

            if attempt < 7:
                time.sleep(5)

    raise RuntimeError(
        "Geçerli ve farklı plan üretilemedi. "
        "Son hata: "
        + str(last_error)
    )


# ============================================================
# PEXELS
# ============================================================


def pexels(query, orientation, used):

    original_query = clean_query(
        query
    )

    fallback_queries = [
        original_query,
        "documentary portrait",
        "dramatic person",
        "television studio",
        "concert crowd"
    ]

    last_error = None

    for search_query in fallback_queries:

        if not search_query:
            continue

        try:
            response = requests.get(
                "https://api.pexels.com/v1/search",
                headers={
                    "Authorization": PEXELS
                },
                params={
                    "query": search_query,
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
                continue

            photo = next(
                (
                    item
                    for item in photos
                    if item["id"] not in used
                ),
                photos[0]
            )

            used.add(
                photo["id"]
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

            if image_url:
                return photo, image_url

        except Exception as error:
            last_error = str(error)

    raise RuntimeError(
        "Pexels görseli bulunamadı: "
        + original_query
        + " | "
        + str(last_error)
    )


def valid_image(path):
    try:
        if (
            not path.exists()
            or path.stat().st_size < 5000
        ):
            return False

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(path)
            ],
            capture_output=True,
            text=True
        )

        return result.returncode == 0

    except Exception:
        return False


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

        success = False
        last_error = None

        for attempt in range(4):

            try:
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

                if not valid_image(
                    file_path
                ):
                    raise RuntimeError(
                        "Bozuk görsel indirildi"
                    )

                files.append(
                    file_path
                )

                credits.append(
                    photo
                )

                success = True
                break

            except Exception as error:
                last_error = str(error)

                time.sleep(
                    2 * (attempt + 1)
                )

        if not success:
            raise RuntimeError(
                f"Görsel indirilemedi: "
                f"{query} | {last_error}"
            )

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
                "0.20",
                "--length-scale",
                "0.92",
                "--noise-scale",
                "0.667",
                "--noise-w",
                "0.8"
            ],
            chunk
        )

        wavs.append(
            wav
        )

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
# VIDEO
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

    for image_file in image_files:
        if not valid_image(
            image_file
        ):
            raise RuntimeError(
                "Geçersiz görsel: "
                + str(image_file)
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
            f"duration {each_duration:.3f}"
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
            "setsar=1,"
            "format=yuv420p"
        )

    else:

        video_filter = (
            "scale=1280:720:"
            "force_original_aspect_ratio=increase,"
            "crop=1280:720,"
            "setsar=1,"
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
            "-pix_fmt",
            "yuv420p",
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

    if not output.exists():
        raise RuntimeError(
            "Video dosyası oluşmadı"
        )

    if output.stat().st_size < 100_000:
        raise RuntimeError(
            "Video dosyası olağan dışı küçük"
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
            "description": str(
                description
            )[:5000],
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
        and valid_image(thumbnail)
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
                "crop=1280:720,"
                "setsar=1"
            ),
            "-q:v",
            "7",
            str(output)
        ]
    )

    if not valid_image(
        output
    ):
        raise RuntimeError(
            "Kapak görseli oluşturulamadı"
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
    print(
        "KAYIP HİKAYELER"
    )
    print(
        "TÜRKİYE'NİN KONUŞTUĞU"
        " GERÇEK HİKAYELER"
    )
    print("=" * 60)

    setup()

    data = plan()

    print("=" * 60)
    print("SEÇİLEN KİŞİ:")
    print(
        data["person"]
    )
    print("=" * 60)

    print("KONU:")
    print(
        data["topic"]
    )
    print("=" * 60)

    print("ANA BAŞLIK:")
    print(
        data["title"]
    )

    print("SHORT BAŞLIK:")
    print(
        data["short_title"]
    )

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

    # Ana video yaklaşık 6-8 dakika hedeflenir.
    # Aşırı kısa veya aşırı uzun çıkarsa yükleme yapılmaz.
    if long_duration < MAIN_MIN_SECONDS:
        raise RuntimeError(
            "Ana video hedef süreden kısa: "
            + f"{long_duration:.1f} saniye"
        )

    if long_duration > MAIN_MAX_SECONDS:
        raise RuntimeError(
            "Ana video hedef süreden uzun: "
            + f"{long_duration:.1f} saniye"
        )

    # Short yaklaşık 25-45 saniye hedeflenir.
    # Güvenli tolerans: 22-55 saniye.
    if short_duration < SHORT_MIN_SECONDS:
        raise RuntimeError(
            "Short fazla kısa: "
            + f"{short_duration:.1f} saniye"
        )

    if short_duration > SHORT_MAX_SECONDS:
        raise RuntimeError(
            "Short fazla uzun: "
            + f"{short_duration:.1f} saniye"
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
        long_video,
        vertical=False
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

    thumbnail_images, thumbnail_credits = images(
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

    all_credits = (
        long_credits
        + short_credits
        + thumbnail_credits
    )

    for photo in all_credits:

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
        + "Bu video kamuya açık ve "
        + "doğrulanabilir bilgiler temel alınarak "
        + "belgesel anlatım formatında hazırlanmıştır."
        + "\n\n"
        + "Anlatım ve görsel düzenleme "
        + "yapay zekâ desteğiyle hazırlanmıştır."
    )

    if unique_credits:
        description += (
            "\n\nPexels kaynakları:\n"
            + "\n".join(
                unique_credits
            )
        )

    description += (
        "\n\n"
        + "#KayıpHikayeler "
        + "#GerçekHikayeler "
        + "#Türkiye "
        + "#Ünlüler "
        + "#Fenomenler"
    )

    print(
        "YouTube bağlantısı hazırlanıyor..."
    )

    api = youtube()

    print("=" * 60)
    print(
        "ANA VİDEO YÜKLENİYOR"
    )
    print("=" * 60)

    long_id = upload(
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

    short_id = upload(
        api,
        short_video,
        data["short_title"],
        description,
        data["tags"]
    )

    # ========================================================
    # SADECE İKİ VİDEO DA BAŞARIYLA YÜKLENDİKTEN SONRA
    # KONUYU GEÇMİŞE KAYDET.
    # Böylece bir sonraki çalıştırmada aynı kişi ve olay
    # tekrar seçilmeyecek.
    # ========================================================

    save_history(
        {
            "person": data["person"],
            "topic": data["topic"],
            "topic_key": data["topic_key"],
            "title": data["title"],
            "short_title": data["short_title"],
            "youtube_video_id": long_id,
            "youtube_short_id": short_id,
            "created_at": int(
                time.time()
            )
        }
    )

    print("=" * 60)
    print(
        "OTOMASYON BAŞARIYLA TAMAMLANDI"
    )
    print("=" * 60)

    print(
        "Ana video ID:",
        long_id
    )

    print(
        "Short ID:",
        short_id
    )

    print(
        "Konu geçmişe kaydedildi."
    )


if __name__ == "__main__":
    main()
