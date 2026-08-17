import os,re,json,time,subprocess
from pathlib import Path
import requests
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

OUT=Path("work"); OUT.mkdir(exist_ok=True)
GEMINI=os.environ["GEMINI_API_KEY"]
PEXELS=os.environ["PEXELS_API_KEY"]
MODEL="gemini-3.1-flash-lite"
client=genai.Client(api_key=GEMINI)

PIPER=OUT/"tr_TR-dfki-medium.onnx"
PIPER_CFG=OUT/"tr_TR-dfki-medium.onnx.json"
PIPER_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/tr/tr_TR/dfki/medium/"

def run(c,inp=None):
    print("$"," ".join(map(str,c)))
    subprocess.run(c,input=inp,text=True,check=True)

def wc(s):
    return len(re.findall(r"\b[\wÇĞİÖŞÜçğıöşü'-]+\b",str(s)))

def dur(f):
    return float(subprocess.check_output([
        "ffprobe","-v","error","-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",str(f)
    ],text=True).strip())

def get(url,path,minimum=500):
    if path.exists() and path.stat().st_size>=minimum:return
    r=requests.get(url,timeout=180);r.raise_for_status()
    path.write_bytes(r.content)
    if path.stat().st_size<minimum:raise RuntimeError("Eksik dosya: "+str(path))

def setup():
    get(PIPER_URL+"tr_TR-dfki-medium.onnx",PIPER,50_000_000)
    get(PIPER_URL+"tr_TR-dfki-medium.onnx.json",PIPER_CFG,500)

def gemini(prompt):
    for n in range(3):
        try:
            r=client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type":"OBJECT",
                        "properties":{
                            "title":{"type":"STRING"},
                            "description":{"type":"STRING"},
                            "tags":{
                                "type":"ARRAY",
                                "items":{"type":"STRING"}
                            },
                            "narration":{"type":"STRING"},
                            "short_title":{"type":"STRING"},
                            "short_narration":{"type":"STRING"},
                            "scene_queries":{
                                "type":"ARRAY",
                                "items":{"type":"STRING"}
                            },
                            "short_queries":{
                                "type":"ARRAY",
                                "items":{"type":"STRING"}
                            },
                            "thumbnail_query":{"type":"STRING"}
                        },
                        "required":[
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
                    temperature=.7
                )
            )

            return json.loads(r.text)

        except Exception as e:
            msg=str(e)
            print("Gemini:",msg)

            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait=20*(n+1)
                print(f"Gemini kotası/hız sınırı. {wait} saniye bekleniyor...")
                time.sleep(wait)
            else:
                if n<2:
                    time.sleep(5)

    raise RuntimeError("Gemini içerik üretilemedi")

def plan():
    p=r"""
Türkçe YouTube belgesel kanalı "Kayıp Hikâyeler" için TEK bölüm hazırla.

Konu gizem, tarih, arkeoloji, bilim, uzay veya şaşırtıcı ama
doğrulanabilir evergreen bir gerçek olsun.

GOOGLE SEARCH İLE ARAŞTIR.
Güvenilir üniversite, müze, devlet kurumu, bilimsel kurum ve
güvenilir haber kaynaklarını tercih et.
Uydurma bilgi, tarih, sayı, isim veya alıntı kullanma.
Belirsiz iddiaları kesin gerçek gibi anlatma.

ANA VİDEO 1150-1350 TÜRKÇE KELİME OLSUN.
Yaklaşık 7-10 dakika.

İLK 15 SANİYE ÇOK GÜÇLÜ OLSUN:
İlk cümle doğrudan gizemli/şaşırtıcı olayla başlasın.
Selamlama, kanal tanıtımı ve "bugün sizlere" kullanma.
İlk 5 saniyede merak oluştur.
İlk 15 saniyede büyük soruyu kur.
İzleyici cevabı öğrenmek için videoda kalmak istesin.

Yapı:
güçlü açılış, arka plan, kronoloji, kanıtlar,
alternatif açıklamalar, en güçlü bulgular, dengeli sonuç,
kısa kapanış.

Doğal Türkçe konuşma dili kullan.
Narration içinde başlık veya kaynak listesi yazma.

SHORT 100-160 kelime, 35-75 saniye.
Bağımsız anlaşılmalı ve ana videonun kopyası olmamalı.
Short da ilk cümlede merak uyandırmalı.

ANA VİDEO İÇİN TAM 12 İNGİLİZCE PEXELS SORGUSU.
SHORT İÇİN TAM 6 İNGİLİZCE PEXELS SORGUSU.
THUMBNAIL İÇİN 1 İNGİLİZCE PEXELS SORGUSU.

Her sorgu 2-6 kelime.
Türkçe karakter kullanma.
Gerçek yer, nesne, bina, insan, doğa, arkeolojik alan veya
bilimsel ekipmanı tarif et.

SADECE GEÇERLİ JSON DÖNDÜR.

ÇOK ÖNEMLİ:
- Hiçbir alan boş bırakılamaz.
- Örnek/şablon değerleri aynen kopyalama.
- Tüm alanları gerçek içerikle doldur.
- narration 1150-1350 Türkçe kelime olmalı.
- short_narration 100-160 Türkçe kelime olmalı.
- scene_queries TAM OLARAK 12 adet İngilizce Pexels arama sorgusu içermeli.
- short_queries TAM OLARAK 6 adet İngilizce Pexels arama sorgusu içermeli.
- thumbnail_query TAM OLARAK 1 adet İngilizce Pexels arama sorgusu içermeli.
- tags en az 5 alakalı etiket içermeli.
- title ve description mutlaka dolu olmalı.
- narration içinde başlık, kaynak listesi veya JSON bulunmamalı.
- Sadece JSON döndür. JSON dışında hiçbir açıklama yazma.

JSON alanları tam olarak şunlar olmalı:
title
description
tags
narration
short_title
short_narration
scene_queries
short_queries
thumbnail_query
"""
               last_error = "İlk üretim"

    for attempt in range(5):
        try:
            d = gemini(p)

            need = [
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

            if not all(x in d for x in need):
                raise ValueError("Alan eksik")

            if not 1150 <= wc(d["narration"]) <= 1350:
                raise ValueError(
                    f"Ana narration kelime sayısı: {wc(d['narration'])}"
                )

            if not 100 <= wc(d["short_narration"]) <= 160:
                raise ValueError(
                    f"Short kelime sayısı: {wc(d['short_narration'])}"
                )

            if len(d["scene_queries"]) != 12:
                raise ValueError("12 sahne gerekli")

            if len(d["short_queries"]) != 6:
                raise ValueError("6 short sahnesi gerekli")

            if not d["thumbnail_query"]:
                raise ValueError("Thumbnail sorgusu boş")

            return d

        except Exception as e:
            last_error = str(e)
            print("Plan kontrol:", last_error)

            if attempt < 4:
                time.sleep(3)

    raise RuntimeError("Geçerli plan üretilemedi: " + last_error)


def pexels(q, orientation, used):
    q = re.sub(r"[^A-Za-z0-9\s.'&-]", " ", str(q))
    q = re.sub(r"\s+", " ", q).strip() or "documentary"

    r = requests.get(
        "https://api.pexels.com/v1/search",
        headers={"Authorization": PEXELS},
        params={
            "query": q,
            "orientation": orientation,
            "per_page": 20
        },
        timeout=60
    )

    r.raise_for_status()
    photos = r.json().get("photos", [])

    if not photos:
        raise RuntimeError("Pexels sonuç yok: " + q)

    p = next(
        (x for x in photos if x["id"] not in used),
        photos[0]
    )

    used.add(p["id"])
    src = p["src"]

    u = (
        src.get("portrait")
        if orientation == "portrait"
        else src.get("landscape")
    ) or src.get("large2x") or src.get("original")

    return p, u


def images(queries, prefix, orientation):
    files = []
    credits = []
    used = set()

    for i, q in enumerate(queries, 1):
        p, u = pexels(q, orientation, used)

        f = OUT / f"{prefix}_{i:02d}.jpg"

        r = requests.get(u, timeout=90)
        r.raise_for_status()

        f.write_bytes(r.content)

        if f.stat().st_size < 5000:
            raise RuntimeError("Bozuk Pexels görseli")

        files.append(f)
        credits.append(p)

    return files, credits

            if attempt < 4:
                time.sleep(3)

    raise RuntimeError(
        f"5 denemede geçerli plan üretilemedi. Son hata: {last_error}"
    )

Ana narration TAM 1150-1350 Türkçe kelime olmalı.
short_narration TAM 100-160 Türkçe kelime olmalı.
scene_queries TAM 12 adet olmalı.
short_queries TAM 6 adet olmalı.
thumbnail_query kesinlikle dolu olmalı.
Tüm JSON alanları gerçek içerikle doldurulmalı.
Sadece geçerli JSON döndür.
"""

            d = gemini(p + extra)

            need = [
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

            if not all(x in d for x in need):
                raise ValueError("Alan eksik")

            if not 1150 <= wc(d["narration"]) <= 1350:
                raise ValueError(
                    f"Ana narration kelime sayısı: {wc(d['narration'])}"
                )

            if not 100 <= wc(d["short_narration"]) <= 160:
                raise ValueError(
                    f"Short kelime sayısı: {wc(d['short_narration'])}"
                )

            if len(d["scene_queries"]) != 12:
                raise ValueError(
                    f"12 sahne sorgusu gerekli, gelen: {len(d['scene_queries'])}"
                )

            if len(d["short_queries"]) != 6:
                raise ValueError(
                    f"6 short sorgusu gerekli, gelen: {len(d['short_queries'])}"
                )

            if not str(d["thumbnail_query"]).strip():
                raise ValueError("Thumbnail sorgusu boş")

            return d

        except Exception as e:
            last_error = str(e)

            print(
                f"Plan kontrolü {attempt + 1}/5: {last_error}"
            )

            if attempt < 4:
                time.sleep(3)

    raise RuntimeError(
        f"5 denemede geçerli plan üretilemedi. Son hata: {last_error}"
    )

def pexels(q,orientation,used):
    q=re.sub(r"[^A-Za-z0-9\s.'&-]"," ",str(q))
    q=re.sub(r"\s+"," ",q).strip() or "documentary"
    r=requests.get(
        "https://api.pexels.com/v1/search",
        headers={"Authorization":PEXELS},
        params={"query":q,"orientation":orientation,"per_page":20},
        timeout=60
    )
    r.raise_for_status()
    photos=r.json().get("photos",[])
    if not photos:raise RuntimeError("Pexels sonuç yok: "+q)
    p=next((x for x in photos if x["id"] not in used),photos[0])
    used.add(p["id"])
    src=p["src"]
    u=src.get("portrait" if orientation=="portrait" else "landscape") or src.get("large2x") or src.get("original")
    return p,u

def images(queries,prefix,orientation):
    files=[];credits=[];used=set()
    for i,q in enumerate(queries,1):
        p,u=pexels(q,orientation,used)
        f=OUT/f"{prefix}_{i:02}.jpg"
        r=requests.get(u,timeout=90);r.raise_for_status()
        f.write_bytes(r.content)
        if f.stat().st_size<5000:raise RuntimeError("Bozuk Pexels görseli")
        files.append(f);credits.append(p)
    return files,credits

def tts(text,out):
    text=re.sub(r"\s+"," ",str(text)).strip()
    chunks=[]
    while len(text)>2200:
        n=text.rfind(" ",0,2200)
        if n<500:n=2200
        chunks.append(text[:n]);text=text[n:].strip()
    if text:chunks.append(text)

    wavs=[]
    for i,x in enumerate(chunks):
        w=OUT/f"{out.stem}_{i:03}.wav"
        run([
            "python","-m","piper",
            "--model",str(PIPER),
            "--output_file",str(w),
            "--sentence-silence","0.18",
            "--length-scale","0.88",
            "--noise-scale","0.667",
            "--noise-w","0.8"
        ],x)
        wavs.append(w)

    lst=OUT/f"{out.stem}_concat.txt"
    lst.write_text(
        "\n".join("file '"+str(x.resolve()).replace("'","'\\''")+"'"
                  for x in wavs),
        encoding="utf-8"
    )
    run([
        "ffmpeg","-y","-f","concat","-safe","0",
        "-i",str(lst),"-c:a","libmp3lame","-b:a","160k",str(out)
    ])

def make_video(audio,imgs,out,vertical=False):
    total=dur(audio)
    each=total/len(imgs)
    if each<2:raise RuntimeError("Sahne süresi çok kısa")

    lst=OUT/f"{out.stem}_slides.txt"
    lines=[]
    for x in imgs:
        lines += [
            "file '"+str(x.resolve()).replace("'","'\\''")+"'",
            f"duration {each:.3f}"
        ]
    lines.append("file '"+str(imgs[-1].resolve()).replace("'","'\\''")+"'")
    lst.write_text("\n".join(lines),encoding="utf-8")

    if vertical:
        vf="scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,format=yuv420p"
    else:
        vf="scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,format=yuv420p"

    run([
        "ffmpeg","-y","-f","concat","-safe","0",
        "-i",str(lst),"-i",str(audio),
        "-vf",vf,"-r","25",
        "-c:v","libx264","-preset","veryfast","-crf","22",
        "-c:a","aac","-b:a","160k",
        "-shortest","-movflags","+faststart",str(out)
    ])

def yt():
    c=Credentials(
        token=None,
        refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"]
    )
    return build("youtube","v3",credentials=c,cache_discovery=False)

def upload(api,file,title,desc,tags,thumb=None):
    body={
        "snippet":{
            "title":str(title)[:100],
            "description":str(desc)[:5000],
            "tags":[str(x) for x in tags[:30]],
            "categoryId":"27"
        },
        "status":{
            "privacyStatus":"public",
            "selfDeclaredMadeForKids":False,
            "containsSyntheticMedia":True
        }
    }
    media=MediaFileUpload(
        str(file),mimetype="video/mp4",
        resumable=True,chunksize=8*1024*1024
    )
    req=api.videos().insert(
        part="snippet,status",body=body,media_body=media
    )
    res=None
    while res is None:
        st,res=req.next_chunk()
        if st:print("YouTube %d%%"%int(st.progress()*100))
    vid=res["id"]

    if thumb and thumb.exists():
        api.thumbnails().set(
            videoId=vid,
            media_body=MediaFileUpload(
                str(thumb),mimetype="image/jpeg"
            )
        ).execute()

    print("Yüklendi:",vid)
    return vid

def thumb(src,out):
    run([
        "ffmpeg","-y","-i",str(src),
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
        "-q:v","7",str(out)
    ])
    if out.stat().st_size>1_900_000:
        run([
            "ffmpeg","-y","-i",str(out),
            "-q:v","10",str(out)
        ])

def main():
    print("="*50)
    print("KAYIP HİKÂYELER OTOMASYONU")
    print("="*50)

    setup()
    d=plan()

    print("Konu:",d["title"])
    print("Ana kelime:",wc(d["narration"]))
    print("Short kelime:",wc(d["short_narration"]))

    long_audio=OUT/"long.mp3"
    short_audio=OUT/"short.mp3"

    tts(d["narration"],long_audio)
    tts(d["short_narration"],short_audio)

    ld=dur(long_audio)
    sd=dur(short_audio)

    print("Ana süre:",round(ld,1),"saniye")
    print("Short:",round(sd,1),"saniye")

    if not 420<=ld<=600:
        raise RuntimeError(f"Ana video 7-10 dakika dışında: {ld:.1f}s")
    if not 35<=sd<=75:
        raise RuntimeError(f"Short süresi dışında: {sd:.1f}s")

    li,lc=images(d["scene_queries"],"long","landscape")
    si,sc=images(d["short_queries"],"short","portrait")

    lv=OUT/"long.mp4"
    sv=OUT/"short.mp4"

    make_video(long_audio,li,lv)
    make_video(short_audio,si,sv,True)

    ti,_=images([d["thumbnail_query"]],"thumbnail","landscape")
    th=OUT/"thumbnail.jpg"
    thumb(ti[0],th)

    credits=[]
    for p in lc+sc:
        u=p.get("url","")
        name=p.get("photographer","Pexels photographer")
        if u:
            credits.append(f"Photo by {name} on Pexels: {u}")

    desc=(
        str(d["description"]).strip()+
        "\n\nBu video özgün senaryo, yapay zekâ destekli "
        "seslendirme ve Pexels görselleri kullanılarak hazırlanmıştır."
        "\n\nPexels kaynakları:\n"+
        "\n".join(dict.fromkeys(credits))+
        "\n\n#KayıpHikâyeler #Gizem #Belgesel"
    )

    api=yt()

    upload(
        api,lv,d["title"],desc,d["tags"],th
    )

    upload(
        api,sv,d["short_title"],desc,d["tags"]
    )

    print("="*50)
    print("OTOMASYON BAŞARIYLA TAMAMLANDI")
    print("="*50)

if __name__=="__main__":
    main()
