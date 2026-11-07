"""Beat-synced car montage renderer (music only, no voice, no captions).

Needs: ffmpeg, pip install librosa numpy
Env:   PEXELS_API_KEY (required), PIXABAY_API_KEY (optional second footage source)
Files: music/*.mp3|wav (royalty-free tracks from YouTube Audio Library)
       used_clips.json (auto-maintained: IDs of clips already used in past videos)
Out:   out.mp4 (1080x1920, ~28s), meta.json (title/description/tags), used_clips.json

No-repeat rules:
  * every clip ID that makes it into a video is remembered and skipped in future renders
  * inside one video each clip is used for one cut only (reused only if footage runs out)
  * clips are only accepted if their title/tags say "car" and nothing like bus/truck/van
  * max 1 clip per videographer per video, and a cap per car (color + brand) so the
    same car doesn't show up several times in one edit
"""
import glob, json, os, random, re, subprocess, sys, urllib.parse, urllib.request
from collections import Counter

import librosa
import numpy as np

PEXELS_KEY = os.environ["PEXELS_API_KEY"]
PIXABAY_KEY = os.environ.get("PIXABAY_API_KEY", "").strip()

QUERIES = [
    "supercar driving", "hypercar road", "sports car driving", "sports car night",
    "supercar night city", "luxury sports car", "muscle car", "muscle car burnout",
    "drift car", "car drifting smoke", "race car track", "race car circuit",
    "rally car gravel", "car exhaust flames", "car backfire fire", "car burnout smoke",
    "drag race car", "tuner car night", "jdm car", "lamborghini", "ferrari",
    "porsche 911", "mclaren", "bugatti", "nissan gtr", "ford mustang", "corvette",
    "bmw m3", "mercedes amg", "audi r8", "dodge charger", "toyota supra",
    "car meet night", "sports car rolling shot", "convertible sports car",
    "sports car mountain road", "car racing circuit", "supercar tunnel",
]
QUERIES_PER_RUN = 12     # random queries searched per render
MAX_PAGE = 6             # random result page (1..MAX_PAGE) per query
PER_QUERY_MAX = 4        # max clips from one query in one video
MAX_USED = 3000          # how many past clip IDs to remember
USED_FILE = "used_clips.json"

TARGET = 28          # seconds
AVG_CUT = 1.2        # average seconds per shot
WORK = "work"
GRADE = ("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
         "eq=contrast=1.15:saturation=1.3:brightness=-0.02,fps=30,setsar=1,format=yuv420p")

TITLES = [
    "Night drive hits different 🏁", "Sound on. Lock in. 🔥", "Built for the apex 🏎️",
    "This is what speed feels like", "Drift season never ends 💨",
]
TAGS = ["shorts", "cars", "drift", "racing", "supercar", "caredit", "aura"]

# ---- clip filtering -------------------------------------------------------------
# A clip is kept only if its title/tags contain a car word and no "junk" word.
BRANDS = {
    "ferrari", "lamborghini", "porsche", "mclaren", "bugatti", "koenigsegg", "pagani",
    "maserati", "bentley", "mustang", "camaro", "corvette", "hellcat", "supra", "gtr",
    "nissan", "bmw", "mercedes", "amg", "audi", "lexus", "tesla", "subaru", "wrx",
    "mazda", "rx7", "huracan", "aventador", "urus", "chiron", "veyron", "gt3", "r8",
    "dodge", "ford", "chevrolet", "chevy", "toyota", "honda",
}   # ambiguous words (skyline, lotus, viper, charger, civic...) are left out on purpose
COLORS = {
    "red", "blue", "black", "white", "yellow", "orange", "green", "silver", "grey",
    "gray", "purple", "pink", "gold", "bronze", "brown",
}
CAR_TOKENS = BRANDS | {
    "car", "cars", "supercar", "supercars", "hypercar", "hypercars", "racecar",
    "coupe", "roadster", "convertible", "muscle", "drift", "drifting", "burnout",
    "burnouts", "rally", "jdm",
}
BAD_TOKENS = {
    "bus", "buses", "school", "truck", "trucks", "van", "vans", "minivan", "suv",
    "pickup", "tractor", "tractors", "trailer", "forklift", "excavator", "bulldozer",
    "construction", "garbage", "bicycle", "bicycles", "bike", "bikes", "motorcycle",
    "motorcycles", "motorbike", "scooter", "train", "trains", "tram", "taxi", "taxis",
    "ambulance", "police", "firetruck", "traffic", "parking", "parked", "wash",
    "washing", "repair", "mechanic", "toy", "toys", "miniature", "model", "rc", "lego",
    "game", "gaming", "cartoon", "animation", "illustration", "kart", "karting",
    "golf", "cart", "airplane", "plane", "boat", "ship", "horse", "interior",
    "dashboard", "steering", "phone", "smartphone", "laptop", "office", "woman", "man",
    "child", "kids", "people", "family", "couple",
}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"failed: {' '.join(cmd[:4])}")


UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def tokens_of(*parts):
    text = " ".join(str(p) for p in parts if p)
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def is_car_clip(tokens):
    return bool(tokens & CAR_TOKENS) and not (tokens & BAD_TOKENS)


def slug_of(url):
    return urllib.parse.urlparse(url or "").path.strip("/").split("/")[-1]


# ---- used-clip tracker ------------------------------------------------------------
def load_used():
    try:
        with open(USED_FILE) as f:
            data = json.load(f)
        return [str(x) for x in data] if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_used(old, new_ids):
    fresh = set(new_ids)
    merged = [x for x in old if x not in fresh] + list(new_ids)   # newest last
    with open(USED_FILE, "w") as f:
        json.dump(merged[-MAX_USED:], f)


# ---- footage sources --------------------------------------------------------------
def http_json(url, headers):
    req = urllib.request.Request(url, headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=30))


def fetch_pexels(query, orientation):
    """Random result page first (variety); fall back to page 1."""
    for page in (random.randint(1, MAX_PAGE), 1):
        q = urllib.parse.urlencode({"query": query, "orientation": orientation,
                                    "per_page": 40, "page": page})
        try:
            vids = http_json(f"https://api.pexels.com/videos/search?{q}",
                             {"Authorization": PEXELS_KEY, "User-Agent": UA}).get("videos", [])
            if vids:
                return vids
        except Exception as e:
            print("pexels fail", query, orientation, page, e)
    return []


def fetch_pixabay(query):
    for page in (random.randint(1, MAX_PAGE), 1):
        q = urllib.parse.urlencode({"key": PIXABAY_KEY, "q": query, "per_page": 50,
                                    "page": page, "video_type": "film",
                                    "category": "transportation", "safesearch": "true",
                                    "order": random.choice(["popular", "latest"])})
        try:
            hits = http_json(f"https://pixabay.com/api/videos/?{q}",
                             {"User-Agent": UA}).get("hits", [])
            if hits:
                return hits
        except Exception as e:
            print("pixabay fail", query, page, e)
    return []


def best_file(video):
    files = [f for f in video["video_files"] if f.get("file_type") == "video/mp4"]
    ok = [f for f in files if (f.get("height") or 0) >= 1080]
    pick = min(ok, key=lambda f: f["height"]) if ok else max(files, key=lambda f: f.get("height") or 0)
    return pick["link"]


def pixabay_link(hit):
    vf = hit.get("videos") or {}
    sized = [vf[k] for k in ("large", "medium", "small") if (vf.get(k) or {}).get("url")]
    ok = [v for v in sized if 1080 <= (v.get("height") or 0) <= 2160
          and (v.get("size") or 0) < 60_000_000]
    if ok:
        return min(ok, key=lambda v: v["height"])["url"]
    return sized[0]["url"] if sized else None


def collect_pool():
    pool, rejected = {}, 0
    for q in random.sample(QUERIES, min(QUERIES_PER_RUN, len(QUERIES))):
        for orient in ("portrait", "landscape"):
            for v in fetch_pexels(q, orient):
                if (v.get("duration") or 0) < 4:
                    continue
                toks = tokens_of(slug_of(v.get("url")), " ".join(map(str, v.get("tags") or [])))
                if not is_car_clip(toks):
                    rejected += 1
                    continue
                cid = f"px{v['id']}"
                if cid in pool:
                    continue
                try:
                    link = best_file(v)
                except Exception:
                    continue
                pool[cid] = dict(id=cid, dur=float(v["duration"]), url=link,
                                 user=f"px{(v.get('user') or {}).get('id')}",
                                 sig=frozenset(toks & (BRANDS | COLORS)), query=q)
        if PIXABAY_KEY:
            for h in fetch_pixabay(q):
                if (h.get("duration") or 0) < 4:
                    continue
                toks = tokens_of(h.get("tags"), slug_of(h.get("pageURL")))
                if not is_car_clip(toks):
                    rejected += 1
                    continue
                cid = f"pb{h['id']}"
                link = pixabay_link(h)
                if cid in pool or not link:
                    continue
                pool[cid] = dict(id=cid, dur=float(h["duration"]), url=link,
                                 user=f"pb{h.get('user_id')}",
                                 sig=frozenset(toks & (BRANDS | COLORS)), query=q)
    print(f"pool: {len(pool)} car clips ({rejected} rejected as not-car/junk)",
          "| pixabay on" if PIXABAY_KEY else "| pixabay off (no PIXABAY_API_KEY)")
    return list(pool.values())


# ---- picking clips ----------------------------------------------------------------
def diverse(cands, n, strict=True):
    """Pick up to n clips. strict = one per videographer, capped per car and per query."""
    chosen, users, cars, queries = [], Counter(), Counter(), Counter()
    for c in cands:
        if len(chosen) >= n:
            break
        if strict:
            if users[c["user"]] >= 1:
                continue
            if c["sig"] and cars[c["sig"]] >= (1 if len(c["sig"]) >= 2 else 2):
                continue
            if queries[c["query"]] >= PER_QUERY_MAX:
                continue
        chosen.append(c)
        users[c["user"]] += 1
        cars[c["sig"]] += 1
        queries[c["query"]] += 1
    return chosen


def pick_clips(pool, n, used):
    """Never-used clips first (diverse), then relaxed, then oldest-used as a last resort."""
    rank = {cid: i for i, cid in enumerate(used)}
    pool = pool[:]
    random.shuffle(pool)
    fresh = [c for c in pool if c["id"] not in rank]
    stale = sorted((c for c in pool if c["id"] in rank), key=lambda c: rank[c["id"]])

    chosen = diverse(fresh, n)
    ids = {c["id"] for c in chosen}
    if len(chosen) < n:
        chosen += diverse([c for c in fresh if c["id"] not in ids], n - len(chosen), strict=False)
    if len(chosen) < n:
        print("WARNING: ran out of never-used clips, reusing oldest-used ones")
        chosen += diverse(stale, n - len(chosen), strict=False)
    print(f"picked {len(chosen)} clips ({len(fresh)} fresh in pool)")
    return chosen


def download_clips(ordered, n):
    os.makedirs(WORK, exist_ok=True)
    clips = []
    for c in ordered:
        if len(clips) >= n:
            break
        path = f"{WORK}/clip_{c['id']}.mp4"
        if not os.path.exists(path):
            try:
                dl = urllib.request.Request(c["url"], headers={"User-Agent": UA})
                with urllib.request.urlopen(dl, timeout=60) as r, open(path, "wb") as f:
                    f.write(r.read())
            except Exception as e:
                print("download fail", c["id"], e)
                continue
        clips.append(dict(c, path=path))
    return clips


def get_clips(n, used):
    pool = collect_pool()
    ordered = pick_clips(pool, n + 6, used)      # +6 spares in case downloads fail
    clips = download_clips(ordered, n)
    if len(clips) < 4:
        raise SystemExit("not enough clips")
    return clips


def assign_clips(durs, clips):
    """One distinct clip per cut. Only if footage runs out are clips reused (never back-to-back)."""
    unused = clips[:]
    random.shuffle(unused)
    spent, out, last = [], [], None
    for dur in durs:
        fits = [c for c in unused if c["dur"] > dur + 0.5]
        if fits:
            c = random.choice(fits)
            unused.remove(c)
        else:
            options = ([c for c in spent if c is not last and c["dur"] > dur + 0.5]
                       or [c for c in clips if c is not last] or clips)
            c = random.choice(options)
        if not any(c is s for s in spent):
            spent.append(c)
        out.append(c)
        last = c
    return out


# ---- audio / cuts -----------------------------------------------------------------
def plan_cuts(track):
    y, sr = librosa.load(track, duration=240)
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    times = librosa.frames_to_time(frames, sr=sr)
    total = len(y) / sr
    if total < TARGET + 2:
        raise SystemExit("track too short")
    starts = [t for t in times if t <= total - TARGET - 1]
    start = float(random.choice(starts[: max(1, len(starts) // 2)]))
    step = max(1, round(AVG_CUT / (60 / tempo)))
    window = [float(t) for t in times if start <= t < start + TARGET]
    cuts = window[::step] + [start + TARGET]
    return start, cuts


def main():
    tracks = glob.glob("music/*.mp3") + glob.glob("music/*.wav")
    if not tracks:
        raise SystemExit("put royalty-free tracks in music/")
    track = random.choice(tracks)
    start, cuts = plan_cuts(track)

    shots = [(i, a, b - a) for i, (a, b) in enumerate(zip(cuts, cuts[1:])) if b - a >= 0.25]
    used = load_used()
    clips = get_clips(len(shots) + 2, used)
    chosen = assign_clips([d for _, _, d in shots], clips)

    segs = []
    for (i, _, dur), c in zip(shots, chosen):
        ss = random.uniform(0, max(0, c["dur"] - dur - 0.2))
        seg = f"{WORK}/seg_{i:03d}.mp4"
        run(["ffmpeg", "-y", "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", c["path"],
             "-an", "-vf", GRADE, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", seg])
        segs.append(seg)

    with open(f"{WORK}/list.txt", "w") as f:
        f.writelines(f"file '{os.path.abspath(s)}'\n" for s in segs)

    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", f"{WORK}/list.txt",
         "-ss", f"{start:.3f}", "-t", str(TARGET), "-i", track,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy",
         "-af", f"afade=t=out:st={TARGET - 1.5}:d=1.5", "-c:a", "aac", "-b:a", "192k",
         "-shortest", "out.mp4"])

    with open("meta.json", "w") as f:
        json.dump({"title": random.choice(TITLES) + " #shorts",
                   "description": "Music-only car edit. #cars #drift #racing #supercar",
                   "tags": TAGS}, f, ensure_ascii=False)

    # remember what this video used so the next render never repeats it
    used_now = list(dict.fromkeys(c["id"] for c in chosen))
    save_used(used, used_now)
    print(f"done: out.mp4, {len(segs)} cuts, {len(used_now)} distinct clips")


if __name__ == "__main__":
    main()
