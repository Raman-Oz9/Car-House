"""Beat-synced car montage renderer (music only, no voice, no captions)."""
import glob, json, os, random, subprocess, sys, urllib.parse, urllib.request

import librosa
import numpy as np

PEXELS_KEY = os.environ["PEXELS_API_KEY"]
QUERIES = [
    "supercar", "hypercar", "sports car driving", "muscle car", "muscle car burnout",
    "sports car night", "supercar night city", "drift car", "race car track",
    "car exhaust flames", "car backfire fire", "car burnout smoke", "rally car",
    "luxury sports car",
]
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


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"failed: {' '.join(cmd[:4])}")


def pexels(query, orientation):
    q = urllib.parse.urlencode({"query": query, "orientation": orientation, "per_page": 20})
    req = urllib.request.Request(f"https://api.pexels.com/videos/search?{q}",
                                 headers={"Authorization": PEXELS_KEY})
    return json.load(urllib.request.urlopen(req, timeout=30))["videos"]


def best_file(video):
    files = [f for f in video["video_files"] if f.get("file_type") == "video/mp4"]
    ok = [f for f in files if (f.get("height") or 0) >= 1080]
    pick = min(ok, key=lambda f: f["height"]) if ok else max(files, key=lambda f: f.get("height") or 0)
    return pick["link"]


def get_clips(n=16):
    os.makedirs(WORK, exist_ok=True)
    pool = {}
    for q in random.sample(QUERIES, 7):
        for orient in ("portrait", "landscape"):
            try:
                for v in pexels(q, orient):
                    if v["duration"] >= 4:
                        pool[f"px{v['id']}"] = (f"px{v['id']}", v["duration"], best_file(v))
            except Exception as e:
                print("pexels fail", q, e)
    chosen = random.sample(list(pool.values()), min(n, len(pool)))
    clips = []
    for cid, dur, url in chosen:
        path = f"{WORK}/clip_{cid}.mp4"
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print("download fail", cid, e)
                continue
        clips.append((path, dur))
    if len(clips) < 4:
        raise SystemExit("not enough clips")
    return clips


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
    clips = get_clips()

    segs, last = [], None
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        dur = b - a
        if dur < 0.25:
            continue
        choices = [c for c in clips if c[0] != last and c[1] > dur + 0.5] or clips
        path, cdur = random.choice(choices)
        last = path
        ss = random.uniform(0, max(0, cdur - dur - 0.2))
        seg = f"{WORK}/seg_{i:03d}.mp4"
        run(["ffmpeg", "-y", "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", path,
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
    print("done: out.mp4", len(segs), "cuts")


if __name__ == "__main__":
    main()
