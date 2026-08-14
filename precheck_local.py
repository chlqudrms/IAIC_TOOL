"""GPU 를 빌리기 전에 로컬에서 PKL 과 실제 데이터를 대조한다.

포드에서 발견하면 돈이 나간다. 여기서 잡으면 공짜다.

로컬엔 av / pandas / pyarrow 가 없으므로 mp4 박스 구조를 직접 읽는다.
헤더(moov)만 읽으므로 빠르다. 얻는 것:
    stsd -> 코덱 해상도 (배치 collate 가 터지는 지점)
    stsz -> 샘플 수 = 프레임 수 (PKL 값과 대조)
"""
import json
import pickle
import struct
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PKL = Path(r"_v\v3_ar\episode_index_filtered.pkl")
REMOTE = "/workspace/data/train"
LOCAL = Path(r"C:\Users\chlqu\Downloads\open1\data\train")
VIDEO_KEY = "observation.images.image"
CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def boxes(f, end):
    """(타입, 내용시작, 내용끝) 을 차례로 내준다."""
    while f.tell() + 8 <= end:
        p = f.tell()
        hdr = f.read(8)
        if len(hdr) < 8:
            return
        size, typ = struct.unpack(">I4s", hdr)
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            body = p + 16
        elif size == 0:
            size = end - p
            body = p + 8
        else:
            body = p + 8
        if size < 8:
            return
        yield typ, body, p + size
        f.seek(p + size)


def probe(path):
    """(가로, 세로, 프레임수) 또는 오류 문자열."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(0)
            wh = frames = None
            handler = None

            def walk(s, e, depth=0):
                nonlocal wh, frames, handler
                if depth > 6:
                    return
                f.seek(s)
                for typ, bs, be in boxes(f, e):
                    if typ in CONTAINERS:
                        cur = f.tell()
                        walk(bs, be, depth + 1)
                        f.seek(cur)
                    elif typ == b"hdlr":
                        f.seek(bs + 8)
                        handler = f.read(4)
                    elif typ == b"stsd" and handler == b"vide":
                        f.seek(bs + 8)          # version/flags + entry_count
                        ent = f.read(40)
                        if len(ent) >= 38:
                            wh = struct.unpack(">HH", ent[32:36])
                    elif typ == b"stsz" and handler == b"vide":
                        f.seek(bs + 8)          # version/flags + sample_size
                        b = f.read(4)
                        if len(b) == 4:
                            frames = struct.unpack(">I", b)[0]

            walk(0, n)
            if wh is None:
                return "해상도를 못 읽음"
            return (wh[0], wh[1], frames if frames is not None else -1)
    except Exception as e:
        return "%s" % str(e)[:50]


def main():
    t0 = time.time()
    # PKL 안은 PosixPath 라 Windows 에서 못 만든다. PurePosixPath 로 바꿔 읽는다.
    # 포드(리눅스)에서는 생기지 않는 문제다.
    import pathlib as _pl

    class PU(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "pathlib" and "Path" in name:
                return _pl.PurePosixPath
            return super().find_class(mod, name)

    entries = PU(open(PKL, "rb")).load()
    print("PKL 에피소드 %s개" % format(len(entries), ","))

    # 원격 절대경로를 로컬로 옮긴다
    def loc(r):
        s = str(r).replace("\\", "/")
        assert s.startswith(REMOTE), "예상 밖 경로: %s" % s
        return LOCAL / s[len(REMOTE) + 1:]

    roots = sorted({str(r) for r, _, _ in entries})
    print("고유 데이터셋 %d개" % len(roots))

    infos, missdir = {}, []
    for r in roots:
        d = loc(r)
        if not d.is_dir():
            missdir.append(r)
            continue
        try:
            infos[r] = json.loads((d / "meta" / "info.json").read_text(encoding="utf-8"))
        except Exception as e:
            missdir.append("%s (info.json: %s)" % (r, str(e)[:40]))
    print("데이터셋 폴더 없음 %d개" % len(missdir))
    for m in missdir[:5]:
        print("   %s" % m)
    if missdir:
        return 1

    def paths(r, ep):
        i = infos[r]
        c = ep // i.get("chunks_size", 1000)
        d = loc(r)
        return (d / i["data_path"].format(episode_chunk=c, episode_index=ep),
                d / i["video_path"].format(episode_chunk=c, episode_index=ep,
                                           video_key=VIDEO_KEY))

    print("\n--- 존재 · 0바이트 검사 (%s x 2개) ---" % format(len(entries), ","))
    mp, mv, zero = [], [], []
    for r, ep, n in entries:
        pq, v4 = paths(str(r), ep)
        for p, box in ((pq, mp), (v4, mv)):
            try:
                if p.stat().st_size == 0:
                    zero.append(p)
            except FileNotFoundError:
                box.append(p)
    print("parquet 없음 %d개 · mp4 없음 %d개 · 0바이트 %d개"
          % (len(mp), len(mv), len(zero)))
    for p in (mp[:3] + mv[:3] + zero[:3]):
        print("   %s" % p)
    if mp or zero:
        return 1

    # 영상 키가 다른 데이터셋은 여기서 빼고 나머지를 본다
    print("\n--- mp4 전수 헤더 읽기 (해상도 · 프레임수) ---")
    gone = {str(p) for p in mv}
    targets = [(str(r), ep, n) for r, ep, n in entries
               if str(paths(str(r), ep)[1]) not in gone]
    print("   대상 %s개 (읽을 수 없는 %d개 제외)" % (format(len(targets), ","), len(mv)))

    def job(t):
        r, ep, n = t
        return (r, ep, n, probe(paths(r, ep)[1]))

    res, errs, fmis = Counter(), [], []
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        for k, (r, ep, n, out) in enumerate(ex.map(job, targets), 1):
            if isinstance(out, str):
                errs.append((r, ep, out))
            else:
                w, h, fr = out
                res[(h, w)] += 1
                if fr > 0 and fr != n:
                    fmis.append((r, ep, n, fr))
            if k % 2000 == 0:
                el = time.time() - t1
                print("   %s/%s  %.0f개/초" % (format(k, ","), format(len(targets), ","),
                                              k / max(el, 1e-9)), flush=True)
    print("   소요 %.1f초" % (time.time() - t1))
    print("읽기 실패 %d개" % len(errs))
    for r, ep, e in errs[:5]:
        print("   %s ep%d : %s" % (Path(r).name, ep, e))
    print("해상도 분포:")
    for k, v in res.most_common():
        print("   %dx%d   %s개" % (k[0], k[1], format(v, ",")))
    print("★ 해상도 한 종류인가: %s" % ("예" if len(res) == 1 else "아니오 — collate 가 터진다"))
    print("PKL 프레임수와 mp4 샘플수가 다른 것 %d개" % len(fmis))
    for r, ep, a, b in fmis[:5]:
        print("   %s ep%d : PKL %d / mp4 %d" % (Path(r).name, ep, a, b))
    short = [(r, ep, n) for r, ep, n in entries if n < 16]
    print("16프레임 미만(학습에서 제외) %d개  -> 실제 학습 %s개"
          % (len(short), format(len(entries) - len(short), ",")))
    print("\n총 %.1f초" % (time.time() - t0))
    return 1 if (errs or len(res) != 1) else 0


if __name__ == "__main__":
    raise SystemExit(main())
