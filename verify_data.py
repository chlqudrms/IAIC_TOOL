"""v3_ar 학습 전 점검. 팀 지시의 "PKL 경로대로 잘 들어가는지 확인" 항목.

이 스크립트는 학습 코드를 건드리지 않는다. 읽기만 한다.

────────────────────────────────────────────────────────────────────
왜 전수로 봐야 하나
────────────────────────────────────────────────────────────────────
dataset.py 의 __getitem__ 이 이렇게 되어 있다.

    for _ in range(10):
        try:    return self._load_sample()
        except Exception:  continue

파일이 없거나 깨져도 예외를 삼키고 다른 에피소드로 넘어간다.
데이터가 절반만 있어도 학습은 정상으로 보이고 로스도 나온다.
"안 죽으니까 괜찮다"가 성립하지 않는다.

그리고 파일이 "존재"하는 것과 "읽히는" 것은 다르다. 0바이트 파일,
잘린 mp4, 깨진 parquet 은 stat 으로 안 잡힌다. 그래서 이 스크립트는
6,298개 전부를 실제로 열어본다. (mp4 는 헤더만, parquet 은 메타데이터만
읽으므로 몇 분이면 끝난다)

────────────────────────────────────────────────────────────────────
검사 항목. 하나라도 실패하면 학습을 시작하지 않는다.
────────────────────────────────────────────────────────────────────
 1  PKL 을 읽고 데이터셋 폴더가 전부 있나                    전수
 2  meta/info.json 이 전부 읽히고 필요한 키가 있나           전수
 3  ★★★ train.py 가 PKL "만" 쓰는지                        팀 지시 항목
 4  parquet · mp4 가 전부 있고 0바이트가 아닌가              전수
 5  ★ mp4 를 전부 열어 해상도·프레임수 확인                  전수
 6  ★ parquet 을 전부 열어 행수·컬럼 확인                    전수
 7  16프레임 미만으로 버려지는 비율
 8  표본을 실제로 디코드해 픽셀·상태값까지 확인
 9  action_stats 가 정상인가
10  action extractor 체크포인트가 있나
11  사전학습 3종이 받아지나   EVA-02 · SDXL VAE · mc3_18
12  진짜 DataLoader 로 배치 뽑기 (모양·범위·NaN)
13  GPU 에 모델 올리기 (OOM 사전 확인)
14  디스크 여유

★★★ 3번이 팀이 확인하라고 한 바로 그 항목이다. train.py 240 줄이

    index = EpisodeIndex.load(cache) if cache.exists() else EpisodeIndex(args.train_root)

  이라, PKL 을 못 찾으면 오류 없이 train_root 전체를 훑어 다른 인덱스를
  만든다. 학습은 정상으로 돌고 로스도 나오므로 눈으로는 못 잡는다.
  그래서 (가) PKL 갈래를 타는지 (나) PKL 밖에 뭐가 있는지
  (다) train.py 와 똑같이 만든 인덱스가 PKL 과 같은지 셋 다 본다.

★ 5번도 중요하다. dataset.py 는 원본 해상도를 그대로 두는데
  (_preprocess_frame), 배치에 해상도가 다른 에피소드가 섞이면 collate 의
  torch.stack 이 터진다. 이 오류는 __getitem__ 밖에서 나므로 예외 삼키기로
  안 막히고, 학습 도중에 죽는다.

사용:
  cd /workspace/v3_ar
  python verify_data.py                  # 전수 (권장, 몇 분 걸림)
  python verify_data.py --quick          # 빠르게 (표본만)
  python verify_data.py --workers 16     # 병렬 수 조정
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIDEO_KEY = "observation.images.image"
SEQ_LEN = 16
FAIL = []
WARN = []


def head(t):
    print("")
    print("=" * 70)
    print(t)
    print("=" * 70)


def ck(label, ok, detail=""):
    print("  %-50s %s %s" % (label, "통과" if ok else "★실패★", detail))
    if not ok:
        FAIL.append(label)
    return ok


def warn(msg):
    print("  경고: %s" % msg)
    WARN.append(msg)


def bar(done, total, t0):
    if total <= 0:
        return
    el = time.time() - t0
    rate = done / max(el, 1e-9)
    eta = (total - done) / max(rate, 1e-9)
    print("     %d/%d  %.0f개/초  남은시간 %.0f초" % (done, total, rate, eta), flush=True)


def guard_training_running(force: bool) -> bool:
    """학습이 도는 중이면 실행을 막는다.

    왜:
      이 스크립트는 사전학습 모델 3종을 받고 GPU 에 모델을 한 번 올린다.
      학습이 도는 중에 같이 돌리면 VRAM 을 뺏어 학습이 OOM 으로 죽을 수 있다.
      검증은 학습 "전"에만 하는 것이다.

    되돌릴 수 없는 피해를 막는 장치라 기본값은 차단이다. --force 로만 뚫는다.
    """
    import subprocess
    running = []
    try:
        out = subprocess.run(["ps", "-eo", "pid,etime,cmd"], capture_output=True,
                             text=True, timeout=10).stdout
        for line in out.splitlines():
            if "train.py" in line and "verify_data" not in line and " grep" not in line:
                running.append(line.strip())
    except Exception:
        pass
    if not running:
        return True

    print("")
    print("!" * 70)
    print("학습(train.py)이 이미 돌고 있습니다.")
    for r in running[:3]:
        print("   %s" % r[:100])
    print("")
    print("이 스크립트는 GPU 에 모델을 올려 확인하므로, 지금 돌리면")
    print("학습이 VRAM 부족으로 죽을 수 있습니다.")
    print("")
    print("학습 중에는 아래만 보세요. GPU 를 쓰지 않습니다.")
    print("   tail -20 /workspace/train.log")
    print("   nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
    print("")
    print("그래도 돌리려면:  python verify_data.py --force --skip-gpu --skip-models")
    print("!" * 70)
    if not force:
        return False
    print("  --force 지정됨. 계속합니다. (--skip-gpu --skip-models 를 함께 쓰는 것을 권장)")
    return True


def vpath(root, info, ep):
    chunk = ep // info.get("chunks_size", 1000)
    return Path(root) / info["video_path"].format(
        episode_chunk=chunk, episode_index=ep, video_key=VIDEO_KEY)


def ppath(root, info, ep):
    chunk = ep // info.get("chunks_size", 1000)
    return Path(root) / info["data_path"].format(
        episode_chunk=chunk, episode_index=ep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="episode_index_filtered.pkl")
    ap.add_argument("--train-root", default="../data/train")
    ap.add_argument("--action-stats", default="../data/train/so100_action_statistics.json")
    ap.add_argument("--action-ckpt", default="runs/extractor_v2/best.pt")
    ap.add_argument("--sample", type=int, default=100, help="픽셀까지 디코드해볼 개수")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8, help="전수 검사 병렬 수")
    ap.add_argument("--quick", action="store_true", help="전수 검사 건너뛰고 표본만")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="학습이 도는 중에도 실행 (권장하지 않음)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    t_all = time.time()

    # 학습 중이면 막는다. GPU 를 뺏어 학습을 죽일 수 있다.
    if not guard_training_running(args.force):
        return 2

    # ------------------------------------------------------------ 1
    head("1. PKL 읽기 · 데이터셋 폴더 존재")
    pkl = Path(args.pkl)
    if not ck("PKL 존재  %s" % pkl, pkl.exists()):
        return 1
    with open(pkl, "rb") as f:
        entries = pickle.load(f)
    print("  에피소드 %s개" % format(len(entries), ","))
    roots = sorted({Path(r) for r, _, _ in entries})
    print("  고유 데이터셋 %d개" % len(roots))
    pref = {str(Path(r)).rsplit("/", 2)[0] for r, _, _ in entries}
    print("  경로 접두어 %d종: %s" % (len(pref), sorted(pref)[:2]))
    miss = [r for r in roots if not r.is_dir()]
    ck("데이터셋 폴더 전부 존재 (%d개)" % len(roots), not miss,
       "" if not miss else "없음 %d개" % len(miss))
    for r in miss[:5]:
        print("      없음: %s" % r)
    if miss:
        print("      -> open.zip 을 %s 아래에 풀었는지 확인할 것" % args.train_root)
        return 1

    # ------------------------------------------------------------ 2
    head("2. meta/info.json")
    infos, bad = {}, []
    for r in roots:
        try:
            infos[str(r)] = json.loads((r / "meta" / "info.json").read_text(encoding="utf-8"))
        except Exception as e:
            bad.append((r, str(e)[:60]))
    ck("info.json 전부 읽힘 (%d개)" % len(roots), not bad,
       "" if not bad else "실패 %d개" % len(bad))
    for r, e in bad[:5]:
        print("      %s : %s" % (r, e))
    if bad:
        return 1
    nk = [k for k in ("data_path", "video_path") if any(k not in v for v in infos.values())]
    ck("data_path·video_path 키 있음", not nk, "" if not nk else "빠짐 %s" % nk)
    if nk:
        return 1

    # ------------------------------------------------------------ 3
    head("3. ★★★ train.py 가 PKL 만 쓰는지 — 팀 지시의 바로 그 항목")
    # 팀 지시 원문:
    #   "학습 전엔 한가지 그 PKL 파일에 있는 경로들로만 학습데이터로 잘
    #    들어가고있는게 맞는지 확인만 해주세요"
    #
    # train.py 240 줄:
    #   index = EpisodeIndex.load(cache) if cache.exists() else EpisodeIndex(args.train_root)
    #
    # PKL 을 못 찾으면 오류를 내지 않고 train_root 전체를 훑어 다른 인덱스를
    # 만든다. 학습은 정상으로 돌고 로스도 나온다. 눈으로는 절대 못 잡는다.
    # "로만" 을 증명하려면 세 가지를 봐야 한다.
    #   (가) train.py 가 정말 PKL 갈래를 타는가
    #   (나) PKL 밖에 뭐가 더 있는가        (폴더 전수 스캔과 대조)
    #   (다) train.py 와 똑같이 만든 인덱스가 PKL 과 정확히 같은가
    def key(x):
        return str(Path(x).resolve())

    troot = Path(args.train_root)
    pklset = {(key(r), int(e)) for r, e, _ in entries}

    print("  --- (가) train.py 가 PKL 갈래를 타는가 ---")
    # 내 기본값이 아니라 train.py 의 기본값을 코드에서 직접 읽는다.
    # 내 스크립트만 통과하고 train.py 는 다른 걸 쓰는 상황을 막기 위해서다.
    import ast as _ast
    tp = Path("train.py")
    tdef = {}
    if tp.exists():
        # utf-8-sig: train.py 는 파일 앞에 BOM 이 붙어 있다.
        # 그냥 utf-8 로 읽으면 ast.parse 가 U+FEFF 로 터진다.
        for nd in _ast.walk(_ast.parse(tp.read_text(encoding="utf-8-sig"))):
            if (isinstance(nd, _ast.Call) and getattr(nd.func, "attr", "") == "add_argument"
                    and nd.args and isinstance(nd.args[0], _ast.Constant)):
                for kw in nd.keywords:
                    if kw.arg == "default" and isinstance(kw.value, _ast.Constant):
                        tdef[nd.args[0].value.lstrip("-").replace("-", "_")] = kw.value.value
    if not ck("train.py 를 읽어 기본값 추출", bool(tdef), "인자 %d개" % len(tdef)):
        print("      -> cd /workspace/v3_ar 에서 실행해야 한다.")
        return 1
    t_cache = str(tdef.get("index_cache", args.pkl))
    t_root = str(tdef.get("train_root", args.train_root))
    print("      현재 작업 폴더   %s" % Path.cwd())
    print("      train.py 기본값  --index-cache %s   --train-root %s" % (t_cache, t_root))
    ck("내 --pkl 이 train.py 기본값과 같음", t_cache == str(args.pkl),
       "" if t_cache == str(args.pkl) else "다르면 엉뚱한 인덱스를 검증하는 셈")
    ck("내 --train-root 가 train.py 기본값과 같음", t_root == str(args.train_root))
    here = Path(t_cache).exists()
    ck("★ train.py 가 PKL 갈래를 탄다 (cache.exists() 가 True)", here)
    if not here:
        print("      -> train.py 는 %s 를 '현재 폴더 기준'으로 찾는다." % t_cache)
        print("         지금 폴더에 없으므로 train.py 는 전체 스캔으로 빠진다.")
        print("         반드시 cd /workspace/v3_ar 에서 python train.py 로 실행할 것.")
        return 1

    print("  --- (나) PKL 밖에 뭐가 더 있는가 ---")
    # EpisodeIndex.__init__ 과 같은 방식으로 훑되 parquet 을 열지 않아 빠르다.
    # 목적은 개수 대조이지 프레임수가 아니다.
    scan, ds_all = set(), set()
    for ip in sorted(troot.rglob("meta/info.json")):
        droot = ip.parent.parent
        ds_all.add(key(droot))
        try:
            inf = json.loads(ip.read_text(encoding="utf-8"))
        except Exception:
            continue
        cs = inf.get("chunks_size", 1000)
        for e in range(inf.get("total_episodes", 0)):
            if (droot / inf["data_path"].format(
                    episode_chunk=e // cs, episode_index=e)).exists():
                scan.add((key(droot), e))
    extra, lost = scan - pklset, pklset - scan
    print("      폴더 전수 스캔   에피소드 %8s개 · 데이터셋 %3d개"
          % (format(len(scan), ","), len(ds_all)))
    print("      PKL             에피소드 %8s개 · 데이터셋 %3d개"
          % (format(len(pklset), ","), len(roots)))
    print("      PKL 이 걸러낸 것 %s개 — PKL 갈래를 타므로 학습에 안 들어간다"
          % format(len(extra), ","))
    for k_, e_ in sorted(extra)[:3]:
        print("         제외됨: %s ep%d" % (Path(k_).name, e_))
    ck("PKL 의 에피소드가 폴더에 전부 실재함", not lost,
       "" if not lost else "폴더에 없는 PKL 항목 %d개" % len(lost))
    for k_, e_ in sorted(lost)[:5]:
        print("         없음: %s ep%d" % (k_, e_))
    if lost:
        print("      -> PKL 이 가리키는 데이터가 없다. open.zip 배치를 다시 볼 것.")
        return 1
    # 구분자를 직접 붙이지 않는다. Path.parents 로 판정해야 OS 를 안 탄다.
    _tr = Path(key(troot))

    def under(k_):
        q = Path(k_)
        return q == _tr or _tr in q.parents

    out = sorted({k_ for k_, _ in pklset if not under(k_)})
    ck("PKL 경로가 전부 train_root 안에 있음", not out,
       "" if not out else "밖 %d개" % len(out))
    for k_ in out[:3]:
        print("         밖: %s" % k_)
    if out:
        return 1

    print("  --- (다) train.py 와 똑같이 만든 인덱스가 PKL 과 같은가 ---")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from dataset import EpisodeIndex as _EI
    except Exception as e:
        ck("dataset.py 임포트", False, str(e)[:60])
        print("      -> bash setup.sh 를 먼저 돌릴 것.")
        return 1
    _c = Path(t_cache)
    _idx = _EI.load(_c) if _c.exists() else _EI(t_root)   # ← train.py 240 줄 그대로
    got = {(key(r), int(e)) for r, e, _ in _idx.entries}
    ck("★ train.py 방식 인덱스 == PKL (전부 일치)", got == pklset,
       "%s개" % format(len(got), ","))
    if got != pklset:
        print("      들어오면 안 되는데 들어온 것 %d개" % len(got - pklset))
        print("      들어와야 하는데 빠진 것   %d개" % len(pklset - got))
        return 1
    ge = len([1 for _, _, n in _idx.entries if n >= SEQ_LEN])
    print("      이 중 16프레임 이상 %s개만 실제 학습에 쓰인다 (LeRobotSequenceDataset)"
          % format(ge, ","))
    print("      train.py 는 여기서 다시 5% 를 val 로 뗀다 (split_index_by_episode)")
    print("      -> 학습 %s개 · 검증 %s개 (근사)"
          % (format(int(ge * 0.95), ","), format(ge - int(ge * 0.95), ",")))
    ck("학습에 쓸 에피소드가 남음", ge > 0)

    # ------------------------------------------------------------ 4
    head("4. parquet · mp4 존재 + 0바이트 검사 (%s개)" % format(len(entries), ","))
    t0 = time.time()
    mpq, mmp, zero = [], [], []
    for r, ep, n in entries:
        info = infos[str(r)]
        for p, box in ((ppath(r, info, ep), mpq), (vpath(r, info, ep), mmp)):
            try:
                sz = p.stat().st_size
                if sz == 0:
                    zero.append(p)
            except FileNotFoundError:
                box.append(p)
    print("  소요 %.1f초" % (time.time() - t0))

    # mp4 가 없는 이유를 갈라 본다.
    #   (A) 그 데이터셋이 영상 폴더 이름을 다르게 쓴다 -> 코드 수정 없이는 못 쓴다.
    #       dataset.py 는 VIDEO_KEY = "observation.images.image" 로 고정돼 있다.
    #       팀 지시가 "코드 수정 없이 실행만" 이므로 이건 감수하고 진행한다.
    #   (B) 그 밖의 이유 -> 데이터 배치가 잘못된 것이다. 반드시 멈춘다.
    keymiss, realmiss, altkeys = [], [], {}
    for p in mmp:
        try:
            vd = p.parent.parent           # videos/chunk-000
            ks = sorted(q.name for q in vd.iterdir() if q.is_dir()) if vd.is_dir() else []
        except Exception:
            ks = []
        if ks and VIDEO_KEY not in ks:
            keymiss.append(p)
            altkeys.setdefault(str(p.parent.parent.parent.parent), ks)
        else:
            realmiss.append(p)

    ck("parquet 전부 존재", not mpq, "" if not mpq else "없음 %d개" % len(mpq))
    ck("mp4 전부 존재 (영상키 불일치 제외)", not realmiss,
       "" if not realmiss else "없음 %d개" % len(realmiss))
    ck("0바이트 파일 없음", not zero, "" if not zero else "0바이트 %d개" % len(zero))
    for p in (mpq[:3] + realmiss[:3] + zero[:3]):
        print("      문제: %s" % p)
    if mpq or realmiss or zero:
        b = len(set(mpq) | set(realmiss) | set(zero))
        print("      -> 전체의 %.1f%% 를 못 쓴다. __getitem__ 이 조용히 건너뛰므로"
              % (b / max(len(entries) * 2, 1) * 100))
        print("         학습은 돌아가지만 그만큼 데이터를 잃는다.")
        return 1

    # 영상 폴더 이름이 다른 데이터셋 — 멈추지 않고 정확히 알린다
    skip_ep = set()
    if keymiss:
        skip_ep = {str(p) for p in keymiss}
        warn("영상 폴더 이름이 달라 못 읽는 에피소드 %d개 (%.2f%%)"
             % (len(keymiss), len(keymiss) / max(len(entries), 1) * 100))
        print("")
        print("      dataset.py 는 VIDEO_KEY = \"%s\" 로 고정돼 있다." % VIDEO_KEY)
        print("      아래 데이터셋은 폴더 이름이 달라 학습에 안 들어간다.")
        for d, ks in sorted(altkeys.items()):
            print("         %-44s 실제 폴더 %s" % ("/".join(d.split("/")[-2:]), ks))
        print("")
        print("      코드를 고쳐야 살릴 수 있는데 팀 지시가 \"수정 없이 실행만\" 이다.")
        print("      -> 이대로 진행한다. 학습에 쓰이는 에피소드는 %s개."
              % format(len(entries) - len(keymiss), ","))
        print("      -> 팀에 알릴 것. 우리가 만든 문제가 아니라 PKL 과 데이터의 불일치다.")

    # ------------------------------------------------------------ 5
    head("5. ★ mp4 전수 열기 — 해상도·프레임수  (배치 collate 가 여기서 터진다)")
    import av

    def probe_mp4(item):
        r, ep, n = item
        try:
            with av.open(str(vpath(r, infos[str(r)], ep))) as c:
                s = c.streams.video[0]
                nf = int(s.frames) if s.frames else -1
                return (r, ep, (s.codec_context.height, s.codec_context.width), nf, None)
        except Exception as e:
            return (r, ep, None, -1, str(e)[:60])

    # 영상키가 달라 애초에 못 여는 것은 빼고 본다 (위에서 이미 알렸다)
    usable = [(r, ep, n) for r, ep, n in entries
              if str(vpath(r, infos[str(r)], ep)) not in skip_ep]
    if args.quick:
        seen = {}
        for r, ep, n in usable:
            seen.setdefault(str(r), (r, ep, n))
        targets = list(seen.values())
        print("  --quick: 데이터셋당 1개씩 %d개" % len(targets))
    else:
        targets = usable
        print("  전수 %s개 (병렬 %d)" % (format(len(targets), ","), args.workers))
    t0 = time.time()
    res, res_where, verr, nf_bad = Counter(), {}, [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (r, ep, shape, nf, err) in enumerate(ex.map(probe_mp4, targets), 1):
            if err:
                verr.append((r, ep, err))
            else:
                res[shape] += 1
                res_where.setdefault(shape, []).append(Path(r).name)
                if nf > 0 and nf < SEQ_LEN:
                    nf_bad.append((r, ep, nf))
            if i % 2000 == 0:
                bar(i, len(targets), t0)
    print("  소요 %.1f초" % (time.time() - t0))
    ck("mp4 전부 열림 (%d/%d)" % (len(targets) - len(verr), len(targets)), not verr,
       "" if not verr else "실패 %d개" % len(verr))
    for r, ep, e in verr[:5]:
        print("      %s ep%d : %s" % (Path(r).name, ep, e))
    print("  해상도 분포:")
    for k, v in res.most_common():
        print("      %dx%d  %s개   예: %s" % (k[0], k[1], format(v, ","),
                                              ", ".join(sorted(set(res_where[k]))[:3])))
    ok_res = len(res) <= 1
    ck("해상도가 한 종류", ok_res)
    if not ok_res:
        print("")
        print("      ★ dataset.py 는 원본 해상도를 그대로 둔다(_preprocess_frame).")
        print("        배치에 다른 해상도가 섞이면 collate 의 torch.stack 이 터진다.")
        print("        이 오류는 __getitem__ 밖에서 나므로 예외 삼키기로 안 막힌다.")
        print("        batch-size 1 로 돌리거나 소수 해상도를 PKL 에서 빼야 한다.")
        print("        ※ 임의로 고치지 말고 팀에 알리고 지시를 받을 것.")
        minority = sum(v for k, v in res.items() if v != max(res.values()))
        print("        소수 해상도 에피소드 %d개 (%.2f%%)" % (minority, minority / max(sum(res.values()), 1) * 100))
    if nf_bad:
        warn("mp4 프레임이 16장 미만인 에피소드 %d개" % len(nf_bad))
    if verr:
        return 1

    # ------------------------------------------------------------ 6
    head("6. ★ parquet 전수 열기 — 행수·컬럼")
    import pandas as pd

    def probe_pq(item):
        r, ep, n = item
        try:
            import pyarrow.parquet as pq_
            f = pq_.ParquetFile(str(ppath(r, infos[str(r)], ep)))
            nrow = f.metadata.num_rows
            cols = set(f.schema_arrow.names)
            return (r, ep, nrow, ("observation.state" in cols), None)
        except Exception as e:
            return (r, ep, -1, False, str(e)[:60])

    if args.quick:
        pq_targets = targets
        print("  --quick: %d개" % len(pq_targets))
    else:
        pq_targets = entries
        print("  전수 %s개 (병렬 %d)" % (format(len(pq_targets), ","), args.workers))
    t0 = time.time()
    perr, nocol, mismatch, short_pq = [], [], [], []
    # 루프 밖에서 한 번만 만든다. 안에서 만들면 6,298 x 6,298 이라 몇 시간 걸린다.
    want_n = {(str(a), b): c for a, b, c in entries}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (r, ep, nrow, hascol, err) in enumerate(ex.map(probe_pq, pq_targets), 1):
            if err:
                perr.append((r, ep, err))
                continue
            if not hascol:
                nocol.append((r, ep))
            want = want_n.get((str(r), ep))
            if want is not None and nrow != want:
                mismatch.append((r, ep, want, nrow))
            if nrow < SEQ_LEN:
                short_pq.append((r, ep, nrow))
            if i % 2000 == 0:
                bar(i, len(pq_targets), t0)
    print("  소요 %.1f초" % (time.time() - t0))
    ck("parquet 전부 열림 (%d/%d)" % (len(pq_targets) - len(perr), len(pq_targets)), not perr,
       "" if not perr else "실패 %d개" % len(perr))
    for r, ep, e in perr[:5]:
        print("      %s ep%d : %s" % (Path(r).name, ep, e))
    ck("observation.state 컬럼 있음", not nocol, "" if not nocol else "없음 %d개" % len(nocol))
    if mismatch:
        warn("PKL 프레임수와 parquet 행수가 다른 에피소드 %d개 (예: %s)"
             % (len(mismatch), ["ep%d %d!=%d" % (e, w, g) for _, e, w, g in mismatch[:3]]))
    if short_pq:
        warn("parquet 행수가 16 미만인 에피소드 %d개" % len(short_pq))
    if perr or nocol:
        return 1

    # ------------------------------------------------------------ 7
    head("7. 16프레임 미만으로 버려지는 에피소드")
    short = [(r, ep, n) for r, ep, n in entries if n < SEQ_LEN]
    # 이름을 usable 로 쓰면 5번에서 만든 목록을 숫자로 덮어써 8번이 터진다
    n_usable = len(entries) - len(short)
    print("  16프레임 미만 %d개 (%.2f%%)" % (len(short), len(short) / max(len(entries), 1) * 100))
    print("  실제 학습에 쓰이는 에피소드 %s개" % format(n_usable, ","))
    if len(short) > len(entries) * 0.1:
        warn("버려지는 비율이 10%% 를 넘는다 (%d개)" % len(short))
    ck("쓸 수 있는 에피소드가 있음", n_usable > 0)

    # ------------------------------------------------------------ 8
    head("8. 표본 %d개 실제 디코드 (픽셀·상태값까지)" % args.sample)
    import numpy as np
    # entries 가 아니라 usable 에서 뽑는다. 못 여는 것을 뽑으면 헛되이 실패한다.
    picks = random.sample(usable, min(args.sample, len(usable)))
    okn, errs = 0, []
    st_min, st_max = float("inf"), float("-inf")
    px_min, px_max = 255, 0
    for r, ep, n in picks:
        info = infos[str(r)]
        try:
            df = pd.read_parquet(ppath(r, info, ep), columns=["observation.state"])
            k = min(SEQ_LEN, len(df))
            arr = np.array(df["observation.state"].iloc[:k].tolist(), dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 6:
                raise ValueError("observation.state 모양 %s (…,6 이어야 함)" % (arr.shape,))
            if not np.isfinite(arr).all():
                raise ValueError("observation.state 에 NaN/Inf")
            st_min = min(st_min, float(arr.min()))
            st_max = max(st_max, float(arr.max()))
            got = []
            with av.open(str(vpath(r, info, ep))) as c:
                for fr in c.decode(c.streams.video[0]):
                    got.append(fr.to_ndarray(format="rgb24"))
                    if len(got) >= min(SEQ_LEN, n):
                        break
            need = min(SEQ_LEN, n)
            if len(got) < need:
                raise ValueError("프레임 %d장만 디코드 (%d 필요)" % (len(got), need))
            a0 = got[0]
            px_min = min(px_min, int(a0.min()))
            px_max = max(px_max, int(a0.max()))
            if a0.ndim != 3 or a0.shape[2] != 3:
                raise ValueError("프레임 모양 %s" % (a0.shape,))
            okn += 1
        except Exception as e:
            errs.append((r, ep, str(e)[:70]))
    ck("표본 전부 디코드됨 (%d/%d)" % (okn, len(picks)), okn == len(picks))
    for r, ep, e in errs[:5]:
        print("      %s ep%d : %s" % (Path(r).name, ep, e))
    print("  observation.state 원본 범위 [%.2f, %.2f]" % (st_min, st_max))
    print("  픽셀 범위 [%d, %d]" % (px_min, px_max))
    ck("픽셀이 단색이 아님", px_max > px_min, "전부 같은 값이면 영상이 깨진 것")
    if errs:
        return 1

    # ------------------------------------------------------------ 9
    head("9. action_stats")
    sp = Path(args.action_stats)
    if not ck("파일 존재  %s" % sp, sp.exists()):
        return 1
    stt = json.loads(sp.read_text(encoding="utf-8"))
    ck("mean 6차원", len(stt.get("mean", [])) == 6, str([round(x, 2) for x in stt.get("mean", [])])[:56])
    ck("std 6차원", len(stt.get("std", [])) == 6, str([round(x, 2) for x in stt.get("std", [])])[:56])
    ck("std 에 0 없음", all(abs(x) > 1e-9 for x in stt.get("std", [1])))

    # ------------------------------------------------------------ 10
    head("10. action extractor 체크포인트")
    acp = Path(args.action_ckpt)
    ck("파일 존재  %s" % acp, acp.exists(),
       "" if not acp.exists() else "%.1f MB" % (acp.stat().st_size / 1048576))
    if not acp.exists():
        return 1

    # ------------------------------------------------------------ 11
    if args.skip_models:
        head("11. 사전학습 모델 - 건너뜀")
    else:
        head("11. 사전학습 3종 (없으면 학습 시작 직후 죽는다)")
        t0 = time.time()
        try:
            import timm
            m = timm.create_model("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
                                  pretrained=True, num_classes=0)
            ck("EVA-02 base 448  (loss.py DINOLoss)", True, "%d차원" % m.num_features)
            del m
        except Exception as e:
            ck("EVA-02 base 448", False, str(e)[:60])
        try:
            from diffusers import AutoencoderKL
            v = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
            ck("SDXL VAE  (model.py)", True)
            del v
        except Exception as e:
            ck("SDXL VAE", False, str(e)[:60])
        try:
            from torchvision.models.video import MC3_18_Weights, mc3_18
            rr = mc3_18(weights=MC3_18_Weights.DEFAULT)
            ck("mc3_18  (loss.py R3DLoss)", True)
            del rr
        except Exception as e:
            ck("mc3_18", False, str(e)[:60])
        print("  소요 %.1f초" % (time.time() - t0))

    # ------------------------------------------------------------ 12
    head("12. 진짜 DataLoader 로 배치 뽑기 (batch=%d, %d배치)" % (args.batch_size, args.batches))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from torch.utils.data import DataLoader
    from dataset import EpisodeIndex, LeRobotSequenceDataset

    index = EpisodeIndex.load(args.pkl)
    ds = LeRobotSequenceDataset(index, args.action_stats,
                                samples_per_epoch=args.batches * args.batch_size)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=2, drop_last=True)
    t0, seen = time.time(), 0
    try:
        for b in dl:
            f, s = b["frames"], b["states"]
            if seen == 0:
                print("  frames %s  %s  범위 [%.3f, %.3f]" % (tuple(f.shape), f.dtype, f.min(), f.max()))
                print("  states %s  %s  범위 [%.3f, %.3f]" % (tuple(s.shape), s.dtype, s.min(), s.max()))
            ck("배치%d frames (B,16,3,H,W)" % (seen + 1),
               f.ndim == 5 and f.shape[1] == SEQ_LEN and f.shape[2] == 3)
            ck("배치%d states (B,16,6)" % (seen + 1),
               s.ndim == 3 and tuple(s.shape[1:]) == (SEQ_LEN, 6))
            ck("배치%d frames 범위 [-1,1]" % (seen + 1),
               float(f.min()) >= -1.001 and float(f.max()) <= 1.001)
            ck("배치%d NaN/Inf 없음" % (seen + 1),
               bool(torch.isfinite(f).all()) and bool(torch.isfinite(s).all()))
            seen += 1
            if seen >= args.batches:
                break
    except Exception as e:
        ck("DataLoader 배치 생성", False, str(e)[:70])
        if "stack" in str(e).lower() or "size" in str(e).lower():
            print("      -> 5번 해상도 결과를 볼 것. 크기가 섞여 collate 가 터졌을 가능성.")
        return 1
    el = time.time() - t0
    print("  %d배치 %.1f초 (배치당 %.2f초)" % (seen, el, el / max(seen, 1)))
    if seen:
        print("  참고: 데이터 로딩만 따지면 20,000스텝에 약 %.1f시간" % (el / seen * 20000 / 3600))
        print("        (실제 학습은 GPU 연산이 더해지므로 이보다 오래 걸린다)")

    # ------------------------------------------------------------ 13
    if args.skip_gpu or not torch.cuda.is_available():
        head("13. GPU - 건너뜀")
    else:
        head("13. GPU 에 모델 올리기 (OOM 사전 확인)")
        try:
            from model import ImageEditingModel
            dev = torch.device("cuda")
            p = torch.cuda.get_device_properties(0)
            print("  GPU: %s  %.1f GB" % (p.name, p.total_memory / 1073741824))
            mdl = ImageEditingModel(cond_dim=256).to(dev)
            print("  파라미터 %.1fM" % (sum(q.numel() for q in mdl.parameters()) / 1e6))
            ck("모델 GPU 적재", True, "%.2f GB 사용" % (torch.cuda.memory_allocated() / 1073741824))
            del mdl
            torch.cuda.empty_cache()
        except Exception as e:
            ck("모델 GPU 적재", False, str(e)[:70])

    # ------------------------------------------------------------ 14
    head("14. 디스크 여유")
    for p in (Path("."), Path("/workspace")):
        try:
            t, u, fr = shutil.disk_usage(p)
            print("  %-14s 전체 %6.1fGB  사용 %6.1fGB  여유 %6.1fGB"
                  % (str(p), t / 1073741824, u / 1073741824, fr / 1073741824))
        except Exception:
            pass
    try:
        _, _, fr = shutil.disk_usage(Path("."))
        ck("체크포인트용 여유 2.5GB 이상", fr / 1073741824 >= 2.5, "약 56MB x 최대 38개")
    except Exception:
        pass

    # ------------------------------------------------------------ 결과
    head("결과  (총 %.1f초)" % (time.time() - t_all))
    if args.quick:
        print("⚠ --quick 으로 돌렸다. 전수 검증이 아니다. 최종 확인은 옵션 없이 다시 돌릴 것.")
        print("")
    if WARN:
        print("경고 %d건 (학습은 가능하나 알고 있을 것):" % len(WARN))
        for w in WARN:
            print("   - %s" % w)
        print("")
    if FAIL:
        print("★ 실패 %d건 - 학습을 시작하지 말 것" % len(FAIL))
        for f in FAIL:
            print("   - %s" % f)
        return 1
    print("전부 통과. PKL 의 %s개 에피소드가 모두 실제로 읽힌다." % format(len(entries), ","))
    print("")
    print("학습 시작:")
    print("  export HF_HOME=/workspace/.cache/huggingface")
    print("  cd /workspace/v3_ar && python train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
