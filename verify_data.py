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
 3  parquet · mp4 가 전부 있고 0바이트가 아닌가              전수
 4  ★ mp4 를 전부 열어 해상도·프레임수 확인                  전수
 5  ★ parquet 을 전부 열어 행수·컬럼 확인                    전수
 6  16프레임 미만으로 버려지는 비율
 7  표본을 실제로 디코드해 픽셀·상태값까지 확인
 8  action_stats 가 정상인가
 9  action extractor 체크포인트가 있나
10  사전학습 3종이 받아지나   EVA-02 · SDXL VAE · mc3_18
11  진짜 DataLoader 로 배치 뽑기 (모양·범위·NaN)
12  GPU 에 모델 올리기 (OOM 사전 확인)
13  디스크 여유

★ 4번이 특히 중요하다. dataset.py 는 원본 해상도를 그대로 두는데
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
            infos[str(r)] = json.loads((r / "meta" / "info.json").read_text())
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
    head("3. parquet · mp4 존재 + 0바이트 검사 (%s개)" % format(len(entries), ","))
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
    ck("parquet 전부 존재", not mpq, "" if not mpq else "없음 %d개" % len(mpq))
    ck("mp4 전부 존재", not mmp, "" if not mmp else "없음 %d개" % len(mmp))
    ck("0바이트 파일 없음", not zero, "" if not zero else "0바이트 %d개" % len(zero))
    for p in (mpq[:3] + mmp[:3] + zero[:3]):
        print("      문제: %s" % p)
    if mpq or mmp or zero:
        b = len(set(mpq) | set(mmp) | set(zero))
        print("      -> 전체의 %.1f%% 를 못 쓴다. __getitem__ 이 조용히 건너뛰므로"
              % (b / max(len(entries) * 2, 1) * 100))
        print("         학습은 돌아가지만 그만큼 데이터를 잃는다.")
        return 1

    # ------------------------------------------------------------ 4
    head("4. ★ mp4 전수 열기 — 해상도·프레임수  (배치 collate 가 여기서 터진다)")
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

    if args.quick:
        seen = {}
        for r, ep, n in entries:
            seen.setdefault(str(r), (r, ep, n))
        targets = list(seen.values())
        print("  --quick: 데이터셋당 1개씩 %d개" % len(targets))
    else:
        targets = entries
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

    # ------------------------------------------------------------ 5
    head("5. ★ parquet 전수 열기 — 행수·컬럼")
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
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (r, ep, nrow, hascol, err) in enumerate(ex.map(probe_pq, pq_targets), 1):
            if err:
                perr.append((r, ep, err))
                continue
            if not hascol:
                nocol.append((r, ep))
            want = dict(((str(a), b), c) for a, b, c in entries).get((str(r), ep))
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

    # ------------------------------------------------------------ 6
    head("6. 16프레임 미만으로 버려지는 에피소드")
    short = [(r, ep, n) for r, ep, n in entries if n < SEQ_LEN]
    usable = len(entries) - len(short)
    print("  16프레임 미만 %d개 (%.2f%%)" % (len(short), len(short) / max(len(entries), 1) * 100))
    print("  실제 학습에 쓰이는 에피소드 %s개" % format(usable, ","))
    if len(short) > len(entries) * 0.1:
        warn("버려지는 비율이 10%% 를 넘는다 (%d개)" % len(short))
    ck("쓸 수 있는 에피소드가 있음", usable > 0)

    # ------------------------------------------------------------ 7
    head("7. 표본 %d개 실제 디코드 (픽셀·상태값까지)" % args.sample)
    import numpy as np
    picks = random.sample(entries, min(args.sample, len(entries)))
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

    # ------------------------------------------------------------ 8
    head("8. action_stats")
    sp = Path(args.action_stats)
    if not ck("파일 존재  %s" % sp, sp.exists()):
        return 1
    stt = json.loads(sp.read_text())
    ck("mean 6차원", len(stt.get("mean", [])) == 6, str([round(x, 2) for x in stt.get("mean", [])])[:56])
    ck("std 6차원", len(stt.get("std", [])) == 6, str([round(x, 2) for x in stt.get("std", [])])[:56])
    ck("std 에 0 없음", all(abs(x) > 1e-9 for x in stt.get("std", [1])))

    # ------------------------------------------------------------ 9
    head("9. action extractor 체크포인트")
    acp = Path(args.action_ckpt)
    ck("파일 존재  %s" % acp, acp.exists(),
       "" if not acp.exists() else "%.1f MB" % (acp.stat().st_size / 1048576))
    if not acp.exists():
        return 1

    # ------------------------------------------------------------ 10
    if args.skip_models:
        head("10. 사전학습 모델 - 건너뜀")
    else:
        head("10. 사전학습 3종 (없으면 학습 시작 직후 죽는다)")
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

    # ------------------------------------------------------------ 11
    head("11. 진짜 DataLoader 로 배치 뽑기 (batch=%d, %d배치)" % (args.batch_size, args.batches))
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
            print("      -> 4번 해상도 결과를 볼 것. 크기가 섞여 collate 가 터졌을 가능성.")
        return 1
    el = time.time() - t0
    print("  %d배치 %.1f초 (배치당 %.2f초)" % (seen, el, el / max(seen, 1)))
    if seen:
        print("  참고: 데이터 로딩만 따지면 20,000스텝에 약 %.1f시간" % (el / seen * 20000 / 3600))
        print("        (실제 학습은 GPU 연산이 더해지므로 이보다 오래 걸린다)")

    # ------------------------------------------------------------ 12
    if args.skip_gpu or not torch.cuda.is_available():
        head("12. GPU - 건너뜀")
    else:
        head("12. GPU 에 모델 올리기 (OOM 사전 확인)")
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

    # ------------------------------------------------------------ 13
    head("13. 디스크 여유")
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
