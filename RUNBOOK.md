# v3_ar 학습 실행 순서

**팀 지시 그대로 진행합니다. 학습 코드는 수정하지 않습니다.**

---

## 0. 포드 사양

```
GPU              RTX PRO 6000        ← 승원 정정. "A6000" 은 잘못 말한 것
컨테이너 디스크    40GB 이상
네트워크 볼륨      30GB 이상 (train 8.5GB + 캐시 1GB + 체크포인트 2GB)
```

**RTX PRO 6000 은 Blackwell(sm_120) 이라 A6000·A40(Ampere) 과 다른 세대입니다.** 오래된 torch 는 이 GPU용 커널이 없어 첫 연산에서 죽습니다. 포드를 켜면 **가장 먼저** 확인하세요.

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

`(12, 0)` 이 나와야 하고 torch 는 **2.7 이상 / CUDA 12.8 이상**이어야 합니다. 낮으면 RunPod 템플릿을 최신 PyTorch 이미지로 바꾸세요. (`verify_data.py` 13번이 실제로 모델을 올려서 이걸 잡습니다)

---

## 1. 볼륨 정리 (기존 볼륨 재사용 시)

dinoproxy 실험 잔재로 약 5.5GB를 회수할 수 있습니다.

```bash
du -sh /workspace/* | sort -rh | head
```

**지워도 되는 것**

```bash
rm -rf /workspace/pairs_big /workspace/pairs_gen /workspace/pairs_gt \
       /workspace/dinopairs /workspace/holdout /workspace/final_results \
       /workspace/gen_all.mp4 /workspace/*.png /workspace/*.log /workspace/*.sh \
       /workspace/old_dino.csv /workspace/runs
cd /workspace/hf_cache/hub && rm -rf models--apple--aimv2* models--timm--aimv2* \
       models--timm--eva02_large* models--timm--vit_base_patch14_dinov2* \
       models--timm--vit_large_patch14_clip* models--timm--vit_base_patch16_clip* \
       models--timm--vit_base_patch16_224.mae models--timm--vit_base_patch16_siglip*
```

**🔴 반드시 남길 것**

```
/workspace/hf_cache/hub/models--timm--eva02_base_patch14_448.*   loss.py 가 씀
/workspace/open/submission_kit                                   제출 CSV 생성용
```

---

## 2. 코드 받기

```bash
cd /workspace
git clone -q --branch v3_ar --single-branch --depth 1 \
    https://github.com/kvara66/IAIC.git _repo
cp -r _repo/v3_ar /workspace/v3_ar && rm -rf _repo
ls /workspace/v3_ar
```

**있어야 할 파일**

```
train.py  loss.py  model.py  dataset.py  infer.py
setup.sh                      ← 커밋 6d5bd13 에서 추가됨
action_extractor_indep.py
episode_index_filtered.pkl
runs/extractor_v2/best.pt
```

`setup.sh` 가 안 보이면 **옛날 클론**입니다. 다시 받으세요. (팀이 나중에 추가했습니다)

점검 도구도 같이 받습니다.

```bash
cd /workspace
git clone -q https://github.com/chlqudrms/IAIC_TOOL.git _tools
cp _tools/verify_data.py _tools/setup.sh /workspace/v3_ar/ && rm -rf _tools
```

---

## 3. 환경 설치 — **팀 공식 setup.sh 를 씁니다**

```bash
export HF_HOME=/workspace/.cache/huggingface
bash /workspace/v3_ar/setup.sh
```

설치하는 것: `diffusers==0.31.0 timm av pandas pyarrow tensorboard accelerate transformers`
그리고 SDXL VAE · EVA02-B · MC3-18 을 미리 받아둡니다. `[OK]` 세 줄이 나와야 정상입니다.

**막힐 수 있는 지점 하나.** 이 스크립트는 `set -e` 라 첫 오류에서 즉시 멈춥니다. RunPod 이미지가 PEP668 로 잠겨 있으면 `pip install` 이 `externally-managed-environment` 로 실패합니다. 그때만 아래로 우회하고, **파일은 고치지 마세요.**

```bash
pip install --break-system-packages -q diffusers==0.31.0 timm av pandas pyarrow tensorboard accelerate transformers
bash /workspace/v3_ar/setup.sh
```

---

## 4. 데이터 배치

DACON에서 받은 `open.zip`을 풀어 **이 구조**로 만듭니다.

```
/workspace/data/train/00ri/so100_battery/meta/info.json
                                        /data/chunk-000/episode_000000.parquet
                                        /videos/chunk-000/observation.images.image/episode_000000.mp4
/workspace/data/train/AndrejOrsula/...
/workspace/data/train/so100_action_statistics.json
```

PKL이 **절대경로 `/workspace/data/train/...`** 를 들고 있어서 위치가 정확히 맞아야 합니다.

PKL이 **절대경로**를 들고 있어 `/workspace/data/train/` 이 정확히 맞아야 합니다. `open1/data/train` 을 통째로 올리면 됩니다.

### 올리는 법 — 로컬 PC 에서 (Git Bash)

파일이 11,132 × 2개라 하나씩 보내면 느립니다. **묶어서 한 줄기로 흘려보냅니다.**

```bash
cd /c/Users/chlqu/Downloads/open1/data && tar -cf - train | ssh -p <포트> root@<주소> "mkdir -p /workspace/data && tar -xf - -C /workspace/data"
```

mp4 는 이미 압축돼 있어 `gzip` 을 걸면 느려지기만 합니다. 안 겁니다.

올린 뒤 **포드에서** 개수를 맞춰 보세요. 로컬과 같아야 합니다.

```bash
find /workspace/data/train -name "*.mp4" | wc -l      # 11132 나와야 함
find /workspace/data/train -name "*.parquet" | wc -l  # 11132 나와야 함
ls /workspace/data/train/so100_action_statistics.json
du -sh /workspace/data/train                          # 8.5G
```

---

## 5. 학습 전 점검 — **여기서 걸리면 학습을 시작하지 않습니다**

```bash
cd /workspace/v3_ar
python verify_data.py
```

전수 검사라 **3~5분** 걸립니다. 급하면 `--quick`으로 표본만 볼 수 있지만, **최종 확인은 옵션 없이** 돌리세요.

### 🔴 학습이 도는 중에는 절대 돌리지 마세요

이 스크립트는 사전학습 모델을 받고 **GPU에 모델을 한 번 올립니다.** 학습 중에 같이 돌리면 VRAM을 뺏어 **학습이 OOM으로 죽습니다.**

코드에 차단 장치를 넣어놨습니다 — `train.py`가 돌고 있으면 실행을 거부합니다. 그래도 학습 중 확인이 필요하면 **GPU를 안 쓰는 것만** 보세요.

```bash
tail -20 /workspace/train.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

### 이 스크립트가 학습을 깨뜨리지 않는 이유

```
학습 코드 수정      없음. dataset.py 를 import 만 함
쓰기·삭제 코드      없음. 전부 읽기 전용 (코드로 확인함)
실행 시점          학습 전. 끝나면 프로세스 종료
동시 실행          차단 장치로 막음
```

검증이 오류로 죽어도 **아무것도 안 바뀝니다.** 다시 돌리거나 건너뛰고 학습해도 됩니다.

**검사 14가지**

```
 1  PKL 의 데이터셋 폴더가 전부 있나                전수
 2  meta/info.json 이 전부 읽히나                  전수
 3  ★★★ train.py 가 PKL "만" 쓰는지               ← 팀이 확인하라고 한 항목
 4  parquet · mp4 가 전부 있고 0바이트가 아닌가     전수
 5  ★ mp4 를 전부 열어 해상도·프레임수             전수
 6  ★ parquet 을 전부 열어 행수·컬럼               전수
 7  16프레임 미만으로 버려지는 비율
 8  표본 100개를 픽셀까지 디코드
 9  action_stats 가 6차원 mean/std 인가
10  action extractor 체크포인트가 있나
11  사전학습 3종이 받아지나  EVA-02 · SDXL VAE · mc3_18
12  진짜 DataLoader 로 배치 뽑기 (모양·범위·NaN)
13  GPU 에 모델 올리기 (OOM 사전 확인)
14  디스크 여유
```

### 3번이 팀 지시의 바로 그 항목

> 학습 전엔 한가지 그 PKL 파일에 있는 경로들**로만** 학습데이터로 잘 들어가고있는게 맞는지 확인만 해주세요

`train.py` 240줄이 이렇게 되어 있습니다.

```python
index = EpisodeIndex.load(cache) if cache.exists() else EpisodeIndex(args.train_root)
```

**PKL을 못 찾으면 오류를 내지 않고 `train_root` 전체를 훑어 다른 인덱스를 만듭니다.** 학습은 정상으로 돌고 로스도 나옵니다. 눈으로는 못 잡습니다.

`cache.exists()`는 **현재 폴더 기준**이라, `python /workspace/v3_ar/train.py`처럼 다른 위치에서 실행하면 그대로 전체 스캔으로 빠집니다. **반드시 `cd /workspace/v3_ar` 후 실행하세요.**

그래서 세 가지를 봅니다.

```
(가) train.py 가 PKL 갈래를 타는가
     train.py 의 argparse 기본값을 코드에서 직접 읽어 대조한다
(나) PKL 밖에 뭐가 더 있는가
     폴더를 전수 스캔해 PKL 과 대조. 걸러진 개수를 숫자로 보여준다
(다) train.py 와 똑같은 식으로 만든 인덱스가 PKL 과 정확히 같은가
     EpisodeIndex.load(cache) if cache.exists() else ... 를 그대로 실행해 집합 비교
```

### 왜 전수로 봐야 하나

`dataset.py`의 `__getitem__`이 이렇게 되어 있습니다.

```python
for _ in range(10):
    try:    return self._load_sample()
    except Exception:  continue
```

**파일이 없거나 깨져도 예외를 삼키고 넘어갑니다.** 데이터가 절반만 있어도 학습은 정상으로 보이고 로스도 나옵니다. "안 죽으니까 괜찮다"가 성립하지 않습니다.

### 5번이 특히 중요한 이유

`dataset.py`는 **원본 해상도를 그대로 둡니다**(`_preprocess_frame`). 배치에 해상도가 다른 에피소드가 섞이면 collate의 `torch.stack`이 터집니다. **이 오류는 `__getitem__` 밖에서 나므로 예외 삼키기로도 안 막히고, 학습 도중에 죽습니다.**

걸리면 임의로 고치지 말고 **팀에 알리고 지시를 받으세요.**

---

## 6. 학습 시작

```bash
export HF_HOME=/workspace/.cache/huggingface
cd /workspace/v3_ar
python train.py
```

끊겨도 살아남게 하려면

```bash
cd /workspace/v3_ar
setsid nohup python train.py > /workspace/train.log 2>&1 < /dev/null &
disown
tail -f /workspace/train.log
```

---

## 7. 로스 확인 주기

**코드가 정한 주기** (`train.py` 기본값)

```
train 로스     50스텝마다      --log-every 50
val 로스      500스텝마다      --val-every 500
체크포인트    2000스텝마다     --save-every 2000
best.ckpt     val 최저 갱신 시
```

**설정 요약**

```
총 20,000스텝 · 배치 4 · 에폭당 3,000샘플(750스텝) · 최대 27에폭
학습 시간 약 12~20시간 (스텝당 2~3초 추정, 실측으로 확정할 것)
```

### 🔴 팀이 중요하다고 한 것은 **500스텝마다 나오는 val 로스**

> 넵 로스도 찍히는지 체크해주세용 500스텝 마다 나오는 val 로스가 중요합니닷 첨엔 클거에요

**첫 val 이 크게 나오는 건 정상입니다.** 봐야 할 건 절대값이 아니라 **내려가는가**입니다.

```bash
grep -i "val" /workspace/train.log | tail -20
```

val 만 뽑아 추이를 보려면:

```bash
grep -oiE "step[^0-9]*[0-9]+.*val[^0-9]*[0-9.]+" /workspace/train.log | tail -20
```

**보고할 때는 스텝 번호와 val 값을 같이 적으세요.** 값 하나만으론 팀이 판단을 못 합니다.

**볼 주기**

```
시작 직후 5분      진짜 도는지, 로스가 숫자인지, GPU 100% 인지
첫 val (500스텝)   ★ 숫자가 찍히는지, NaN 아닌지   — 여기까지 약 20분
그 뒤 500스텝마다  ★ val 이 내려가는지 (약 20분 간격)
1시간마다         train/val 추이, best 갱신, 디스크, GPU
```

**즉시 중단해야 하는 신호**

```
🔴 로스 NaN / inf
🔴 프로세스 죽음 (OOM · 디스크 참사)
🔴 GPU 사용률 0%
🔴 디스크 90% 초과
🟡 val 이 3회 연속 안 떨어짐  (과적합·정체 신호. 중단은 아니고 보고 대상)
```

**확인 명령**

```bash
tail -20 /workspace/train.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
df -h /workspace | tail -1
ls -lh /workspace/v3_ar/runs/v2_ar/
```

---

## 8. 알아둘 것

```
출력 폴더        /workspace/v3_ar/runs/v2_ar
TensorBoard      runs/v2_ar/tb
체크포인트        best.ckpt · step{N}.ckpt · epoch{N}.ckpt  (개당 약 56MB)
이어서 학습       train.py 에 resume 인자가 있음 (--resume)
```

`loss.py`의 DINOLoss가 쓰는 모델은 **`eva02_base_patch14_448`** 입니다. dinoproxy 조사에서 채점기 DINO와 가장 잘 맞았던 모델이 그것입니다(1,200쌍 기준 r=0.8707, 2위와 유의차).
