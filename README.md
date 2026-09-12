# duplex-pipelines

Chạy các pipeline tách hai speaker trên cùng audio mono: `vilier`,
`cholimex`, `duplexchat`, và bản gốc vendor `sommelier`. Mỗi pipeline có môi
trường `uv` riêng để dependency không xung đột.

## Cài đặt

```bash
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/vilier
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/cholimex
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/duplexchat
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/sommelier
```

Để chạy diarization bằng NVIDIA Streaming Sortformer, cài thêm thư viện hệ
thống và NeMo trong môi trường DuplexChat:

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]
```

Trong môi trường `uv`, có thể dùng lệnh tương đương sau khi đã sync project:

```bash
uv pip install Cython packaging
uv pip install 'nemo_toolkit[asr]'
```

Vilier và Sommelier cần checkpoint SepReformer thật (`.pth`/`.pt`, không phải
Git-LFS pointer):

```bash
export VILIER_SEPREFORMER_CHECKPOINT=/path/to/epoch.0180.pth
```

Sommelier cũng cần `HF_TOKEN` hoặc `HUGGINGFACE_TOKEN` để tải pyannote.

## Chạy một audio

```bash
uv run --project pipeline/vilier python -m vilier single \
  --input mixture.wav --output-dir outputs/vilier --debug

uv run --project pipeline/cholimex python -m cholimex single \
  --input mixture.wav --output-dir outputs/cholimex --debug

uv run --project pipeline/duplexchat python -m duplexchat single \
  --input mixture.wav --output-dir outputs/duplexchat --debug

uv run --project pipeline/sommelier python -m sommelier single \
  --input mixture.wav --output-dir outputs/sommelier --debug
```

`audio.stereo.wav`, `run.json`, và `benchmark.json` nằm trong `--output-dir`.
Hai nguồn tách nằm lần lượt ở kênh 0 và 1; tên file không gán danh tính speaker.
Nếu input không có đúng hai speaker, Vilier/Sommelier fail thay vì tạo output giả.
Với `--debug`, Cholimex ghi từng overlap tại
`debug/overlaps/<id>/{mixture,audio.stereo}.wav`; Vilier giữ các native source
debug ở `debug/overlaps/<id>/{mixture,source_01,source_02}.wav`.
DuplexChat ghi raw diarization/linking artifacts trong `debug/phase_02_diarization/`.

## Metrics và benchmark

### Single audio có ground truth

Để có metric separation thật, phải đưa **cả hai** reference source. Không có
reference, `benchmark.json` vẫn được ghi nhưng các metric reference là `null`;
đó không phải kết quả benchmark.

```bash
uv run --project pipeline/cholimex python -m cholimex single \
  --input mixture.wav --output-dir outputs/cholimex-ref \
  --gt-speaker-a speaker_1.wav --gt-speaker-b speaker_2.wav
```

Metric được tính với PIT speaker assignment trên toàn bộ audio, non-overlap và
overlap: PIT-SI-SDR, SI-SDRi, SAR, SIR, ESTOI, PESQ, crosstalk rate, VAD F1,
onset/offset MAE, overlap F1 và overlap IoU. `PESQ`/`ESTOI` có thể là `null`
nếu package optional không sẵn có hoặc đoạn audio không hợp lệ.

## Smoke test sáu checkpoint separation

`tools/test_separation_models.py` chỉ tách audio, không diarization hay
reconstruction. Nó ghi native sources, elapsed time và RTF; **không thể tính
SI-SDR/PESQ** vì overlap WAV không có source reference.

```bash
git clone --depth 1 https://github.com/alibabasglab/MossFormer2 third_party/MossFormer2

UV_CACHE_DIR=.uv-cache uv run --project pipeline/vilier \
  --with 'diffusers>=0.30' --with 'asteroid>=0.7' --with 'rotary-embedding-torch' \
  python tools/test_separation_models.py \
  --overlaps-json compare/vi/vilier-1/debug/overlaps.json \
  --output-dir outputs/vilier-1-overlap-models \
  --models all --device cpu \
  --mossformer2-source third_party/MossFormer2
```

Model: SepFormer WSJ02Mix, SepFormer WHAMR, MossFormer2 LibriMix,
DialogueSidon, Rahma89 Conv-TasNet (native 3 source), và SepReformer Base
WSJ0. Output: `<output>/<model>/<overlap-id>/report.json` và WAV sources;
`<output>/summary.json` tổng hợp success/failure. Rahma89 không có
`audio.stereo.wav` vì checkpoint có ba source.

`MOSSFORMER2_SOURCE` có thể thay `--mossformer2-source`. Thêm
`SEPARATION_SMOKE_TRACEBACK=1` để ghi traceback vào report khi debug.

## Test code

```bash
UV_CACHE_DIR=.uv-cache uv run --project . --extra dev pytest -q
```
