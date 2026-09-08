# duplex-pipelines

Một repo chạy ba pipeline tách speaker trên cùng input: `vilier`, `duplexchat`,
và `cholimex`. Mỗi pipeline có project/lockfile riêng để stack Sortformer/NeMo
không xung đột dependency DuplexChat hoặc Cholimex.

## Cài môi trường

```bash
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/vilier
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/duplexchat
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/cholimex
```

Vilier cần resolve Sortformer/NeMo lần đầu. Sau khi dependency thay đổi, chạy
`UV_CACHE_DIR=.uv-cache uv lock --project pipeline/vilier` trên máy có quyền
truy cập PyPI trước khi dùng lockfile trong automation.

## Chạy

```bash
uv run --project pipeline/vilier python -m vilier single --input input.wav --output-dir outputs/vilier --debug
uv run --project pipeline/cholimex python -m cholimex single --input input.wav --output-dir outputs/cholimex --gt-speaker-a gt_a.wav --gt-speaker-b gt_b.wav
uv run --project pipeline/duplexchat python -m duplexchat otospeech --max-gb 1 --max-samples 1 --max-seconds 60
uv run --project . python cli.py compare-otospeech --max-gb 1 --max-samples 1 --max-seconds 60
```

Mỗi package có hai subcommand: `single` và `otospeech`. `single` thiếu GT sẽ ghi
benchmark reference-free với metric reference là `null`; đủ hai GT sẽ ghi
reference benchmark. `compare-otospeech` tải/chuẩn bị mỗi mixture một lần,
worker chỉ nhận mixture, rồi benchmark từng pipeline
và bảng tổng trên giao các sample hợp lệ.

Vilier cố định Silero VAD, Sortformer `nvidia/diar_sortformer_4spk-v1`,
overlap-only SepReformer `SepReformer_Base_WSJ0`, concat và cosine matching.
Không có cờ đổi model Vilier.

SepReformer cần weights `.pt`/`.pth` thật, không phải Git-LFS pointer. Trên
Kaggle, attach dataset weights rồi đặt `VILIER_SEPREFORMER_CHECKPOINT` tới file
checkpoint trước khi chạy Vilier; preflight sẽ kiểm tra file này trước
diarization.
