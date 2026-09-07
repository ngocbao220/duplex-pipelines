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
uv run python single.py --pipeline vilier --input input.wav --debug
uv run python single.py --pipeline cholimex --input input.wav --gt-speaker-a gt_a.wav --gt-speaker-b gt_b.wav
uv run python end2end.py --pipeline all --data otospeech --max_gb 1
```

`single.py` thiếu GT sẽ ghi benchmark reference-free với metric reference là
`null`; đủ hai GT sẽ ghi reference benchmark. `end2end.py` tải/chuẩn bị mỗi
mixture OtoSpeech một lần, worker chỉ nhận mixture, rồi benchmark từng pipeline
và bảng tổng trên giao các sample hợp lệ.

Vilier cố định Silero VAD, Sortformer `nvidia/diar_sortformer_4spk-v1`,
overlap-only SepReformer `SepReformer_Base_WSJ0`, concat và cosine matching.
Không có cờ đổi model Vilier.
