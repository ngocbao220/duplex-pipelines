# duplex-pipelines

Một repo chạy ba pipeline tách speaker trên cùng input: `vilier`, `duplexchat`,
và `cholimex`. Mỗi pipeline có project/lockfile riêng để stack Sortformer/NeMo
không xung đột dependency DuplexChat hoặc Cholimex.

## Cài môi trường

```bash
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/vilier
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/duplexchat
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/cholimex
UV_CACHE_DIR=.uv-cache uv sync --project pipeline/sommelier
```

Vilier cần resolve Sortformer/NeMo lần đầu. Sau khi dependency thay đổi, chạy
`UV_CACHE_DIR=.uv-cache uv lock --project pipeline/vilier` trên máy có quyền
truy cập PyPI trước khi dùng lockfile trong automation.

## Chạy

```bash
uv run --project pipeline/vilier python -m vilier single --input input.wav --output-dir outputs/vilier --debug
uv run --project pipeline/sommelier python -m sommelier single --input input.wav --output-dir outputs/sommelier --debug
uv run --project pipeline/cholimex python -m cholimex single --input input.wav --output-dir outputs/cholimex --gt-speaker-a gt_a.wav --gt-speaker-b gt_b.wav
uv run --project pipeline/duplexchat python -m duplexchat single --input input.wav --output-dir outputs/duplexchat --scale false --debug
uv run --project pipeline/duplexchat python -m duplexchat single --input input.wav --output-dir outputs/duplexchat-scale --scale true --debug
uv run --project pipeline/duplexchat python -m duplexchat otospeech --max-gb 1 --max-samples 1 --max-seconds 60
uv run --project . python cli.py compare-otospeech --max-gb 1 --max-samples 1 --max-seconds 60
```

Mỗi package có hai subcommand: `single` và `otospeech`. `single` thiếu GT sẽ ghi
benchmark reference-free với metric reference là `null`; đủ hai GT sẽ ghi
reference benchmark. `compare-otospeech` tải/chuẩn bị mỗi mixture một lần,
worker chỉ nhận mixture, rồi benchmark từng pipeline
và bảng tổng trên giao các sample hợp lệ.

Log console của worker dùng cùng format màu với Sommelier: `timestamp - pipeline
- [INFO] - ...`; `worker.log` giữ bản không ANSI và chứa cả noise từ thư viện.
Khi `--debug`, Vilier và Cholimex ghi overlap đã được gán speaker tại
`debug/overlaps.json` và `debug/overlaps/overlap_00000/{metadata.json,mixture.wav,speakerA.wav,speakerB.wav}`.
DuplexChat ghi `debug/conversations_2spk.json`. `--scale false` là full-input
debug separation, còn `--scale true` theo flow gốc: chỉ tách từng hội thoại hai
speaker tại `conversations/conversation_00000/`; mode này không tạo full-duration
`speakerA.wav`/`speakerB.wav` và không nhận ground truth full-input hoặc OtoSpeech.

Cholimex mặc định `cholimex.speaker_assignment_mode=relative_similarity`: chọn
mapping candidate-to-speaker có cosine tương đối cao hơn, như Vilier. Đặt
`strict_threshold` để giữ quality gate cũ theo
`cholimex.cosine_similarity_threshold`; overlap debug ghi candidate, track đã
gán, score matrix và margin để audit.

Vilier cố định Silero VAD, Sortformer `nvidia/diar_sortformer_4spk-v1`,
overlap-only SepReformer `SepReformer_Base_WSJ0`, concat và cosine matching.
Không có cờ đổi model Vilier.

Profile Vilier dùng để tạo full-duplex giữ đường xử lý trước-track của
Sommelier: peak-normalized mono 16 kHz, silence-preserving Sortformer chunks
tối đa 120 giây, sequential cosine speaker linking, và SepReformer source
assignment bằng `pyannote/embedding`. Job dừng trước ASR/LLM và chỉ trả hai
WAV mono full-duration: `outputs/vilier/speakerA.wav` và
`outputs/vilier/speakerB.wav`. Nếu diarization không có đúng hai speaker,
job fail thay vì tạo track giả.

SepReformer cần weights `.pt`/`.pth` thật, không phải Git-LFS pointer. Trên
Kaggle, attach dataset weights rồi đặt `VILIER_SEPREFORMER_CHECKPOINT` tới file
checkpoint trước khi chạy Vilier; preflight sẽ kiểm tra file này trước
diarization.

Sommelier là bản gốc được vendor riêng, chỉ chạy bằng `single` (không thuộc
`compare-otospeech --all`). Nó cũng cần `VILIER_SEPREFORMER_CHECKPOINT` và
`HUGGINGFACE_TOKEN` (hoặc `HF_TOKEN`). Chạy `uv sync --project
pipeline/sommelier` sau mỗi lần cập nhật dependency; code gốc import `openai`,
`onnxruntime`, `faster-whisper`, và `whisperx` ngay khi khởi động, dù các nhánh
ASR/LLM tương ứng đang bị tắt.
