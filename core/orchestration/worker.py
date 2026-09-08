"""One isolated process per pipeline, receiving mixtures only (never reference audio)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path = [entry for entry in sys.path if Path(entry).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT))
from core.orchestration.contract import run_sample, write_json


def cholimex(source, output, config):
    from core.config import Config, PATH_FIELDS
    from cholimex.runner import run_cholimex_file
    from cholimex.devices import resolve_device
    config = dict(config)
    debug = bool(config.pop('debug', False))
    cfg = Config(**{key: Path(value) if key in PATH_FIELDS and value is not None else value
                    for key, value in config.items()})
    result = run_cholimex_file(source, output, cfg)
    if not debug:
        import shutil
        shutil.rmtree(output / 'debug', ignore_errors=True)
    return [output / 'speaker_0.wav', output / 'speaker_1.wav'], {
        'device': resolve_device(cfg.runtime_device, cfg.allow_cpu_fallback), 'pipeline_result': result}


def duplexchat(source, output, config):
    from duplexchat.runner import run_single_audio
    from duplexchat.devices import resolve_device
    config = dict(config)
    debug = bool(config.pop('debug', False))
    phase_dir = output / 'phases'
    run_single_audio(str(source), output_prefix=str(output / 'speaker'),
                     output_dir=str(phase_dir), **config)
    if debug:
        import shutil
        shutil.move(str(phase_dir), str(output / 'debug'))
    else:
        import shutil
        shutil.rmtree(phase_dir)
    return [output / 'speakerA.wav', output / 'speakerB.wav'], {
        'device': resolve_device(config.get('runtime_device', 'auto'))}


def vilier(source, output, config):
    from vilier.runner import run
    return run(source, output, config)



ADAPTERS = {'cholimex': cholimex, 'duplexchat': duplexchat, 'vilier': vilier}


def run_batch(request: dict, adapter=None) -> list[dict]:
    name = request['pipeline']
    adapter = adapter or ADAPTERS[name]
    results = []
    print(f"========= Phase 3: Running Pipeline: {name} =========", flush=True)
    with tqdm(total=len(request['samples']), desc=f"{name} / samples", unit="sample") as progress:
        for index, sample in enumerate(request['samples'], 1):
            progress.set_postfix_str(sample['key'], refresh=True)
            print(f"[{name} {index}/{len(request['samples'])}] Processing sample {sample['key']}", flush=True)
            result = run_sample(name, sample, Path(request['pred_root']) / sample['key'],
                                request['config'], request['code'], adapter, force=request['force'])
            results.append(result)
            write_json(Path(request['results']), results)
            progress.update(1)
            if result["status"] == "complete":
                print(f"[{name}] Complete sample {sample['key']}"
                      f"{' (resumed)' if result.get('resumed') else ''}", flush=True)
            else:
                print(f"[{name}] Failed sample {sample['key']}; details: "
                      f"{Path(request['pred_root']) / sample['key'] / 'run.json'}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--request', type=Path)
    group.add_argument('--check-imports', choices=list(ADAPTERS))
    args = parser.parse_args()
    if args.check_imports:
        __import__(args.check_imports)
        print(f'{args.check_imports} imports OK', flush=True)
        return 0
    request = json.loads(args.request.read_text())
    print(f"Pipeline={request['pipeline']} interpreter={sys.executable}", flush=True)
    print(json.dumps(request['config'], ensure_ascii=False, indent=2), flush=True)
    rows = run_batch(request)
    return int(any(row['status'] != 'complete' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
