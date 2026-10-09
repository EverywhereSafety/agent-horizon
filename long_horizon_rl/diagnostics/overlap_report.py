"""Align explicit actor stage wall times with node-wide GPU samples."""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def stage_intervals(log):
    pending, intervals = {}, []
    pattern = re.compile(
        r"LONG_HORIZON_STAGE (start|end) step=(\d+) stage=(\w+).*?wall_time=([\d.]+)"
    )
    for match in pattern.finditer(log):
        event, step, stage, stamp = match.groups()
        key = (int(step), stage)
        if event == "start":
            pending[key] = float(stamp)
        elif key in pending:
            start = pending.pop(key)
            end = float(stamp)
            if end < start:
                raise ValueError("stage clock moved backwards")
            intervals.append((key[0], stage, start, end))
    return intervals


def gpu_samples(path, timezone):
    samples = {}
    with Path(path).open() as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            if len(row) != 5:
                continue
            stamp, index, uuid, utilization, memory = (part.strip() for part in row)
            when = (
                datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f")
                .replace(tzinfo=ZoneInfo(timezone))
                .timestamp()
            )
            # A single nvidia-smi query can report sub-millisecond differences
            # between devices; group in one-second buckets and require all GPUs.
            bucket = int(when)
            samples.setdefault(bucket, {})[int(index)] = float(utilization.split()[0])
    return samples


def summarize(intervals, samples, trainer_index, gpu_count, threshold):
    result = []
    for step, stage, start, end in intervals:
        if stage not in ("old_log_prob", "update_actor"):
            continue
        selected = [
            values
            for stamp, values in samples.items()
            if start <= stamp <= end
            and len(values) == gpu_count
            and trainer_index in values
        ]
        active = [values for values in selected if values[trainer_index] >= threshold]
        overlap = sum(
            any(
                value >= threshold
                for index, value in values.items()
                if index != trainer_index
            )
            for values in active
        )
        result.append(
            {
                "step": step,
                "stage": stage,
                "seconds": end - start,
                "samples": len(selected),
                "trainer_active_samples": len(active),
                "overlap_samples": overlap,
                "overlap_fraction_of_trainer_active": (
                    overlap / len(active) if active else None
                ),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--gpu-csv", required=True)
    parser.add_argument("--gpu-timezone", required=True)
    parser.add_argument("--trainer-index", type=int, required=True)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=20)
    parser.add_argument("--output")
    args = parser.parse_args()
    stages = stage_intervals(Path(args.log).read_text(errors="replace"))
    if not stages:
        raise SystemExit("No completed timestamped stages; overlap is unqualified")
    report = {
        "scope": "GPU activity during explicit old-logprob/actor stages; not Tensor Core efficiency",
        "gpu_timezone": args.gpu_timezone,
        "trainer_index": args.trainer_index,
        "threshold_percent": args.threshold,
        "stages": summarize(
            stages,
            gpu_samples(args.gpu_csv, args.gpu_timezone),
            args.trainer_index,
            args.gpu_count,
            args.threshold,
        ),
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(output)
    print(output)


if __name__ == "__main__":
    main()
