import os
import argparse

from pathlib import Path
from typing import List, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

from torchcodec.decoders import AudioDecoder
from tqdm import tqdm


def scan_dir_fast(root: Path, exts: Set[str]) -> List[Path]:
    """Fast directory scanning using os.scandir."""
    stack = [root]
    results = []
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            if Path(entry.name).suffix.lower() in exts:
                                results.append(Path(entry.path))
                    except OSError:
                        continue
        except (OSError, PermissionError):
            continue
    return results


def gather_audio_files(paths: List[str], exts: Set[str]) -> List[Path]:
    """Collect audio files from given paths."""
    found = []
    seen = set()
    for p in tqdm(paths, desc="Gathering audio files", unit="path"):
        pth = Path(p).expanduser().resolve()
        if not pth.exists():
            continue
        if pth.is_file():
            if pth.suffix.lower() in exts:
                found.append(pth)
        else:
            for f in scan_dir_fast(pth, exts):
                rp = f.resolve()
                if rp not in seen:
                    seen.add(rp)
                    found.append(rp)
    return found


def get_duration_and_sample_rate(uri):
    """
    torchaudio.info is removed in torchaudio 2.9, so we have to use torchcodec...
    使用 torchcodec 获取音频的时长（秒）和采样率。
    优先使用 metadata.duration_seconds_from_header；
    如果为 None，则解码整段音频来计算时长。

    返回:
        (duration_in_seconds: float, sample_rate: int)
    """
    # 只创建解码器 + 读 metadata，这一步不会解完整个音频
    dec = AudioDecoder(uri)
    meta = dec.metadata

    # header 里给的采样率 & 时长（可能为 None）
    sample_rate = meta.sample_rate
    duration = getattr(meta, "duration_seconds_from_header", None)

    samples = None  # 用来缓存完整解码结果，避免多次 get_all_samples

    # 当 header 中没有时长，或者采样率也拿不到时，需要解码整段
    if duration is None or sample_rate is None:
        samples = dec.get_all_samples()  # 这里才真正把整段音频解成 tensor

        # 采样率优先从 metadata 拿，拿不到就从 samples 里拿
        if sample_rate is None:
            sample_rate = getattr(samples, "sample_rate", None)

        # 如果 AudioSamples 提供了 duration_seconds，直接用
        if hasattr(samples, "duration_seconds") and samples.duration_seconds is not None:
            duration = float(samples.duration_seconds)

        # 否则根据样本数和采样率计算
        if duration is None:
            if sample_rate is None:
                raise RuntimeError("无法从 metadata 或解码后的样本中确定 sample_rate，无法计算时长。")
            num_samples = samples.data.shape[-1]  # [channels, num_samples]
            duration = num_samples / float(sample_rate)

    # 再兜底一次采样率（极端情况下 header 和上面的分支都没拿到）
    if sample_rate is None:
        if samples is None:
            samples = dec.get_all_samples()
        sample_rate = getattr(samples, "sample_rate", None)
        if sample_rate is None:
            raise RuntimeError("无法确定音频的 sample_rate。")

    return float(duration), int(sample_rate)


def _check_one(args_tuple):
    """
    Check whether an audio file is valid and return duration info.

    使用 torchcodec 的 AudioDecoder 来获取 metadata：
    - 只访问 decoder.metadata，不调用 get_all_samples()
    - 如果能拿到 sample_rate 与 duration_seconds / num_frames，则计算时长
    """

    path_str, min_duration = args_tuple
    p = Path(path_str)

    try:
        # 创建解码器：这一步会让 FFmpeg probe 文件并读取 header，
        # 但不会完整解码整段音频。
        duration, sr = get_duration_and_sample_rate(str(p))

        if duration is None or duration <= 0 or sr is None or sr <= 0:
            return False, 0.0

        # 检查最小时长
        if duration >= min_duration:
            return True, duration
        else:
            return False, duration

    except Exception as e:
        # 任何异常都视作该文件不可用（损坏 / 不支持等）
        print(f"Error checking file {path_str}: {e}")
        return False, 0.0


def write_lines(out_txt: Path, lines: List[str], append: bool = False):
    """Write lines to output file."""
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out_txt.open(mode, encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Parallel audio integrity checker (torchcodec.AudioDecoder + scandir)"
    )
    ap.add_argument("--paths", nargs="+", required=True, help="Input file or folder paths")
    ap.add_argument("--out", required=True, help="Output txt path")
    ap.add_argument(
        "--exts",
        default="wav,mp3,flac,aac,m4a,ogg,opus,wma,alac,aiff,aif,aifc",
        help="Audio file extensions (comma-separated)",
    )
    ap.add_argument("--workers", type=int, default=8, help="Number of processes")
    ap.add_argument("--flush-every", type=int, default=1000, help="Flush to disk every N lines")
    ap.add_argument(
        "--min-duration",
        type=float,
        default=10.0,
        help="Minimum audio duration in seconds (default: 10s)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    exts = {"." + e.strip().lower() for e in args.exts.split(",") if e.strip()}
    out_path = Path(args.out).expanduser().resolve()

    audio_files = gather_audio_files(args.paths, exts)
    if not audio_files:
        print("No audio files found.")
        write_lines(out_path, [], append=False)
        return

    print(f"Found {len(audio_files)} audio files, starting validation...")

    write_lines(out_path, [], append=False)
    valid_buf = []
    total_ok = total_bad = total_too_short = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_check_one, (str(p), args.min_duration)): str(p) for p in audio_files
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Validating",
            unit="file",
        ):
            path_str = futures[fut]
            try:
                ok, duration = fut.result()
            except Exception as e:
                print(f"Error processing file {path_str}: {e}")
                ok, duration = False, 0.0

            if ok:
                valid_buf.append(path_str)
                total_ok += 1
            else:
                # 区分：太短 vs. 损坏/不支持
                if duration > 0 and duration < args.min_duration:
                    total_too_short += 1
                else:
                    total_bad += 1

            if len(valid_buf) >= args.flush_every:
                write_lines(out_path, valid_buf, append=True)
                valid_buf.clear()

    if valid_buf:
        write_lines(out_path, valid_buf, append=True)

    print(
        f"Validation complete: ✅ {total_ok} valid, "
        f"❌ {total_bad} corrupted or unsupported",
        end="",
    )
    if args.min_duration > 0:
        print(f", ⏱️  {total_too_short} shorter than {args.min_duration} seconds.")
    else:
        print(".")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
