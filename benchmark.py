#!/usr/bin/env python3
"""
ASR WebSocket Streaming Benchmark

Streams WAV files to the ASR server via WebSocket and measures latency.
Matches the UI project's WebSocket protocol exactly.

Supports English models: 80ms CTC and 1040ms CTC.
Output: Text files organized by model into separate folders.
"""

import argparse
import asyncio
import base64
import json
import os
import re
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import websockets
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class BenchmarkResult:
    model: str
    wav_file: str
    audio_duration_s: float
    total_time_s: float
    time_to_first_transcript_s: float | None
    final_transcript: str
    transcripts: list[dict] = field(default_factory=list)
    rtf: float = 0.0  # Real-time factor

    def __post_init__(self):
        if self.audio_duration_s > 0:
            self.rtf = self.total_time_s / self.audio_duration_s


ASR_URL = os.getenv("ASR_URL", "wss://tekstiks.ee/asr/v2")

# English models
MODELS = {
    "80ms": "fastconformer_transducer_en_80ms",
    "1040ms": "fastconformer_transducer_en_1040ms",
}

# Match UI project settings
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 200  # 200ms chunks like the UI
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 3200 samples
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * 2  # 16-bit = 2 bytes per sample


def read_wav_file(file_path: str) -> tuple[bytes, int, int]:
    """Read WAV file and return (pcm_data, sample_rate, num_samples)."""
    with wave.open(file_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        if channels != 1:
            raise ValueError(f"Expected mono audio, got {channels} channels")
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit audio, got {sample_width * 8}-bit")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sample_rate}Hz. Please resample.")

        pcm_data = wf.readframes(num_frames)
        return pcm_data, sample_rate, num_frames


def pcm_to_base64(pcm_data: bytes) -> str:
    """Convert PCM bytes to base64 string (matching UI's btoa approach)."""
    return base64.b64encode(pcm_data).decode("ascii")


def format_transcript(text: str) -> str:
    """
    Format transcript text matching UI's textFormatting.ts.
    - Remove extra spaces before punctuation
    - Ensure space after punctuation
    - Normalize multiple spaces
    """
    if not text:
        return text
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Ensure space after punctuation (if not end of string)
    text = re.sub(r'([.,!?;:])(?!\s|$)', r'\1 ', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_for_comparison(text: str) -> str:
    """Normalize text for duplicate comparison (lowercase, no punctuation)."""
    # Remove punctuation and lowercase
    return re.sub(r'[.,!?;:\'"()-]', '', text.lower()).strip()


def is_duplicate_of_accumulated(new_text: str, accumulated: str, prev_text: str) -> bool:
    """
    Check if new_text is a duplicate/overlap of already accumulated content.
    Considers case and punctuation differences.
    Only checks for overlap at the END of accumulated text, not arbitrary substrings.
    """
    if not new_text.strip():
        return False

    new_norm = normalize_for_comparison(new_text)
    if not new_norm:
        return False

    # Check 1: Exact match with previous message (ignoring case/punctuation)
    prev_norm = normalize_for_comparison(prev_text)
    if new_norm == prev_norm:
        return True

    # Check 2: New text overlaps with END of accumulated text
    # (e.g., accumulated ends with "Russia's invasion" and new text is "Russia's invasion.")
    accumulated_norm = normalize_for_comparison(accumulated)
    words_new = new_norm.split()
    words_acc = accumulated_norm.split()

    if len(words_new) >= 1 and len(words_acc) >= 1:
        # Check if new text exactly matches last N words of accumulated
        if len(words_new) <= len(words_acc):
            if words_acc[-len(words_new):] == words_new:
                return True

        # Check if first words of new text match last words of accumulated (partial overlap)
        # Only skip if there's significant overlap (at least 2 words)
        for overlap_len in range(min(len(words_new), len(words_acc), 3), 1, -1):
            if words_acc[-overlap_len:] == words_new[:overlap_len]:
                return True

    return False


async def transcribe_file(
    wav_path: str,
    model_key: str,
    realtime: bool = True,
    verbose: bool = True,
    debug: bool = False,
    log_dir: Path | None = None,
) -> BenchmarkResult:
    """
    Transcribe a single WAV file with the specified model.
    Matches the UI project's WebSocket protocol exactly.
    """
    model_name = MODELS[model_key]
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_key} ({model_name})")
        print(f"File: {wav_path}")
        print(f"{'=' * 60}")

    pcm_data, sample_rate, num_samples = read_wav_file(wav_path)
    audio_duration_s = num_samples / sample_rate
    if verbose:
        print(f"Audio duration: {audio_duration_s:.2f}s")
        print(f"Chunk size: {CHUNK_DURATION_MS}ms ({SAMPLES_PER_CHUNK} samples)")

    transcripts: list[dict] = []
    all_text_parts: list[str] = []
    time_to_first: float | None = None
    session_ended = asyncio.Event()
    stream_ready = asyncio.Event()

    # Debug tracking
    prev_text = ""
    prev_time = 0.0
    msg_count = 0
    duplicate_count = 0
    log_lines: list[str] = []

    def log(msg: str):
        log_lines.append(msg)
        if debug:
            print(msg)

    start_time = time.perf_counter()

    async with websockets.connect(ASR_URL) as ws:
        if verbose:
            print(f"[{time.perf_counter() - start_time:.3f}s] WebSocket connected")

        # Send start message (matching UI exactly)
        start_msg = {
            "type": "start",
            "sample_rate": SAMPLE_RATE,
            "format": "pcm",
            "language": model_name,
        }
        await ws.send(json.dumps(start_msg))
        if verbose:
            print(f"[{time.perf_counter() - start_time:.3f}s] Sent start message")

        # Receiver task - runs concurrently with sender
        async def receive_messages():
            nonlocal time_to_first, prev_text, prev_time, msg_count, duplicate_count
            while not session_ended.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
                    data = json.loads(msg)
                    recv_time = time.perf_counter() - start_time

                    # Log raw message for debugging
                    if debug:
                        print(f"[DEBUG RAW] {msg[:200]}{'...' if len(msg) > 200 else ''}")

                    if data.get("type") == "ready":
                        if verbose:
                            print(f"[{recv_time:.3f}s] Stream ready, session_id: {data.get('session_id')}")
                        stream_ready.set()
                        continue

                    if data.get("type") == "error":
                        if verbose:
                            print(f"[{recv_time:.3f}s] Error: {data.get('message')}")
                        session_ended.set()
                        break

                    if data.get("type") == "transcript":
                        text = data.get("text", "")
                        msg_count += 1

                        if text == "[Session Ended]":
                            if verbose:
                                print(f"[{recv_time:.3f}s] Session ended")
                            if debug:
                                print(f"[DEBUG] Total messages: {msg_count}, duplicates: {duplicate_count}")
                            session_ended.set()
                            break

                        is_final = data.get("is_final", False)

                        # Debug logging
                        time_delta = (recv_time - prev_time) * 1000  # ms
                        is_duplicate = text.strip() == prev_text.strip()
                        is_near_dup = text.rstrip(".,!?") == prev_text.rstrip(".,!?")
                        log(f"[MSG] #{msg_count} t={recv_time:.3f}s dt={time_delta:.0f}ms final={is_final}")
                        log(f"      text: {repr(text)}")
                        if is_duplicate or is_near_dup:
                            duplicate_count += 1
                            log(f"      NOTE: duplicate detected (exact={is_duplicate}, near={is_near_dup})")

                        transcript_entry = {
                            "time": recv_time,
                            "text": text,
                            "is_final": is_final,
                        }
                        transcripts.append(transcript_entry)

                        if time_to_first is None and text.strip():
                            time_to_first = recv_time

                        if verbose:
                            status = "FINAL" if is_final else "partial"
                            print(f"[{recv_time:.3f}s] [{status}] {text}")

                        # Collect text - detect cumulative vs delta mode
                        accumulated_so_far = " ".join(all_text_parts)
                        if text.strip():
                            text_clean = text.strip()
                            # Check if this is cumulative mode (new text starts with accumulated)
                            acc_norm = normalize_for_comparison(accumulated_so_far)
                            text_norm = normalize_for_comparison(text_clean)
                            if acc_norm and text_norm.startswith(acc_norm):
                                # Cumulative mode - replace accumulated with full text
                                log(f"      CUMULATIVE (replacing accumulated)")
                                all_text_parts.clear()
                                all_text_parts.append(text_clean)
                            else:
                                is_dup = is_duplicate_of_accumulated(text, accumulated_so_far, prev_text)
                                if is_dup:
                                    log(f"      SKIPPED (duplicate of accumulated)")
                                    duplicate_count += 1
                                else:
                                    all_text_parts.append(text_clean)

                        prev_text = text
                        prev_time = recv_time

                except asyncio.TimeoutError:
                    if verbose:
                        print("Timeout waiting for message")
                    session_ended.set()
                    break
                except websockets.exceptions.ConnectionClosed:
                    session_ended.set()
                    break

        # Sender task - waits for ready, then streams audio
        async def send_audio():
            # Wait for stream_ready before sending any audio
            if verbose:
                print(f"[{time.perf_counter() - start_time:.3f}s] Waiting for stream ready...")

            try:
                await asyncio.wait_for(stream_ready.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                if verbose:
                    print("Timeout waiting for stream ready")
                session_ended.set()
                return

            if verbose:
                print(f"[{time.perf_counter() - start_time:.3f}s] Starting audio stream")

            # Stream audio in chunks (matching UI's 200ms chunks)
            offset = 0
            chunk_count = 0
            stream_start = time.perf_counter()
            chunk_timings: list[tuple[int, float, float]] = []  # (chunk_num, actual_time, drift)

            while offset < len(pcm_data) and not session_ended.is_set():
                chunk = pcm_data[offset : offset + BYTES_PER_CHUNK]
                offset += BYTES_PER_CHUNK
                chunk_count += 1

                # Send audio chunk as base64-encoded JSON (matching UI exactly)
                msg = {"type": "audio", "data": pcm_to_base64(chunk)}
                await ws.send(json.dumps(msg))

                elapsed = time.perf_counter() - stream_start
                expected_time = chunk_count * (CHUNK_DURATION_MS / 1000)
                drift = elapsed - expected_time
                chunk_timings.append((chunk_count, elapsed, drift))

                if realtime:
                    # Simulate real-time: wait until the right time for next chunk
                    sleep_time = expected_time - elapsed
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

            # Send stop signal
            await ws.send(json.dumps({"type": "stop"}))
            if verbose:
                print(f"[{time.perf_counter() - start_time:.3f}s] Sent stop signal ({chunk_count} chunks)")

            # Log chunk timing stats
            if chunk_timings:
                # Calculate inter-chunk intervals
                intervals = []
                for i in range(1, len(chunk_timings)):
                    interval = (chunk_timings[i][1] - chunk_timings[i-1][1]) * 1000
                    intervals.append(interval)

                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    min_interval = min(intervals)
                    max_interval = max(intervals)
                    log(f"[AUDIO] Total chunks: {chunk_count}, expected interval: {CHUNK_DURATION_MS}ms")
                    log(f"[AUDIO] Actual intervals: min={min_interval:.1f}ms avg={avg_interval:.1f}ms max={max_interval:.1f}ms")

        # Run sender and receiver concurrently
        receiver_task = asyncio.create_task(receive_messages())
        sender_task = asyncio.create_task(send_audio())

        # Wait for sender to complete
        await sender_task

        # Wait for session to end (with timeout)
        try:
            await asyncio.wait_for(session_ended.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            if verbose:
                print("Timeout waiting for session end")

        # Cancel receiver if still running
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass

        total_time = time.perf_counter() - start_time

    # Combine all text parts and format (matching UI's textFormatting)
    final_transcript = format_transcript(" ".join(all_text_parts))

    result = BenchmarkResult(
        model=model_key,
        wav_file=wav_path,
        audio_duration_s=audio_duration_s,
        total_time_s=total_time,
        time_to_first_transcript_s=time_to_first,
        final_transcript=final_transcript,
        transcripts=transcripts,
    )

    if verbose:
        print(f"\n--- Results ---")
        print(f"Total time: {result.total_time_s:.3f}s")
        if result.time_to_first_transcript_s:
            print(f"Time to first transcript: {result.time_to_first_transcript_s:.3f}s")
        print(f"RTF: {result.rtf:.3f}")
        print(f"Transcript: {result.final_transcript}")

    # Save log file if log_dir provided
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        wav_name = Path(wav_path).stem
        log_path = log_dir / f"{wav_name}_{model_key}.log"
        log(f"[SUMMARY] Total messages: {msg_count}, duplicates skipped: {duplicate_count}")
        log(f"[SUMMARY] Final transcript: {result.final_transcript}")
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        if verbose:
            print(f"Log saved: {log_path}")

    return result


def save_result(result: BenchmarkResult, output_dir: Path) -> Path:
    """Save transcription result to text file in model-specific folder."""
    model_dir = output_dir / result.model
    model_dir.mkdir(parents=True, exist_ok=True)

    wav_name = Path(result.wav_file).stem
    txt_path = model_dir / f"{wav_name}.txt"

    txt_path.write_text(result.final_transcript, encoding="utf-8")
    return txt_path


async def process_files(
    wav_files: list[Path],
    output_dir: Path,
    models: list[str],
    realtime: bool,
    verbose: bool,
    debug: bool = False,
    log_dir: Path | None = None,
) -> list[BenchmarkResult]:
    """Process multiple WAV files with multiple models."""
    results: list[BenchmarkResult] = []

    for wav_file in wav_files:
        for model_key in models:
            try:
                result = await transcribe_file(
                    str(wav_file),
                    model_key,
                    realtime=realtime,
                    verbose=verbose,
                    debug=debug,
                    log_dir=log_dir,
                )
                results.append(result)

                # Save transcript
                txt_path = save_result(result, output_dir)
                print(f"Saved: {txt_path}")

            except Exception as e:
                print(f"Error processing {wav_file} with {model_key}: {e}")

    return results


def find_wav_files(input_path: Path) -> list[Path]:
    """Find WAV files in input path (file or directory)."""
    if input_path.is_file():
        if input_path.suffix.lower() == ".wav":
            return [input_path]
        else:
            raise ValueError(f"Not a WAV file: {input_path}")
    elif input_path.is_dir():
        wav_files = sorted(input_path.glob("*.wav"))
        if not wav_files:
            raise ValueError(f"No WAV files found in: {input_path}")
        return wav_files
    else:
        raise ValueError(f"Path not found: {input_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="ASR WebSocket Streaming Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  %(prog)s test.wav -o results/

  # Bulk processing
  %(prog)s audio_folder/ -o results/

  # Single model only
  %(prog)s test.wav -o results/ --models 80ms

  # Fast mode (no real-time simulation)
  %(prog)s test.wav -o results/ --no-realtime
        """,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input WAV file or folder containing WAV files",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output",
        help="Output folder for transcripts (default: output)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Models to use (default: all)",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Send audio as fast as possible (not recommended)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed debug logging for server messages",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default=None,
        help="Directory to save detailed log files (e.g., logs/)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=200,
        help="Chunk duration in milliseconds (default: 200)",
    )
    args = parser.parse_args()

    # Update global chunk settings based on argument
    global CHUNK_DURATION_MS, SAMPLES_PER_CHUNK, BYTES_PER_CHUNK
    CHUNK_DURATION_MS = args.chunk
    SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
    BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * 2

    input_path = Path(args.input)
    output_dir = Path(args.output)

    try:
        wav_files = find_wav_files(input_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("=" * 60)
    print("ASR WebSocket Streaming Benchmark")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Files: {len(wav_files)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Chunk: {CHUNK_DURATION_MS}ms ({SAMPLES_PER_CHUNK} samples)")
    print(f"Real-time: {not args.no_realtime}")
    if args.debug:
        print(f"Debug: enabled")

    log_dir = Path(args.logs) if args.logs else None
    if log_dir:
        print(f"Logs: {log_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = await process_files(
        wav_files,
        output_dir,
        args.models,
        realtime=not args.no_realtime,
        verbose=not args.quiet,
        debug=args.debug,
        log_dir=log_dir,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'File':<30} {'Model':<8} {'Duration':<10} {'RTF':<8} {'First (s)':<10}")
    print("-" * 66)
    for r in results:
        fname = Path(r.wav_file).name[:28]
        first = f"{r.time_to_first_transcript_s:.3f}" if r.time_to_first_transcript_s else "N/A"
        print(f"{fname:<30} {r.model:<8} {r.audio_duration_s:<10.2f} {r.rtf:<8.3f} {first:<10}")

    print(f"\nOutput saved to: {output_dir.absolute()}")
    for model in args.models:
        model_dir = output_dir / model
        if model_dir.exists():
            count = len(list(model_dir.glob("*.txt")))
            print(f"  {model}/: {count} files")


if __name__ == "__main__":
    asyncio.run(main())
