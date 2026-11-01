# SPDX-License-Identifier: GPL-3.0-only
"""Decode every video/audio frame; optionally compare a same-seed reference MP4."""
import argparse
import json
import math
from itertools import zip_longest
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def audio_chunks(path, block=32768):
    pending = np.empty(0, dtype=np.float64)
    with av.open(str(path)) as context:
        for frame in context.decode(audio=0):
            raw = frame.to_ndarray()
            data = raw.astype(np.float64)
            if np.issubdtype(raw.dtype, np.integer):
                data /= max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max)
            data = data.T.reshape(-1) if frame.format.is_planar else data.reshape(-1)
            pending = np.concatenate((pending, data))
            while pending.size >= block:
                yield pending[:block]
                pending = pending[block:]
        if pending.size:
            yield pending


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contact", type=Path)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--frames", type=int, default=362)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    report = {"file": a.video.name, "bytes": a.video.stat().st_size}
    with av.open(str(a.video)) as c:
        if len(c.streams.video) != 1 or len(c.streams.audio) != 1:
            raise ValueError("Expected one video and one audio stream")
        v, audio = c.streams.video[0], c.streams.audio[0]
        report["video"] = {"codec": v.codec_context.name, "width": v.width, "height": v.height,
                           "fps": float(v.average_rate)}
        report["audio"] = {"codec": audio.codec_context.name, "sample_rate": audio.codec_context.sample_rate,
                           "channels": audio.codec_context.channels}
        if (v.width, v.height) != (a.width, a.height) or abs(float(v.average_rate) - 24) > .001:
            raise ValueError("Video geometry/FPS mismatch")
        if (audio.codec_context.sample_rate, audio.codec_context.channels) != (32000, 2):
            raise ValueError("Expected 32kHz stereo")
    chosen = (1, (a.frames + 1) // 2, a.frames)
    images = {}
    count = 0
    minimum_std = 255.
    ref_context = av.open(str(a.reference)) if a.reference else None
    ref_frames = ref_context.decode(video=0) if ref_context else None
    difference_sum = squared_sum = elements = maximum = 0
    try:
        with av.open(str(a.video)) as c:
            for count, frame in enumerate(c.decode(video=0), 1):
                rgb = frame.to_ndarray(format="rgb24")
                minimum_std = min(minimum_std, float(rgb.std()))
                if count in chosen:
                    images[count] = Image.fromarray(rgb)
                if ref_frames is not None:
                    ref = next(ref_frames).to_ndarray(format="rgb24")
                    if ref.shape != rgb.shape:
                        raise ValueError("Reference shape differs")
                    difference = rgb.astype(np.int16) - ref.astype(np.int16)
                    difference_sum += int(abs(difference).sum())
                    squared_sum += int(np.square(difference.astype(np.int32)).sum())
                    elements += rgb.size
                    maximum = max(maximum, int(abs(difference).max()))
            if count != a.frames:
                raise ValueError(f"Expected {a.frames} frames, decoded {count}")
            if ref_frames is not None and next(ref_frames, None) is not None:
                raise ValueError("Reference has more frames")
    finally:
        if ref_context:
            ref_context.close()
    report["video"].update(decoded_frames=count, duration_seconds=count / 24, minimum_frame_std=minimum_std)
    n = 0
    square = peak = 0.
    with av.open(str(a.video)) as c:
        for frame in c.decode(audio=0):
            raw = frame.to_ndarray()
            data = raw.astype(np.float64)
            if np.issubdtype(raw.dtype, np.integer):
                data /= max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max)
            if not np.isfinite(data).all():
                raise ValueError("Nonfinite audio")
            n += data.size
            square += float(np.square(data).sum())
            peak = max(peak, float(abs(data).max()))
    if not n or square / n < 1e-14:
        raise ValueError("Missing or silent audio")
    report["audio"].update(samples_per_channel=n // 2, rms=(square / n) ** .5, peak=peak)
    if elements:
        mse = squared_sum / elements
        report["reference_comparison"] = {"reference": a.reference.name, "decoded_pixels_bit_exact": maximum == 0,
            "mae_0_255": difference_sum / elements, "maximum_abs_0_255": maximum,
            "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
            "scope": "Decoded RGB pixel comparison, not LPIPS, latent equivalence or subjective quality"}
        audio_error = audio_ref_square = 0.0
        audio_max = 0.0
        aligned = True
        for left, right in zip_longest(audio_chunks(a.video), audio_chunks(a.reference)):
            if left is None or right is None or left.shape != right.shape:
                aligned = False
                break
            audio_error += float(np.square(left - right).sum())
            audio_ref_square += float(np.square(right).sum())
            audio_max = max(audio_max, float(abs(left - right).max()))
        report["reference_comparison"]["audio"] = {
            "sample_count_matches": aligned,
            "decoded_pcm_bit_exact": aligned and audio_max == 0,
            "nrmse": (audio_error / max(audio_ref_square, 1e-30)) ** .5 if aligned else None,
            "max_abs": audio_max if aligned else None,
            "scope": "Aligned decoded PCM comparison, not subjective audio quality"}
    if a.contact:
        a.contact.parent.mkdir(parents=True, exist_ok=True)
        sheet = Image.new("RGB", (1344, 284), "#161616")
        draw = ImageDraw.Draw(sheet)
        for index, number in enumerate(chosen):
            sheet.paste(images[number].resize((448, 256)), (448 * index, 28))
            draw.text((448 * index + 10, 8), f"Frame {number}", fill="white")
        sheet.save(a.contact, quality=92)
    report["limitations"] = "Full decode and non-silence checks plus optional RGB comparison; no subjective audio or broad perceptual evaluation."
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
