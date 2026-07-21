#!/usr/bin/env python3
"""
selector/scorer.py — Low Cost Analyzer + Change Score Fusion

优化: TorchCodec 批量解码 (减少 I/O roundtrip)。
"""

import torch
import torch.nn.functional as F
import json
import os
import sys
import time


class LowCostAnalyzer:
    """GPU-only 低成本分析器。"""

    def __init__(self, device="cuda", small_size=(160, 90)):
        self.device = device
        self.small_h, self.small_w = small_size

    @torch.no_grad()
    def resize_score(self, frame, prev):
        small = F.interpolate(frame.unsqueeze(0), size=(self.small_h, self.small_w),
                               mode="bilinear", align_corners=False).squeeze(0)
        prev_small = F.interpolate(prev.unsqueeze(0), size=(self.small_h, self.small_w),
                                    mode="bilinear", align_corners=False).squeeze(0)
        return float(torch.abs(small - prev_small).mean().cpu())

    @torch.no_grad()
    def hist_score(self, frame, prev):
        y_frame = (0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]).clamp(0, 1)
        y_prev = (0.299 * prev[0] + 0.587 * prev[1] + 0.114 * prev[2]).clamp(0, 1)
        h_f = torch.histc((y_frame * 255).long().float(), bins=256, min=0, max=255)
        h_p = torch.histc((y_prev * 255).long().float(), bins=256, min=0, max=255)
        h_f = h_f / (h_f.sum() + 1e-10)
        h_p = h_p / (h_p.sum() + 1e-10)
        return float(((h_f - h_p) ** 2 / (h_f + h_p + 1e-10)).sum() * 0.5)

    @torch.no_grad()
    def score(self, frame, prev):
        return 0.6 * self.hist_score(frame, prev) + 0.4 * self.resize_score(frame, prev)


def build_change_curve(video_path, device="cuda", batch_size=128, stride=1):
    """
    构建 change curve (Section 7)。

    Args:
        video_path: 视频路径
        device: cuda / cpu
        batch_size: TorchCodec 批量解码大小
        stride: 帧采样步长 (1=全帧, 5=每5帧取1帧, 加速)
    """
    sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    from decoder.gpu_reader import create_decoder

    print(f"[scorer] {os.path.basename(video_path)}")
    t0 = time.time()

    reader = create_decoder(video_path, device)
    reader.open()
    n_frames = reader.meta.num_frames if hasattr(reader, 'meta') else reader.total_frames
    fps = reader.meta.average_fps if hasattr(reader, 'meta') else reader.fps

    analyzer = LowCostAnalyzer(device=device)
    curve = [0.0]
    prev_tensor = None
    processed = 0

    # 批量读取 (TorchCodec: get_frames_played_at)
    if hasattr(reader, 'get_frame_batch'):
        all_indices = list(range(0, n_frames, stride))
        for batch_start in range(0, len(all_indices), batch_size):
            batch_idx = all_indices[batch_start:batch_start + batch_size]
            pts_list = [i / fps for i in batch_idx]
            frames = reader.decoder.get_frames_played_at(pts_list)

            for i, f in enumerate(frames):
                idx = batch_idx[i]
                t = f.data.float().div(255.0).to(device, non_blocking=True)
                if prev_tensor is not None:
                    s = analyzer.score(t, prev_tensor)
                    curve.append(round(s, 6))
                prev_tensor = t
                processed += 1

            if batch_start % (batch_size * 10) == 0:
                print(f"  {processed}/{len(all_indices)}...")
    else:
        # FFmpeg fallback: 逐帧
        for i, frame in enumerate(reader):
            if i % stride == 0:
                if prev_tensor is not None:
                    s = analyzer.score(frame.tensor, prev_tensor)
                    curve.append(round(s, 6))
                prev_tensor = frame.tensor
                processed += 1
            if processed % 500 == 0:
                print(f"  {processed}...")

    reader.close()
    elapsed = time.time() - t0
    print(f"  done: {n_frames} frames (stride={stride}), {elapsed:.1f}s ({processed/max(elapsed,0.01):.0f} fps)")

    meta = {
        "video": os.path.basename(video_path),
        "total_frames": n_frames,
        "fps": round(fps, 2),
        "stride": stride,
        "method": "gpu_resize_hist",
        "weights": {"hist": 0.6, "resize": 0.4},
    }
    return curve, meta


def save_change_curve(curve, meta, out_path):
    data = {**meta, "curve": curve}
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved: {out_path}")
