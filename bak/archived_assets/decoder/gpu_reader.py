#!/usr/bin/env python3
"""
decoder/gpu_reader.py — GPU Video Decoder Layer

优先 TorchCodec (decode → CPU tensor → 一次性 .to(cuda))。
Fallback: FFmpeg hwaccel cuda → rawvideo pipe → GPU tensor。

禁止: cv2.VideoCapture, ffmpeg subprocess循环抽帧到CPU numpy
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Optional
import os

try:
    from torchcodec.decoders import VideoDecoder as _TCDecoder
    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False


@dataclass
class GPUFrame:
    index: int
    timestamp: float
    tensor: torch.Tensor  # (C, H, W), device=cuda, dtype=float32, range[0,1]


class GPUVideoReader:
    """TorchCodec GPU 视频读取器。"""

    def __init__(self, path: str, device: str = "cuda"):
        self.path = path
        self.device = device
        self.decoder = None
        self.meta = None

    def open(self) -> GPUVideoReader:
        if not HAS_TORCHCODEC:
            raise ImportError("TorchCodec not available")
        self.decoder = _TCDecoder(self.path)
        self.meta = self.decoder.metadata
        print(f"[decoder] TorchCodec: {self.path}")
        print(f"  {self.meta.width}x{self.meta.height}  {self.meta.num_frames}fr  {self.meta.average_fps:.2f}fps")
        return self

    def __iter__(self):
        """逐帧迭代 → GPUFrame。"""
        n = self.meta.num_frames
        for i in range(n):
            f = self.decoder.get_frame_at(i)
            # Frame.data: (3, H, W) uint8 CPU → (3, H, W) float32 CUDA
            t = f.data.float().div(255.0).to(self.device, non_blocking=True)
            yield GPUFrame(index=i, timestamp=f.pts_seconds, tensor=t)

    def get_frame_batch(self, indices: list[int]) -> list[GPUFrame]:
        """批量读取指定帧号。"""
        pts_list = [i / self.meta.average_fps for i in indices]
        frames = self.decoder.get_frames_played_at(pts_list)
        result = []
        for idx, f in zip(indices, frames):
            t = f.data.float().div(255.0).to(self.device, non_blocking=True)
            result.append(GPUFrame(index=idx, timestamp=f.pts_seconds, tensor=t))
        return result

    def close(self):
        if self.decoder is not None:
            del self.decoder
            self.decoder = None
            torch.cuda.empty_cache()

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()


class FFmpegGPUDecoder:
    """FFmpeg hwaccel cuda → rawvideo pipe → GPU tensor (fallback)。"""

    def __init__(self, path: str, device: str = "cuda"):
        self.path = path
        self.device = device
        self.proc = None
        self.width = 0
        self.height = 0
        self.fps = 25.0
        self.total_frames = 0

    def open(self) -> FFmpegGPUDecoder:
        import subprocess as sp
        rp = sp.run(["ffprobe","-v","error","-select_streams","v:0",
            "-show_entries","stream=nb_frames,width,height,r_frame_rate",
            "-of","csv=p=0",self.path], capture_output=True, text=True)
        parts = rp.stdout.strip().split(",")
        self.width, self.height = int(parts[0]), int(parts[1])
        self.fps = float(parts[2].split("/")[0]) / float(parts[2].split("/")[1]) if "/" in parts[2] else float(parts[2])
        rd = sp.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",self.path], capture_output=True, text=True)
        self.total_frames = int(float(rd.stdout.strip()) * self.fps)
        print(f"[decoder] FFmpeg: {self.width}x{self.height}  ~{self.total_frames}fr  {self.fps:.2f}fps")
        return self

    def iter_frames(self):
        """Yields (index, GPUFrame) for every frame."""
        import subprocess as sp
        import numpy as np

        cmd = ["ffmpeg","-loglevel","error","-hwaccel","cuda","-hwaccel_output_format","cuda",
               "-i",self.path,"-vf","scale=256:256","-f","rawvideo","-pix_fmt","rgb24","-"]
        self.proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=8*1024**2)
        frame_bytes = self.width * self.height * 3
        idx = 0
        while True:
            raw = self.proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
            t = torch.from_numpy(arr).permute(2,0,1).float().div(255.0).to(self.device, non_blocking=True)
            yield idx, GPUFrame(index=idx, timestamp=idx/self.fps, tensor=t)
            idx += 1
        self.proc.wait()
        self.proc = None

    def close(self):
        if self.proc:
            self.proc.stdout.close(); self.proc.stderr.close(); self.proc.wait()
            self.proc = None
        torch.cuda.empty_cache()


def create_decoder(path: str, device: str = "cuda"):
    """工厂函数: TorchCodec 优先, FFmpeg fallback。"""
    if HAS_TORCHCODEC:
        return GPUVideoReader(path, device)
    else:
        return FFmpegGPUDecoder(path, device)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/dahe/miniconda3/envs/amaterasu/lib/python3.10/site-packages/tests/test.mp4"
    t0 = __import__("time").time()
    reader = create_decoder(path)
    reader.open()
    n = 0
    for frame in reader:
        n += 1
        if n % 500 == 0:
            print(f"  fr{frame.index}: {frame.tensor.shape} on {frame.tensor.device}")
        if n >= 5: break
    elapsed = __import__("time").time() - t0
    print(f"  {n} frames in {elapsed:.2f}s")
    reader.close()
