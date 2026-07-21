#!/usr/bin/env python3
"""02 — 心跳检测: 长段渐变检测 (DINO余弦). 输出补充切点."""
import json, sys, os, subprocess as sp
import numpy as np, torch, torch.nn.functional as F
from torchvision import transforms

output = sys.argv[1]
with open(os.path.join(output, "01_raw_cuts", "raw_cuts.json")) as f: d = json.load(f)

out_dir = os.path.join(output, "02_heartbeat")
os.makedirs(out_dir, exist_ok=True)

video = d["video"]; tf = d["total_frames"]; fps = d["fps"]
cuts = set(d["cuts"])

torch.backends.cudnn.benchmark = True
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
dinov2 = dinov2.to("cuda").eval().half()
TR = transforms.Compose([
    transforms.ToTensor(), transforms.Resize(256), transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

HB_GAP = 200; HB_COS = 0.70
HEARTBEAT = [0.20, 0.40, 0.60, 0.80]
bl = [0] + sorted(cuts) + [tf]
new_cuts = set()

for i in range(len(bl)-1):
    sf, ef = bl[i], bl[i+1]-1
    if ef - sf + 1 <= HB_GAP: continue
    sel = "+".join(["eq(n\\,%d)" % f for f in [sf, ef]])
    proc = sp.Popen(["ffmpeg","-hwaccel","cuda","-loglevel","error","-i",video,
        "-vf","select="+sel+",scale=224:224","-vsync","0","-f","rawvideo","-pix_fmt","rgb24","-"],
        stdout=sp.PIPE, bufsize=16*1024*1024)
    raw = proc.stdout.read(2 * 224 * 224 * 3); proc.terminate()
    if len(raw) < 2 * 224 * 224 * 3: continue
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(2, 224, 224, 3)
    t = torch.cat([TR(arr[j]).unsqueeze(0) for j in range(2)]).to("cuda").half()
    with torch.no_grad():
        e1, e2 = dinov2(t).chunk(2)
    cos = float(F.cosine_similarity(F.normalize(e1,dim=-1), F.normalize(e2,dim=-1)).item())
    if cos < HB_COS:
        dur = ef - sf
        for r in HEARTBEAT:
            new_cuts.add(sf + int(dur * r))
        print(f"  seg[{i}] {cos:.3f} < {HB_COS} -> split")

all_cuts = sorted(cuts | new_cuts)
print(f"  heartbeats: +{len(all_cuts) - len(cuts)}")

data = {"video": video, "total_frames": tf, "duration": d["duration"],
        "fps": fps, "width": d["width"], "height": d["height"], "cuts": all_cuts}
with open(os.path.join(out_dir, "heartbeat.json"), "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
