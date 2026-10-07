"""Isolated-process latency benchmark; no accuracy rerun required."""
import argparse
import json
import os
import time
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    cache = a.cache.resolve()
    os.environ['YOLO_CONFIG_DIR'] = str(cache.parent / 'config')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    LOGGER.setLevel('ERROR')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    images = [cv2.imread(str(f)) for f in sorted((cache / 'val2017').glob('*.jpg'))[:100]]
    assert len(images) == 100 and all(im is not None for im in images)
    model = YOLO(str(cache / 'weights' / f'{a.model}.pt'))
    kw = dict(imgsz=640, device=0, half=False, conf=0.001, iou=0.7, max_det=300,
              rect=False, augment=False, verbose=False, save=False, batch=1)
    for i in range(50):
        model.predict(images[i % 100], **kw)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples, stages = [], []
    for repeat in range(3):
        for im in images:
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = model.predict(im, **kw)[0]
            torch.cuda.synchronize()
            samples.append(1000 * (time.perf_counter() - start))
            stages.append(result.speed)
    row = {'model': a.model, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'protocol': 'fresh process per model; FP32 TF32 off; 50 warmups; first 100 sorted val2017 images x 3; predecoded BGR; CUDA sync; logging disabled; predict API inclusive; excludes IO/rendering',
           'mean_ms': float(np.mean(samples)), 'median_ms': float(np.median(samples)),
           'p95_ms': float(np.percentile(samples, 95)),
           'repeat_means_ms': np.mean(np.array(samples).reshape(3,100), axis=1).tolist(),
           'stage_mean_ms': {k: float(np.mean([s[k] for s in stages])) for k in stages[0]},
           'peak_allocated_MiB': torch.cuda.max_memory_allocated()/1024**2,
           'peak_reserved_MiB': torch.cuda.max_memory_reserved()/1024**2,
           'samples_ms': samples}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / (a.model + '.json')).write_text(json.dumps(row, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in row.items() if k != 'samples_ms'}), flush=True)

if __name__ == '__main__':
    main()
