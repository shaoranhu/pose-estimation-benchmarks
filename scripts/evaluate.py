"""Reproducible COCOeval keypoint evaluation of official YOLO26 checkpoints."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import time
from pathlib import Path


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['yolo26n-pose', 'yolo26s-pose', 'yolo26m-pose'])
    parser.add_argument('--limit', type=int, default=0, help='0 = full validation; otherwise sorted first N images, smoke only')
    args = parser.parse_args()
    cache, output = args.cache.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_dir = cache.parent / 'config'
    (config_dir / 'Ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import cv2
    import numpy as np
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO

    assert torch.cuda.is_available(), 'This protocol requires a CUDA GPU.'
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    annotation = cache / 'annotations/person_keypoints_val2017.json'
    coco = COCO(str(annotation))
    image_ids = sorted(coco.getImgIds())
    if args.limit:
        image_ids = image_ids[:args.limit]
    image_paths = [cache / 'val2017' / coco.imgs[i]['file_name'] for i in image_ids]
    assert all(p.exists() for p in image_paths)
    assert len(image_ids) == (args.limit or 5000)
    save_json(output / 'image_ids.json', image_ids)
    environment = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'python': platform.python_version(),
        'platform': platform.platform(), 'gpu': torch.cuda.get_device_name(0),
        'gpu_total_MiB': torch.cuda.get_device_properties(0).total_memory / 1024**2,
        'torch_cuda': torch.version.cuda, 'threads': torch.get_num_threads(),
        'packages': {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python', 'pycocotools']},
        'annotation_sha256': hashlib.file_digest(annotation.open('rb'), 'sha256').hexdigest(),
        'num_images': len(image_ids),
        'gt_annotations': sum(a['image_id'] in set(image_ids) for a in coco.anns.values()),
    }
    save_json(output / 'environment.json', environment)
    settings = dict(imgsz=640, device=0, half=False, conf=0.001, iou=0.7,
                    max_det=300, rect=False, augment=False, verbose=False, save=False, batch=1)
    save_json(output / 'protocol.json', {
        'predict': settings, 'end2end': True, 'input': '640x640 letterbox',
        'score': 'person box confidence; no keypoint rescoring; no extra OKS NMS',
        'evaluator': 'pycocotools.COCOeval(iouType=keypoints); default maxDets=20, OKS sigmas and area ranges',
        'warmup': 20, 'timing_repeats': 3, 'timing_images': min(100, len(image_ids)),
        'timing': 'batch=1 predict on predecoded BGR arrays, CUDA synchronized, excludes disk IO and visualization',
        'precision': 'FP32; TF32 disabled', 'split': 'COCO val2017, all images including no-person images',
    })
    # Fixed, predecoded images shared by all models for latency measurement.
    timing_images = [cv2.imread(str(p)) for p in image_paths[:100]]
    assert all(im is not None for im in timing_images)
    summary = []
    for model_name in args.models:
        print(f'START {model_name}', flush=True)
        folder = output / model_name
        folder.mkdir(exist_ok=True)
        weights = cache / 'weights' / (model_name + '.pt')
        model = YOLO(str(weights))
        checkpoint_params = sum(p.numel() for p in model.model.parameters())
        assert model.model.end2end, 'Expected YOLO26 end-to-end checkpoint'
        for i in range(20):
            model.predict(timing_images[i % len(timing_images)], **settings)
        fused_params = sum(p.numel() for p in model.predictor.model.model.parameters())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings, stages = [], []
        for repeat in range(3):
            for im in timing_images:
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = model.predict(im, **settings)[0]
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000)
                stages.append(result.speed)
        predictions = []
        visualized = 0
        start = time.perf_counter()
        for index, (image_id, path) in enumerate(zip(image_ids, image_paths)):
            result = model.predict(str(path), **settings)[0].cpu()
            assert result.keypoints.data.shape[1:] == (17, 3)
            boxes = result.boxes.xyxy.numpy()
            scores = result.boxes.conf.numpy()
            keypoints = result.keypoints.data.numpy()
            assert np.isfinite(keypoints).all() and np.isfinite(boxes).all()
            for box, score, kpts in zip(boxes, scores, keypoints):
                x1, y1, x2, y2 = box.tolist()
                predictions.append({'image_id': image_id, 'category_id': 1,
                                    'bbox': [x1, y1, x2-x1, y2-y1], 'score': float(score),
                                    'keypoints': kpts.reshape(-1).tolist()})
            if visualized < 6 and any(a['num_keypoints'] > 0 for a in coco.imgToAnns[image_id]):
                filtered = result[result.boxes.conf >= 0.25]
                filtered.save(str(folder / f'example_{image_id:012d}.jpg'))
                visualized += 1
            if (index + 1) % 250 == 0:
                print(f'{model_name}: {index+1}/{len(image_ids)} images', flush=True)
        evaluation_wall = time.perf_counter() - start
        pred_path = folder / 'predictions.json'
        # Compact JSON keeps the full 17x3 output without excessive disk overhead.
        pred_path.write_text(json.dumps(predictions), encoding='utf-8')
        assert predictions, 'No predictions; cannot evaluate.'
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            dt = coco.loadRes(str(pred_path))
            evaluator = COCOeval(coco, dt, 'keypoints')
            evaluator.params.imgIds = image_ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        (folder / 'cocoeval.txt').write_text(capture.getvalue(), encoding='utf-8')
        names = ['AP', 'AP50', 'AP75', 'AP_medium', 'AP_large', 'AR', 'AR50', 'AR75', 'AR_medium', 'AR_large']
        row = {
            'model': model_name, 'checkpoint_params': checkpoint_params, 'fused_inference_params': fused_params,
            'weights_MiB': weights.stat().st_size / 1024**2,
            'weights_sha256': hashlib.file_digest(weights.open('rb'), 'sha256').hexdigest(),
            'metrics': dict(zip(names, evaluator.stats.tolist())),
            'latency_ms_mean': float(np.mean(timings)), 'latency_ms_median': float(np.median(timings)),
            'latency_ms_p95': float(np.percentile(timings, 95)),
            'latency_repeat_means': np.mean(np.array(timings).reshape(3, -1), axis=1).tolist(),
            'stage_ms_mean': {key: float(np.mean([r[key] for r in stages])) for key in stages[0]},
            'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 1024**2,
            'peak_reserved_MiB': torch.cuda.max_memory_reserved() / 1024**2,
            'num_images': len(image_ids), 'num_predictions': len(predictions), 'evaluation_wall_s': evaluation_wall,
        }
        save_json(folder / 'metrics.json', row)
        save_json(folder / 'latency_samples_ms.json', timings)
        summary.append(row)
        save_json(output / 'summary.json', summary)
        print(json.dumps(row), flush=True)
        del model, result, filtered, dt, evaluator, predictions
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
