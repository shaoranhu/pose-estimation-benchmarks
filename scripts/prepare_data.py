"""Download public COCO validation data and official pose weights; resumable by file."""
import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    part = target.with_suffix(target.suffix + '.part')
    for attempt in range(3):
        try:
            print(f'Download {url}', flush=True)
            with urllib.request.urlopen(url, timeout=45) as response, part.open('wb') as out:
                count = 0
                last = time.monotonic()
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    count += len(chunk)
                    if time.monotonic() - last > 15:
                        print(f'{target.name}: {count / 1024**2:.1f} MiB', flush=True)
                        last = time.monotonic()
            part.replace(target)
            return
        except Exception as exc:
            print(f'Attempt {attempt + 1}: {exc}', flush=True)
            if attempt == 2:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    root = args.cache.resolve()
    sources = [
        ('https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip', root / 'annotations_trainval2017.zip'),
        ('https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip', root / 'val2017.zip'),
    ] + [(f'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26{s}-pose.pt', root / 'weights' / f'yolo26{s}-pose.pt') for s in 'nsm']
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: download(*item), sources[:2]))
    annotation = root / 'annotations/person_keypoints_val2017.json'
    if not annotation.exists():
        with zipfile.ZipFile(root / 'annotations_trainval2017.zip') as archive:
            archive.extract('annotations/person_keypoints_val2017.json', root)
    if len(list((root / 'val2017').glob('*.jpg'))) != 5000:
        with zipfile.ZipFile(root / 'val2017.zip') as archive:
            archive.extractall(root)
    for item in sources[2:]:
        download(*item)
    metadata = []
    for url, file in sources:
        metadata.append({'url': url, 'file': str(file), 'bytes': file.stat().st_size,
                         'sha256': hashlib.file_digest(file.open('rb'), 'sha256').hexdigest()})
    (root / 'download_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('READY: COCO val2017 5000 images, keypoint annotations, three weights.', flush=True)


if __name__ == '__main__':
    main()
