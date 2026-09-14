"""Extract frames at integer seconds, retaining the legacy six-digit frame names."""
import argparse
import json
import math
from pathlib import Path


def extract(video, output):
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f'Cannot read video metadata: {video}')
        output.mkdir(parents=True, exist_ok=True)
        entries = []
        for second in range(int(count / fps)):
            frame_index = int(second * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f'Cannot decode frame at {second} s in {video}')
            name = f'{second + 1:06d}.jpg'
            if not cv2.imwrite(str(output / name), frame):
                raise OSError(f'Cannot write {output / name}')
            entries.append({'frame': name, 'source_frame_index': frame_index,
                            'timestamp_seconds': frame_index / fps})
        if not entries:
            raise ValueError('Video is shorter than one second under the legacy extraction protocol.')
        return entries
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose an empty output directory to avoid mixing frame sets.')
    entries = extract(args.video, args.output)
    # Store outside the image directory because legacy notebook loaders iterate all files.
    manifest = args.output.with_name(args.output.name + '_manifest.json')
    manifest.write_text(json.dumps({'video_id': args.video.stem, 'frames': entries}, indent=2)+'\n')
    print(f'Wrote {len(entries)} frames and {manifest.name}')


if __name__ == '__main__':
    main()
