"""특정 시간대 프레임 추출"""

import cv2
from PIL import Image


def extract_frames_around(
    video_path: str,
    timestamp: float,
    window: float = 2.0,
    interval: float = 0.5,
) -> list[tuple[float, Image.Image]]:
    """timestamp 기준 앞뒤 window초 구간에서 interval초 간격으로 프레임 추출"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [프레임 추출] 비디오 열기 실패: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    start_time = max(0, timestamp - window)
    end_time = min(duration, timestamp + window)

    frames = []
    t = start_time
    while t <= end_time:
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            # 타임스탬프 오버레이
            cv2.putText(
                frame,
                f"t={t:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((t, Image.fromarray(rgb)))
        t += interval

    cap.release()
    print(f"  [프레임 추출] {len(frames)}장 추출 (t={start_time:.1f}~{end_time:.1f}s)")
    return frames


def extract_frames_uniform(
    video_path: str,
    num_frames: int = 20,
) -> list[tuple[float, Image.Image]]:
    """비디오 전체에서 균등 간격으로 프레임 추출"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [프레임 추출] 비디오 열기 실패: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if total_frames < num_frames:
        num_frames = total_frames

    interval = total_frames / num_frames
    frames = []

    for i in range(num_frames):
        frame_num = int(i * interval)
        t = frame_num / fps if fps > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            cv2.putText(
                frame,
                f"t={t:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((t, Image.fromarray(rgb)))

    cap.release()
    print(f"  [프레임 추출] {len(frames)}장 균등 추출 (전체 {duration:.1f}s)")
    return frames


def extract_frames_dense(
    video_path: str,
    timestamp: float,
    window: float = 2.0,
    interval: float = 0.2,
) -> list[tuple[float, Image.Image]]:
    """고밀도 프레임 추출 (재시도용)"""
    return extract_frames_around(video_path, timestamp, window, interval)


def extract_frames_at_end(
    video_path: str,
    count: int = 2,
    margin: float = 1.0,
) -> list[tuple[float, Image.Image]]:
    """비디오 마지막 margin초 구간에서 count장 추출"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if duration <= 0:
        cap.release()
        return []

    start_time = max(0, duration - margin)
    interval = margin / max(count, 1)

    frames = []
    for i in range(count):
        t = start_time + i * interval
        frame_num = min(int(t * fps), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            cv2.putText(
                frame,
                f"t={t:.1f}s (END)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((t, Image.fromarray(rgb)))

    cap.release()
    return frames
