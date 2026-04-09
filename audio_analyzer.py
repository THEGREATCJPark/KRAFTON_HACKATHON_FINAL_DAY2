"""오디오 분리 및 타임스탬프 추출"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from prompts import AUDIO_TIMESTAMP_PROMPT, SOUND_EXTRACTION_PROMPT

# ffmpeg 사용 가능 여부 (한번만 체크)
HAS_FFMPEG = shutil.which("ffmpeg") is not None

# 오디오 관련 키워드 (질문 유형 분류용)
AUDIO_KEYWORDS = [
    "소리", "울리", "들리", "사이렌", "경적", "호루라기", "효과음",
    "안내방송", "박수", "타이머", "경고음", "목소리", "벨소리",
    "비프음", "알림음", "노래", "음악", "연주", "부는", "치는",
    "sound", "hear", "ring", "horn", "whistle", "siren", "beep",
    "alarm", "bell", "music", "singing", "announce",
    "honk", "buzzer", "clap", "applause", "knock", "bang",
]


def _safe_print(msg: str) -> None:
    """Windows cp949 콘솔에서도 안전하게 출력"""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def classify_question(question: str) -> str:
    """질문 유형 분류: A(오디오), B(시각), C(시간/순서)"""
    q_lower = question.lower()

    for kw in AUDIO_KEYWORDS:
        if kw in q_lower:
            return "A"

    time_keywords = ["먼저", "나중", "처음", "마지막", "순서", "before", "after", "first", "last", "order"]
    for kw in time_keywords:
        if kw in q_lower:
            return "C"

    return "B"


def extract_audio_ffmpeg(video_path: str, output_path: str) -> str | None:
    """ffmpeg로 비디오에서 오디오 분리 (.wav). 실패 시 None 반환."""
    if not HAS_FFMPEG:
        return None
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def find_audio_file(video_path: str, audio_dir: str) -> str | None:
    """비디오에 대응하는 오디오 파일 찾기. 없으면 ffmpeg 추출 시도."""
    video_name = Path(video_path).stem
    video_parent = Path(video_path).parent

    # 1) av_assets/audio에서 같은 이름으로 찾기
    audio_path = os.path.join(audio_dir, f"{video_name}.wav")
    if os.path.exists(audio_path):
        return audio_path

    # 2) 비디오와 같은 디렉토리에서 같은 이름으로 찾기
    local_path = video_parent / f"{video_name}.wav"
    if local_path.exists():
        return str(local_path)

    # 3) videoN -> audioN 매핑
    num_match = re.search(r"(\d+)$", video_name)
    if num_match:
        audio_name = f"audio{num_match.group(1)}.wav"
        local_audio = video_parent / audio_name
        if local_audio.exists():
            return str(local_audio)

    # 4) ffmpeg로 추출 시도
    if HAS_FFMPEG:
        output_dir = video_parent / "output"
        output_path = str(output_dir / f"{video_name}.wav")
        extracted = extract_audio_ffmpeg(video_path, output_path)
        if extracted:
            _safe_print(f"  [오디오] ffmpeg로 추출 완료: {extracted}")
            return extracted

    return None


def extract_sound_description(question: str, client, model: str) -> str:
    """질문에서 찾아야 할 소리 설명 추출"""
    prompt = SOUND_EXTRACTION_PROMPT.format(question=question)
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
    )
    return response.text.strip()


def find_timestamps(audio_path: str, sound_description: str, client, model: str) -> list[float]:
    """오디오 파일에서 특정 소리의 타임스탬프 추출"""
    _safe_print(f"  [오디오 분석] '{sound_description}' 타임스탬프 검색 중...")

    audio_file = client.files.upload(file=audio_path)

    while audio_file.state.name == "PROCESSING":
        time.sleep(1)
        audio_file = client.files.get(name=audio_file.name)

    if audio_file.state.name == "FAILED":
        _safe_print("  [오디오 분석] 파일 업로드 실패")
        return []

    prompt = AUDIO_TIMESTAMP_PROMPT.format(sound_description=sound_description)
    response = client.models.generate_content(
        model=model,
        contents=[audio_file, prompt],
    )

    return _parse_timestamps(response.text)


def find_timestamps_from_video(video_file_ref, sound_description: str, client, model: str) -> list[float]:
    """이미 업로드된 비디오 파일에서 오디오 타임스탬프 추출.
    Gemini는 비디오의 오디오 트랙도 이해 가능."""
    _safe_print(f"  [비디오->오디오] '{sound_description}' 타임스탬프 검색 중...")

    prompt = AUDIO_TIMESTAMP_PROMPT.format(sound_description=sound_description)
    response = client.models.generate_content(
        model=model,
        contents=[video_file_ref, prompt],
    )

    return _parse_timestamps(response.text)


def _parse_timestamps(text: str) -> list[float]:
    """응답 텍스트에서 타임스탬프 파싱"""
    _safe_print(f"  [타임스탬프] 응답: {text[:200]}")

    timestamps = []
    match = re.search(r"TIMESTAMPS?:\s*(.+)", text)
    if match:
        ts_str = match.group(1).strip()
        if ts_str != "-1":
            for num in re.findall(r"[\d.]+", ts_str):
                try:
                    timestamps.append(float(num))
                except ValueError:
                    pass

    _safe_print(f"  [타임스탬프] 발견: {timestamps}")
    return timestamps
