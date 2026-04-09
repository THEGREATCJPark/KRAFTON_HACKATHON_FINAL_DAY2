"""VideoAgent 메인 실행 파일 - KRAFTON AI 해커톤"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Windows 콘솔 인코딩 문제 방지
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from dotenv import load_dotenv
from google import genai

from audio_analyzer import (
    classify_question,
    extract_sound_description,
    find_audio_file,
    find_timestamps,
    find_timestamps_from_video,
)
from frame_extractor import extract_frames_around, extract_frames_at_end, extract_frames_dense, extract_frames_uniform
from video_qa import (
    ask_with_frames,
    ask_with_video,
    extract_options,
    upload_file,
    validate_answer,
)

# 모델 상수 (쉽게 교체 가능)
MODEL_FAST = "gemini-3-flash-preview"
MODEL_STRONG = "gemini-3.1-pro-preview"

MODEL_FALLBACKS = {
    "gemini-3-flash-preview": ["gemini-2.5-flash", "gemini-2.0-flash"],
    "gemini-3.1-pro-preview": ["gemini-2.5-pro"],
}


def verify_model(client, model_name: str) -> str:
    """모델이 실제로 사용 가능한지 확인. 불가 시 fallback."""
    try:
        client.models.generate_content(model=model_name, contents=["test"])
        return model_name
    except Exception as e:
        print(f"  [경고] {model_name} 사용 불가: {e}")
        for fallback in MODEL_FALLBACKS.get(model_name, []):
            try:
                client.models.generate_content(model=fallback, contents=["test"])
                print(f"  [전환] {fallback} 사용")
                return fallback
            except Exception:
                continue
        print(f"  [경고] fallback 없음, {model_name} 그대로 시도")
        return model_name


# av_assets 경로
AV_ASSETS_DIR = r"C:\Users\cik61\Desktop\hackathon\hackathon-final-2\av_assets"
VIDEO_DIR = os.path.join(AV_ASSETS_DIR, "video")
AUDIO_DIR = os.path.join(AV_ASSETS_DIR, "audio")

# 전역 시간 관리
MAX_TOTAL_TIME = 14 * 60       # 전체 14분 (1분 여유)
MAX_PER_VIDEO = 180            # 비디오당 최대 180초 (3분)
FALLBACK_TIME = 12 * 60        # 12분 경과 시 미처리 건 즉시 fallback
MAX_UPLOAD_SIZE = 700 * 1024 * 1024  # 700MB 이상이면 업로드 대신 프레임 추출

# hard 질문 키워드 — scout 건너뛰고 무조건 full-video
HARD_KEYWORDS = [
    # brand/OCR
    "brand", "logo", "text", "model", "name", "label",
    "브랜드", "로고", "상표", "텍스트", "글자",
    # scene counting
    "distinct", "unique", "angle", "scene", "camera", "framing",
    "앵글", "장면 수", "카메라 위치",
    # ordered events
    "first", "second", "third", "fourth", "fifth", "last",
    "1st", "2nd", "3rd", "4th", "5th",
    "첫 번째", "두 번째", "세 번째", "네 번째", "다섯 번째", "마지막",
    # filtered conditions
    "close-up", "only", "exclude", "actively", "while",
    "클로즈업", "제외", "동안만",
    # complex counting
    "simultaneously", "at the same time", "동시에",
    # 기존 skip scout 키워드 통합
    "몇 명", "몇 대", "몇 개", "총 몇", "최대 수",
    "들리", "소리", "목소리", "경적", "음악", "울리", "안내방송",
    "자막", "워터마크", "번호",
    "등장하나", "등장하는", "보이나",
]


def is_hard_question(question: str) -> bool:
    """scout 건너뛰고 full-video로 가야 하는 질문인지 판단"""
    q = question.lower()
    return any(kw in q for kw in HARD_KEYWORDS)


def load_prompt(prompt_path: str) -> str:
    """프롬프트 파일 로드"""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_video_size(video_path: str) -> int:
    """비디오 파일 크기 반환 (정렬용)"""
    return os.path.getsize(video_path)


def scout_pass(
    video_path: str,
    question: str,
    options: list[tuple[str, str]] | None,
    client,
    model: str,
) -> dict:
    """프레임 기반 빠른 1차 판단 (비디오 업로드 없음)"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    # 30초 미만: 9장, 30초 이상: 19장
    if duration < 30:
        n_uniform, n_end = 6, 2
    else:
        n_uniform, n_end = 15, 3

    print(f"  [스카우트] 프레임 기반 빠른 판단 (영상 {duration:.0f}s, 균등 {n_uniform}장 + 끝 {n_end}장)")

    frames = extract_frames_around(video_path, 0.0, window=0.0, interval=1.0)  # 시작 1장
    frames += extract_frames_uniform(video_path, num_frames=n_uniform)
    frames += extract_frames_at_end(video_path, count=n_end, margin=1.5)

    # 중복 제거 (시간 기준 ±0.3초 이내)
    unique = []
    seen_times = []
    for t, img in frames:
        if not any(abs(t - st) < 0.3 for st in seen_times):
            unique.append((t, img))
            seen_times.append(t)
    frames = unique

    print(f"  [스카우트] {len(frames)}장 프레임 사용")

    result = ask_with_frames(
        frames, question, "전체 구간 (시작~끝)",
        model, client, options,
    )

    print(f"  [스카우트] 결과: {result['answer']}, 신뢰도: {result['confidence']:.2f}")
    return result


def process_single(
    video_path: str,
    question: str,
    client,
    model: str,
    idx: int,
    options: list[tuple[str, str]] | None = None,
) -> dict:
    """단일 비디오+질문 처리 (3단계 파이프라인)"""
    video_name = Path(video_path).name
    print(f"\n{'='*60}")
    print(f"[문항 {idx}] {video_name}")
    print(f"  [매칭] video{idx}.mp4 <-> prompt{idx}.txt")
    print(f"[질문] {question[:100]}...")
    if options:
        print(f"  [보기] {len(options)}개: {', '.join(f'{l})' + t[:15] for l, t in options)}")

    # Stage 1: 질문 분류 (hard → audio → easy 우선순위)
    result = None
    uploaded_video = None

    hard = is_hard_question(question)
    audio_type = classify_question(question)

    if hard:
        q_route = "HARD"
    elif audio_type == "A":
        q_route = "AUDIO"
    else:
        q_route = "EASY"
    print(f"  [분류] 경로={q_route} (hard={hard}, audio={audio_type=='A'})")

    if q_route == "HARD":
        # === HARD: full-video 직행 (brand/OCR, 순서, 장면수 등) ===
        print("  [경로] hard 질문 -> full-video 직행")

    elif q_route == "AUDIO":
        # === AUDIO: 오디오 퍼스트 서치 → 핀포인트 스나이핑 ===
        sound_desc = extract_sound_description(question, client, model)
        print(f"  [소리 설명] {sound_desc}")

        timestamps = []

        audio_path = find_audio_file(video_path, AUDIO_DIR)
        if audio_path:
            print(f"  [오디오] wav 파일 사용: {Path(audio_path).name}")
            timestamps = find_timestamps(audio_path, sound_desc, client, model)

        if not timestamps and not audio_path:
            file_size = os.path.getsize(video_path)
            if file_size <= MAX_UPLOAD_SIZE:
                print("  [오디오] wav 없음 -> 비디오 업로드하여 오디오 분석")
                try:
                    uploaded_video = upload_file(video_path, client)
                    timestamps = find_timestamps_from_video(uploaded_video, sound_desc, client, model)
                except Exception as e:
                    print(f"  [오디오] 비디오 기반 오디오 분석 실패: {e}")
            else:
                print(f"  [오디오] wav 없음 + 파일 {file_size//(1024*1024)}MB 초과 -> 스킵")

        if timestamps:
            ts = timestamps[0]
            frames = extract_frames_around(video_path, ts)
            if frames:
                result = ask_with_frames(frames, question, ts, model, client, options)

                if result["confidence"] < 0.7:
                    print(f"  [재시도] 신뢰도 {result['confidence']:.2f} < 0.7, 고밀도 프레임으로 재시도")
                    dense_frames = extract_frames_dense(video_path, ts)
                    retry = ask_with_frames(dense_frames, question, ts, model, client, options)
                    if retry["confidence"] > result["confidence"]:
                        result = retry

    else:
        # === EASY: scout 허용 (짧은 영상 + easy 질문) ===
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps > 0 else 0
        cap.release()

        if duration < 60:
            result = scout_pass(video_path, question, options, client, model)
            if result["confidence"] >= 0.85:
                print(f"  [스카우트] 신뢰도 {result['confidence']:.2f} >= 0.85, 채택")
                print(f"  [결과] 답: {result['answer']}, 신뢰도: {result['confidence']:.2f}")
                print(f"  [근거] {result['reasoning'][:150]}")
                return result
            print(f"  [스카우트] 신뢰도 {result['confidence']:.2f} < 0.85, full-video 재시도")
        else:
            print(f"  [경로] 영상 {duration:.0f}s -> full-video 직행")

    # === 폴백: A에서 실패 또는 B/C 스카우트 신뢰도 부족 또는 scout 스킵 ===
    if result is None or result["confidence"] < 0.85:
        file_size = os.path.getsize(video_path)

        if file_size > MAX_UPLOAD_SIZE:
            print(f"  [폴백] 파일 {file_size // (1024*1024)}MB > {MAX_UPLOAD_SIZE // (1024*1024)}MB, 프레임 추출 사용")
            frames = extract_frames_uniform(video_path, 30)
            if frames:
                result = ask_with_frames(frames, question, 0, model, client, options)
        else:
            print("  [폴백] 비디오 전체 업로드로 전환")
            try:
                result = ask_with_video(
                    video_path, question, model, client, options,
                    uploaded_ref=uploaded_video,
                )
            except Exception as e:
                print(f"  [폴백 실패] {e}")
                frames = extract_frames_uniform(video_path, 20)
                if frames:
                    result = ask_with_frames(frames, question, 0, model, client, options)

    if result is None:
        default_ans = options[0][0] if options else "A"
        result = {"answer": default_ans, "confidence": 0.0, "reasoning": "처리 실패"}

    print(f"  [결과] 답: {result['answer']}, 신뢰도: {result['confidence']:.2f}")
    print(f"  [근거] {result['reasoning'][:150]}")
    return result


def process_single_with_vote(
    video_path: str,
    question: str,
    client,
    model: str,
    idx: int,
    options: list[tuple[str, str]] | None = None,
) -> dict:
    """같은 질문을 3번 돌려서 다수결로 답 선택."""
    from collections import Counter
    answers = []
    all_results = []
    for trial in range(3):
        print(f"  [투표] {trial+1}/3 실행 중...")
        r = process_single(video_path, question, client, model, idx, options)
        answers.append(r["answer"])
        all_results.append(r)
        print(f"  [투표] {trial+1}/3 답: {r['answer']} (신뢰도: {r['confidence']:.2f})")

    counts = Counter(answers)
    winner, win_count = counts.most_common(1)[0]
    print(f"  [투표] 결과: {dict(counts)} -> 다수결: {winner} ({win_count}/3)")

    # 다수결 답을 낸 결과 중 confidence 가장 높은 것 사용
    best = max(
        (r for r in all_results if r["answer"] == winner),
        key=lambda r: r["confidence"],
    )
    best["confidence"] = min(1.0, best["confidence"] + 0.1 * (win_count - 1))
    return best


def process_single_with_timeout(
    video_path: str,
    question: str,
    client,
    model: str,
    idx: int,
    options: list[tuple[str, str]] | None = None,
    timeout: float = MAX_PER_VIDEO,
    use_vote: bool = False,
) -> dict:
    """타임아웃이 있는 단일 비디오 처리. 스레드가 늦게라도 답을 넣으면 살린다."""
    default_ans = options[0][0] if options else "A"
    container = {"result": None}

    def _run():
        try:
            if use_vote:
                container["result"] = process_single_with_vote(
                    video_path, question, client, model, idx, options,
                )
            else:
                container["result"] = process_single(
                    video_path, question, client, model, idx, options,
                )
        except Exception as e:
            print(f"  [에러] Q{idx} {e}")
            container["result"] = {
                "answer": default_ans,
                "confidence": 0.0,
                "reasoning": f"error: {e}",
            }

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # 타임아웃이지만 container에 중간 결과가 있을 수 있음
        if container["result"] is not None:
            print(f"  [타임아웃] Q{idx} {timeout:.0f}초 초과 - 중간 결과 사용 (답: {container['result'].get('answer', '?')})")
            return container["result"]
        print(f"  [타임아웃] Q{idx} {timeout:.0f}초 초과 - 답 없음, 기본값 사용")
        return {"answer": default_ans, "confidence": 0.5, "reasoning": "timeout"}

    # 정상 완료
    if container["result"] is not None:
        return container["result"]
    return {"answer": default_ans, "confidence": 0.0, "reasoning": "no result"}


def quick_fallback(
    video_path: str,
    question: str,
    client,
    options: list[tuple[str, str]] | None = None,
) -> dict:
    """시간 부족 시 비디오 통째로 Flash에 던지는 최소 처리"""
    default_ans = options[0][0] if options else "A"
    try:
        return ask_with_video(video_path, question, MODEL_FAST, client, options)
    except Exception:
        return {"answer": default_ans, "confidence": 0.0, "reasoning": "fallback failed"}


def load_pairs(test_folder: str, total: int = 20) -> list[tuple[str, str, int]]:
    """videoN.mp4 <-> promptN.txt를 숫자 기준으로 1:1 매칭.
    Returns: [(video_path, prompt_path, question_number), ...]
    question_number는 1-based (1, 2, ..., total).
    절대 정렬하지 않음. 숫자 순서대로."""
    test_path = Path(test_folder)
    pairs = []

    for i in range(1, total + 1):
        video = test_path / f"video{i}.mp4"
        prompt = test_path / f"prompt{i}.txt"

        if video.exists() and prompt.exists():
            pairs.append((str(video), str(prompt), i))
        else:
            # mp4 말고 다른 확장자 체크
            video_alt = list(test_path.glob(f"video{i}.*"))
            if video_alt and prompt.exists():
                pairs.append((str(video_alt[0]), str(prompt), i))

    return pairs


def validate_final_output(results: dict, total_questions: int = 20) -> str:
    """최종 답안 문자열 검증. results key는 1-based (1..total)."""
    answer_string = ""
    for i in range(1, total_questions + 1):
        if i in results and results[i].get("answer"):
            letter = results[i]["answer"].upper()
            if len(letter) == 1 and letter.isalpha():
                answer_string += letter
            else:
                answer_string += "A"
                print(f"  [경고] Q{i}: 유효하지 않은 답변 '{letter}' -> A로 대체")
        else:
            answer_string += "A"
            print(f"  [경고] Q{i}: 답변 없음 -> A로 대체")

    assert len(answer_string) == total_questions, \
        f"답안 길이 오류: {len(answer_string)} != {total_questions}"

    print(f"\n[검증 통과] {len(answer_string)}글자, 모두 알파벳")
    return answer_string


def main():
    parser = argparse.ArgumentParser(description="VideoAgent - KRAFTON AI 해커톤")
    parser.add_argument("test_folder", help="테스트 폴더 경로")
    parser.add_argument("--pro", action="store_true", help="Pro 모델 사용")
    parser.add_argument("--model", type=str, default=None, help="모델명 직접 지정")
    parser.add_argument("--total", type=int, default=20, help="총 문항 수 (기본 20)")
    parser.add_argument("--sequential", action="store_true", help="순차 처리 (기본은 병렬)")
    parser.add_argument("--vote", action="store_true", help="3회 투표 다수결 모드 (시간 3배)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "placeholder":
        print("[오류] GEMINI_API_KEY를 .env 파일에 설정해주세요")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    model = args.model or (MODEL_STRONG if args.pro else MODEL_FAST)
    model = verify_model(client, model)
    use_vote = args.vote
    per_video_timeout = 360 if args.pro else MAX_PER_VIDEO
    mode_desc = f"[설정] 모델: {model} (타임아웃: {per_video_timeout}s)"
    if use_vote:
        mode_desc += " (3회 투표)"
    print(mode_desc)

    # videoN <-> promptN 숫자 기준 매칭 (절대 정렬 안 함)
    pairs = load_pairs(args.test_folder, args.total)
    if not pairs:
        print("[오류] 비디오-프롬프트 쌍을 찾을 수 없습니다")
        sys.exit(1)

    # 사전 로드: 모든 prompt 읽고 options 추출 (qnum은 1-based)
    tasks = []
    for video_path, prompt_path, qnum in pairs:
        question = load_prompt(prompt_path)
        options = extract_options(question)
        tasks.append((qnum, video_path, question, options))
        print(f"  [로드] video{qnum}.mp4 <-> prompt{qnum}.txt ({len(options)}개 보기)")

    mode = "순차" if args.sequential else "병렬"
    print(f"[시작] {len(tasks)}개 문항 {mode} 처리 (제한: {MAX_TOTAL_TIME}초)")
    global_start = time.time()

    results = {}

    if args.sequential:
        for qnum, video_path, question, options in tasks:
            elapsed = time.time() - global_start

            if elapsed > FALLBACK_TIME:
                print(f"\n  [긴급] Q{qnum} 시간 부족 ({elapsed:.0f}s) - Flash 즉시 처리")
                results[qnum] = quick_fallback(video_path, question, client, options)
                continue

            remaining = MAX_TOTAL_TIME - elapsed
            remaining_count = len(tasks) - len(results)
            dynamic_timeout = min(per_video_timeout, remaining / max(remaining_count, 1))
            timeout = max(30, dynamic_timeout)

            print(f"  [시간] 경과: {elapsed:.0f}s, 남은: {remaining:.0f}s, 타임아웃: {timeout:.0f}s")

            results[qnum] = process_single_with_timeout(
                video_path, question, client, model, qnum, options,
                timeout=timeout, use_vote=use_vote,
            )
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            for qnum, video_path, question, options in tasks:
                future = executor.submit(
                    process_single_with_timeout,
                    video_path, question, client, model, qnum, options,
                    timeout=per_video_timeout, use_vote=use_vote,
                )
                futures[future] = (qnum, video_path, question, options)

            for future in as_completed(futures):
                qnum, video_path, question, options = futures[future]
                elapsed = time.time() - global_start

                try:
                    results[qnum] = future.result()
                except Exception as e:
                    print(f"  [에러] Q{qnum} {e}")
                    default_ans = options[0][0] if options else "A"
                    results[qnum] = {
                        "answer": default_ans,
                        "confidence": 0.0,
                        "reasoning": f"executor error: {e}",
                    }

                print(f"  [진행] {len(results)}/{len(tasks)} 완료 ({elapsed:.0f}s)")

        for qnum, video_path, question, options in tasks:
            if qnum not in results:
                print(f"  [긴급] Q{qnum} 미처리 - Flash fallback")
                results[qnum] = quick_fallback(video_path, question, client, options)

    total_elapsed = time.time() - global_start

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"[완료] 소요 시간: {total_elapsed:.1f}초")

    # 최종 검증 (1-based)
    answer_str = validate_final_output(results, args.total)
    print(f"[답안] {answer_str}")

    # 각 문항 요약 (1-based 순서)
    for i in range(1, args.total + 1):
        r = results.get(i, {})
        ans = r.get("answer", "A")
        conf = r.get("confidence", 0.0)
        print(f"  {i}번: {ans} (신뢰도: {conf:.2f})")

    return answer_str


if __name__ == "__main__":
    main()
