"""API 키 확인 + 오디오/프레임 추출 + 새 기능 테스트"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

AV_ASSETS_DIR = r"C:\Users\cik61\Desktop\hackathon\hackathon-final-2\av_assets"
VIDEO_DIR = os.path.join(AV_ASSETS_DIR, "video")
AUDIO_DIR = os.path.join(AV_ASSETS_DIR, "audio")
_HAS_AV_ASSETS = os.path.isdir(VIDEO_DIR) and os.path.isdir(AUDIO_DIR)


def test_api_key():
    """Gemini API 키 테스트"""
    print("=== 1. API 키 테스트 ===")
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "placeholder":
        print("[실패] GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 수정해주세요.")
        return False

    from google import genai
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=["Say 'Hello, VideoAgent!' in Korean"],
    )
    print(f"[성공] API 응답: {response.text.strip()}")
    return True


def test_frame_extraction():
    """프레임 추출 테스트"""
    print("\n=== 2. 프레임 추출 테스트 ===")
    if not _HAS_AV_ASSETS:
        print("[스킵] av_assets 폴더 없음")
        return True
    videos = list(Path(VIDEO_DIR).glob("*.mp4"))
    if not videos:
        print("[실패] 비디오 파일을 찾을 수 없습니다.")
        return False

    video_path = str(videos[0])
    print(f"테스트 비디오: {Path(video_path).name}")

    from frame_extractor import extract_frames_around, extract_frames_uniform

    frames = extract_frames_uniform(video_path, 5)
    print(f"[균등 추출] {len(frames)}장 추출 완료")
    for t, img in frames:
        print(f"  t={t:.1f}s, 크기={img.size}")

    frames2 = extract_frames_around(video_path, 5.0, window=1.0, interval=0.5)
    print(f"[시점 추출] t=5.0s 기준 {len(frames2)}장 추출 완료")

    return len(frames) > 0


def test_audio_files():
    """오디오 파일 존재 확인"""
    print("\n=== 3. 오디오 파일 확인 ===")
    if not _HAS_AV_ASSETS:
        print("[스킵] av_assets 폴더 없음")
        return True
    audios = list(Path(AUDIO_DIR).glob("*.wav"))
    videos = list(Path(VIDEO_DIR).glob("*.mp4"))

    print(f"비디오 파일: {len(videos)}개")
    print(f"오디오 파일: {len(audios)}개")

    matched = 0
    for v in videos:
        audio_path = Path(AUDIO_DIR) / f"{v.stem}.wav"
        if audio_path.exists():
            matched += 1
        else:
            print(f"  [경고] 오디오 없음: {v.name}")

    print(f"매칭된 쌍: {matched}/{len(videos)}")
    return matched > 0


def test_question_classifier():
    """질문 분류 테스트"""
    print("\n=== 4. 질문 분류 테스트 ===")
    from audio_analyzer import classify_question

    test_cases = [
        ("경적 소리가 들리는 순간 화면에 보이는 차량의 색상은?", "A"),
        ("영상에서 빨간색 물체는 몇 개인가?", "B"),
        ("사이렌이 울리는 동안 사람이 몇 명 보이는가?", "A"),
        ("영상에서 가장 먼저 등장하는 동물은?", "C"),
        ("How many people are visible when the horn sounds?", "A"),
    ]

    passed = 0
    for q, expected in test_cases:
        result = classify_question(q)
        ok = "OK" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        print(f"  [{ok}] '{q[:40]}...' => {result} (기대: {expected})")

    print(f"통과: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_extract_options():
    """보기 추출 테스트"""
    print("\n=== 5. 보기 추출 테스트 ===")
    from video_qa import extract_options

    test_cases = [
        # 형식 1: A) 텍스트
        (
            "질문?\nA) 흰색\nB) 검은색\nC) 빨간색\nD) 파란색",
            [("A", "흰색"), ("B", "검은색"), ("C", "빨간색"), ("D", "파란색")],
        ),
        # 형식 2: A. 텍스트
        (
            "What color?\nA. White\nB. Black\nC. Red",
            [("A", "White"), ("B", "Black"), ("C", "Red")],
        ),
        # 형식 3: (A) 텍스트
        (
            "몇 개?\n(A) 1개\n(B) 2개\n(C) 3개\n(D) 4개\n(E) 5개",
            [("A", "1개"), ("B", "2개"), ("C", "3개"), ("D", "4개"), ("E", "5개")],
        ),
    ]

    passed = 0
    for text, expected in test_cases:
        result = extract_options(text)
        letters_ok = [r[0] for r in result] == [e[0] for e in expected]
        count_ok = len(result) == len(expected)
        ok = letters_ok and count_ok
        if ok:
            passed += 1
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {len(result)}개 추출: {result}")

    print(f"통과: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_validate_answer():
    """답변 검증 테스트"""
    print("\n=== 6. 답변 검증 테스트 ===")
    from video_qa import validate_answer

    options = [("A", "흰색"), ("B", "검은색"), ("C", "빨간색")]

    test_cases = [
        ("A", options, "A"),
        ("B", options, "B"),
        ("c", options, "C"),    # 소문자
        ("Z", options, "A"),    # 유효하지 않은 보기 -> fallback
        ("", options, "A"),     # 빈 문자열 -> fallback
        ("A", None, "A"),       # options 없이
    ]

    passed = 0
    for answer, opts, expected in test_cases:
        result = validate_answer(answer, opts)
        ok = result == expected
        if ok:
            passed += 1
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] validate_answer('{answer}') => '{result}' (기대: '{expected}')")

    print(f"통과: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_validate_final_output():
    """최종 출력 검증 테스트"""
    print("\n=== 7. 최종 출력 검증 테스트 ===")
    from main import validate_final_output

    # 정상 케이스
    results = {i: {"answer": chr(65 + (i % 4))} for i in range(20)}
    output = validate_final_output(results, 20)
    ok1 = len(output) == 20 and output.isalpha()
    print(f"  [{'OK' if ok1 else 'FAIL'}] 정상 20문항: '{output}'")

    # 누락 케이스
    results2 = {0: {"answer": "B"}, 2: {"answer": "C"}}
    output2 = validate_final_output(results2, 5)
    ok2 = len(output2) == 5 and output2[0] == "B" and output2[1] == "A" and output2[2] == "C"
    print(f"  [{'OK' if ok2 else 'FAIL'}] 누락 포함 5문항: '{output2}'")

    passed = sum([ok1, ok2])
    print(f"통과: {passed}/2")
    return passed == 2


def test_build_options_verification():
    """보기 검증 프롬프트 생성 테스트"""
    print("\n=== 8. 보기 검증 프롬프트 생성 테스트 ===")
    from prompts import build_options_verification

    options = [("A", "흰색"), ("B", "검은색")]
    result = build_options_verification("차량 색상은?", options)
    ok1 = "Option A" in result and "Option B" in result
    print(f"  [{'OK' if ok1 else 'FAIL'}] 기본: {result[:80]}")

    result2 = build_options_verification("질문?", [])
    ok2 = "No options" in result2
    print(f"  [{'OK' if ok2 else 'FAIL'}] 빈 보기: {result2[:80]}")

    passed = sum([ok1, ok2])
    print(f"통과: {passed}/2")
    return passed == 2


if __name__ == "__main__":
    print("VideoAgent 테스트 시작\n")

    r2 = test_frame_extraction()
    r3 = test_audio_files()
    r4 = test_question_classifier()
    r5 = test_extract_options()
    r6 = test_validate_answer()
    r7 = test_validate_final_output()
    r8 = test_build_options_verification()

    # API 키 테스트
    r1 = test_api_key()

    print(f"\n{'='*40}")
    results = {
        "API 키": r1,
        "프레임 추출": r2,
        "오디오 파일": r3,
        "질문 분류": r4,
        "보기 추출": r5,
        "답변 검증": r6,
        "최종 출력": r7,
        "검증 프롬프트": r8,
    }
    for name, ok in results.items():
        print(f"{name}: {'OK' if ok else 'FAIL'}")

    if all(results.values()):
        print("\n모든 테스트 통과!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n실패 항목: {', '.join(failed)}")
