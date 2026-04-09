"""Gemini API 호출 및 답변 파싱"""

import json
import re
import time

from google.genai import types
from PIL import Image

from prompts import (
    VISUAL_ANALYSIS_PROMPT,
    FULL_VIDEO_PROMPT,
    EXTRA_INSTRUCTIONS,
    build_options_verification,
    detect_extra_instruction,
)


def extract_options(prompt_text: str) -> list[tuple[str, str]]:
    """프롬프트에서 보기를 동적으로 추출. A-Z 모두 지원."""
    pattern = r'(?:^|\n)\s*\(?([A-Z])\s*[\):\.\]]\s*(.+?)(?=\n\s*\(?[A-Z]\s*[\):\.\]]|\n*$)'
    matches = re.findall(pattern, prompt_text, re.DOTALL)

    if not matches:
        pattern = r'([A-Z])\)\s*(.+?)(?:\n|$)'
        matches = re.findall(pattern, prompt_text)

    seen = set()
    options = []
    for letter, text in matches:
        if letter not in seen:
            seen.add(letter)
            options.append((letter, text.strip()))

    return sorted(options, key=lambda x: x[0])


def validate_answer(answer: str, options: list[tuple[str, str]] | None = None) -> str:
    """답변이 유효한 보기 문자인지 검증. 아니면 첫 번째 보기로 fallback."""
    if options:
        valid_letters = {opt[0] for opt in options}
        if answer.upper() in valid_letters:
            return answer.upper()
        return options[0][0]
    if len(answer) == 1 and answer.upper().isalpha():
        return answer.upper()
    return "A"


def upload_file(file_path: str, client) -> object:
    """Gemini Files API로 파일 업로드 및 처리 대기"""
    print(f"  [업로드] {file_path} 업로드 중...")
    uploaded = client.files.upload(file=file_path)

    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state.name == "FAILED":
        raise RuntimeError(f"파일 업로드 실패: {file_path}")

    print(f"  [업로드] 완료: {uploaded.name}")
    return uploaded


def _call_gemini_structured(client, model, contents, options=None, max_retries=3):
    """Gemini 호출 공통 함수 - structured output 우선, 재시도, fallback"""
    valid_letters = [opt[0] for opt in options] if options else list("ABCDE")

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Single letter from the given options",
                    "enum": valid_letters,
                },
                "confidence": {
                    "type": "number",
                    "description": "0.0 to 1.0",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Evidence from the video",
                },
            },
            "required": ["answer", "confidence", "reasoning"],
        },
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            data = json.loads(response.text)
            return {
                "answer": validate_answer(data.get("answer", "A"), options),
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": data.get("reasoning", ""),
                "raw": response.text,
            }
        except Exception as e:
            error_str = str(e).upper()
            if "429" in str(e) or "RATE_LIMIT" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = 2 ** attempt * 5
                print(f"  [Rate limit] {wait}초 대기 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            elif attempt < max_retries - 1:
                print(f"  [에러] 재시도 ({attempt+1}/{max_retries}): {e}")
                time.sleep(2)
                continue
            else:
                # 최종 시도: structured 없이 텍스트 fallback
                print(f"  [structured 최종 실패, fallback] {e}")
                try:
                    response = client.models.generate_content(
                        model=model, contents=contents,
                    )
                    return parse_response(response.text, options)
                except Exception as e2:
                    print(f"  [fallback도 실패] {e2}")
                    return parse_response("", options)

    return parse_response("", options)


def ask_with_frames(
    frames: list[tuple[float, Image.Image]],
    question: str,
    timestamp: float | str,
    model: str,
    client,
    options: list[tuple[str, str]] | None = None,
) -> dict:
    """프레임 이미지들 + 질문을 Gemini에 전송"""
    ts_label = timestamp if isinstance(timestamp, str) else f"{timestamp:.1f}"
    options_text = build_options_verification(question, options or [])
    prompt = VISUAL_ANALYSIS_PROMPT.format(
        timestamp=ts_label,
        question=question,
        options_verification=options_text,
    )
    prompt += "\n\nRespond in valid JSON format with keys: answer, confidence, reasoning"

    contents = []
    for t, img in frames:
        contents.append(f"[Frame at t={t:.1f}s]")
        contents.append(img)
    contents.append(prompt)

    return _call_gemini_structured(client, model, contents, options)


def ask_with_video(
    video_path: str,
    question: str,
    model: str,
    client,
    options: list[tuple[str, str]] | None = None,
    uploaded_ref=None,
) -> dict:
    """비디오 통째로 업로드하여 질의. uploaded_ref가 있으면 재업로드 생략."""
    video_file = uploaded_ref or upload_file(video_path, client)

    opts = options or []
    options_text = build_options_verification(question, opts)

    extra_key = detect_extra_instruction(question)
    extra_text = EXTRA_INSTRUCTIONS.get(extra_key, "")
    augmented_question = extra_text + "\n" + question if extra_text else question
    if extra_key:
        print(f"  [프롬프트] 추가 지시: {extra_key}")

    prompt = FULL_VIDEO_PROMPT.format(
        question=augmented_question,
        options_verification=options_text,
        num_options=len(opts) if opts else "unknown",
    )
    prompt += "\n\nRespond in valid JSON format with keys: answer, confidence, reasoning"

    return _call_gemini_structured(client, model, [video_file, prompt], options)


def ask_with_audio_and_frames(
    audio_path: str,
    frames: list[tuple[float, Image.Image]],
    question: str,
    timestamp: float,
    model: str,
    client,
    options: list[tuple[str, str]] | None = None,
) -> dict:
    """오디오 + 프레임을 함께 전송 (크로스모달 분석 강화)"""
    audio_file = upload_file(audio_path, client)

    options_text = build_options_verification(question, options or [])
    prompt = VISUAL_ANALYSIS_PROMPT.format(
        timestamp=f"{timestamp:.1f}",
        question=question,
        options_verification=options_text,
    )
    prompt += "\n\nRespond in valid JSON format with keys: answer, confidence, reasoning"

    contents = [audio_file]
    for t, img in frames:
        contents.append(f"[Frame at t={t:.1f}s]")
        contents.append(img)
    contents.append(prompt)

    return _call_gemini_structured(client, model, contents, options)


def parse_response(
    text: str, options: list[tuple[str, str]] | None = None
) -> dict:
    """응답에서 ANSWER, CONFIDENCE, REASONING 추출 (fallback용)"""
    result = {
        "answer": "",
        "confidence": -1.0,
        "reasoning": "",
        "raw": text,
    }

    # JSON 파싱 시도
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "answer" in data:
            return {
                "answer": validate_answer(data.get("answer", "A"), options),
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": data.get("reasoning", ""),
                "raw": text,
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # 정규 포맷 파싱
    answer_match = re.search(r"ANSWER:\s*([A-Za-z])", text)
    if answer_match:
        result["answer"] = answer_match.group(1).upper()

    conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
    if conf_match:
        try:
            result["confidence"] = float(conf_match.group(1))
        except ValueError:
            pass

    reason_match = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()

    # answer fallback
    if not result["answer"]:
        valid_letters = {opt[0] for opt in options} if options else set("ABCDE")

        ref_match = re.search(r'[\(\s]?([A-Za-z])\s*[\)\.\:]', text)
        if ref_match and ref_match.group(1).upper() in valid_letters:
            result["answer"] = ref_match.group(1).upper()

        if not result["answer"]:
            found = set()
            for ch in text:
                if ch.upper() in valid_letters:
                    found.add(ch.upper())
            if len(found) == 1:
                result["answer"] = found.pop()

    result["answer"] = validate_answer(result["answer"] or "A", options)

    if result["confidence"] < 0:
        result["confidence"] = 0.5

    if not result["reasoning"]:
        cleaned = text.strip()
        result["reasoning"] = cleaned[:200] if cleaned else "(no response)"

    return result
