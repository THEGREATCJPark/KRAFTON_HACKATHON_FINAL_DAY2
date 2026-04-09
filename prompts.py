"""프롬프트 템플릿 모음"""

AUDIO_TIMESTAMP_PROMPT = """Listen to this audio carefully.

Find the EXACT time(s) in seconds when the specified sound/event occurs.

Sound to find: {sound_description}

IMPORTANT: You MUST respond using EXACTLY this format. No other text before or after.

TIMESTAMPS: [comma-separated seconds, e.g., 12.5, 25.0]
DESCRIPTION: [brief description of the sound at those moments]

If the sound cannot be found at all:
TIMESTAMPS: -1
DESCRIPTION: Cannot find the specified sound

Examples of correct responses:
TIMESTAMPS: 3.2, 15.7
DESCRIPTION: Horn honking sound heard twice

TIMESTAMPS: -1
DESCRIPTION: Cannot find the specified sound
"""

VISUAL_ANALYSIS_PROMPT = """You are an expert video frame analyst.

Below are consecutive frames extracted at 0.5-second intervals around timestamp {timestamp}s.
Images are in chronological order.

## QUESTION
{question}

## YOUR TASK
This is a multiple-choice question. Do NOT generate a free-form answer.
Instead, verify each option against the visual evidence:

{options_verification}

## ANALYSIS RULES
1. Examine every frame carefully and find the most relevant one
2. For counting: list each object one by one with its location
3. For colors: describe the exact color you see, then match to options
4. For direction/motion: compare positions across consecutive frames
5. For presence/absence: check ALL frames before answering "not present"
6. If uncertain between two options, choose the one with stronger visual evidence
7. For temporal questions (before/after, first/last, changes over time): compare the START and END of the video explicitly, noting what changed between them

## RESPONSE FORMAT (strictly follow this)
ANSWER: [single option letter only, e.g., A]
CONFIDENCE: [0.0-1.0]
REASONING: [2-3 lines of evidence-based justification]"""

FULL_VIDEO_PROMPT = """You are an expert video analyst.
Watch this ENTIRE video from start to finish with extreme attention to detail.

## QUESTION
{question}

## ANSWER OPTIONS ({num_options} choices)
{options_verification}

## YOUR TASK
This is a multiple-choice question with {num_options} options.
Before answering, you MUST:

1. IDENTIFY what the question is asking for precisely
2. WATCH the entire video and note every relevant moment chronologically
3. For counting questions: list each instance with its timestamp
4. For identification questions: find the clearest frame showing the target
5. For sequential questions (e.g., "the Nth occurrence"):
   list ALL occurrences in order, then select the Nth one
6. For brand/text recognition: look for logos, labels, text on objects
7. For temporal questions: compare START and END explicitly
8. VERIFY your answer against ALL options before selecting

## CRITICAL RULES
- Answer based ONLY on direct visual/audio observation
- Do NOT use general knowledge, popularity reasoning, or assumptions from video context
- When uncertain, prefer conservative answers (lower counts, "not visible", "unrecognizable")
- If the answer requires watching the full video, confirm you tracked the relevant element throughout
- Pay attention to small details: logos, text, colors, numbers

## ACCURACY RULES
- When counting: list each item individually with timestamp before giving total
- When identifying sequence/order: number each event [1], [2], [3]... then recount before selecting
- When reading text/brands: quote exactly what you see, describe color and position of the marking
- When uncertain between two options: choose the more conservative one
- Confidence below 0.7 means you should note what made you uncertain

## RESPONSE FORMAT
ANSWER: [single letter only]
CONFIDENCE: [0.0-1.0]
REASONING: [list the evidence chronologically, then explain your choice]"""

EXTRA_INSTRUCTIONS = {
    "counting_scene": """
SCENE COUNTING INSTRUCTIONS:
- Watch the ENTIRE video and note every camera angle/scene change
- Number each UNIQUE camera position or framing
- If the video returns to a previous angle, do NOT double-count
- List all unique angles with approximate timestamps, then give final count
- Be STRICT about what counts as 'distinct':
  Minor zoom, slight pan, brightness change within same position = SAME angle
  Only count truly DIFFERENT camera locations/positions
- When in doubt, MERGE rather than split
- After listing, review and consolidate similar viewpoints
- Common error: overcounting by treating small variations as new angles
""",
    "ordered_event": """
SEQUENTIAL EVENT TRACKING:
- Watch from beginning to end, do NOT skip any part
- List EVERY qualifying event with timestamp and detailed description:
  [1] 00:xx - description
  [2] 00:xx - description
  ...
- Apply ALL filters (e.g., "close-up only", "actively playing", "exclude X")
- After listing, RECOUNT from the beginning to verify
- Double-check each entry: does it truly meet ALL criteria?
- Then select the specific Nth one the question asks for
- Common error: miscounting by including non-qualifying events
- If uncertain about one entry, note it and check if removing it changes your final answer
""",
    "brand_ocr": """
BRAND/TEXT IDENTIFICATION:
- Find frames where the target object is most clearly visible
- Look for: brand name, logo, color scheme, model markings
- In REASONING, you MUST describe:
  (a) the tool's color and shape
  (b) exact text/logo you read from the video
  (c) where on the object the branding appears
- Do NOT use context reasoning (e.g., 'this brand is common for this task')
- Do NOT guess based on color alone
- If you cannot quote specific text from the tool, choose 'not visible/unrecognizable'
""",
    "audio_trigger": """
AUDIO-VISUAL CROSS-REFERENCE:
- Listen for the specific sound mentioned in the question
- Note the exact timestamp when it occurs
- Then examine the visual content at THAT PRECISE MOMENT
- Answer based on what is visible at the sound's timestamp
""",
}


def detect_extra_instruction(question: str) -> str:
    """질문 키워드로 추가 지시문 유형 결정"""
    q = question.lower()

    if any(kw in q for kw in [
        "distinct", "angle", "scene", "camera", "vantage",
        "composition", "framing", "앵글", "장면 수",
    ]):
        return "counting_scene"

    if any(kw in q for kw in [
        "first", "second", "third", "fourth", "fifth",
        "1st", "2nd", "3rd", "4th", "5th",
        "close-up", "only", "exclude", "actively",
        "첫 번째", "두 번째", "세 번째", "네 번째", "다섯 번째",
        "클로즈업", "제외",
    ]):
        return "ordered_event"

    if any(kw in q for kw in [
        "brand", "logo", "model", "manufacturer",
        "브랜드", "로고", "제조사", "상표",
        "text", "label", "name on",
        "텍스트", "라벨", "글자",
    ]):
        return "brand_ocr"

    if any(kw in q for kw in [
        "소리", "들리", "경적", "울리", "sound", "hear",
        "audio", "noise", "voice", "music",
    ]):
        return "audio_trigger"

    return ""


SOUND_EXTRACTION_PROMPT = """From the following question, identify what specific sound or audio event needs to be found.

Question: {question}

Reply with ONLY the sound description, nothing else. Be specific.
Example: "car horn honking", "whistle blowing", "siren sound", "applause"
"""


def build_options_verification(
    question_text: str, options: list[tuple[str, str]]
) -> str:
    """보기별 검증 지시문 생성. 10개 이상이면 간결 모드."""
    if not options:
        return "(No options detected - answer the question directly)"

    if len(options) >= 10:
        # 간결 모드: 나열만
        lines = []
        for letter, text in options:
            lines.append(f"{letter}) {text}")
        return "\n".join(lines)

    # 상세 모드
    lines = []
    for letter, text in options:
        lines.append(
            f"Option {letter} ({text}): Does the video evidence support this?"
        )
    return "\n".join(lines)
