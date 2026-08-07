"""
GPT-4o-mini classifier — takes transcript + acoustic features, returns the 9-field schema.
Cost: ~$0.0002–0.0005 per call, well under $0.003/min ceiling.
"""
import json
from openai import OpenAI
from src.config.settings import (
    OPENAI_API_KEY,
    CLASSIFIER_MODEL,
    GPT4O_MINI_INPUT_COST_PER_1M,
    GPT4O_MINI_OUTPUT_COST_PER_1M,
)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


SYSTEM_PROMPT = """You are an expert call center audio analyst. You classify emotional tone and detect background noise from transcripts and acoustic data.

CRITICAL RULES:
1. Do NOT classify emotional_tone as frustrated/upset/distressed based on loud or energetic speech alone. Base tone primarily on transcript CONTENT and word choice.
2. Do NOT mark background_noise_present=true based on poor audio quality (clipping, low SNR, distortion). Noise and quality are separate dimensions.
3. Distinguish: background_noise = environmental sounds (chatter, music, traffic). audio_quality = technical signal issues (clipping, echo, static, muffling).
4. speaker_overlap_present = two people speaking simultaneously enough to affect understanding — not just brief interruptions.
5. long_silence_present = unusually long dead air suggesting a call-flow or technical problem.
6. Return ONLY valid JSON, no extra text."""

USER_TEMPLATE = """Analyze this production call audio and classify the CUSTOMER's emotional state.

TRANSCRIPT:
{transcript}

ACOUSTIC FEATURES:
{features}

EMOTIONAL TONE DETECTION GUIDE — read carefully before classifying:
- Base tone on TRANSCRIPT CONTENT first, acoustic features second.
- upset signals: repeated "Hello?", "Are you a real person?", short angry bursts, complaints with anger, not getting acknowledgement despite repeated attempts.
- frustrated signals: "why is this taking so long?", sighing, "I can't believe this", impatient but not explosive.
- distressed signals: crying, panicking, "I don't know what to do", highly emotional overwhelm.
- satisfied signals: "thank you so much", "that's great", "perfect", politely completing a transaction successfully, customer whose needs are being met and who sounds pleased — even if the call is routine. A customer happily booking an appointment = satisfied, not neutral.
- neutral signals: flat matter-of-fact exchange, no positive or negative coloring, information-only calls with no emotional resolution.
- If the call is in a non-English language, classify based on vocal patterns (pitch, rhythm, repetition) and any translatable words. Assume standard conversational tone unless signals suggest otherwise.
- Use acoustic pitch_std_hz and rms_mean as SUPPORTING signals only.

Return a JSON object with exactly these fields and enum values:
{{
  "emotional_tone": "neutral|satisfied|frustrated|upset|distressed",
  "emotional_intensity": "low|medium|high",
  "background_noise_present": true/false,
  "background_noise_type": "<concise description or empty string>",
  "background_noise_severity": "none|low|medium|high",
  "audio_quality": "clear|slightly_impaired|severely_impaired",
  "speaker_overlap_present": true/false,
  "long_silence_present": true/false,
  "confidence": <0.0 to 1.0>
}}

Definitions:
- neutral: no clear positive/negative emotion — routine transactional exchange
- satisfied: pleased, relieved, appreciative, clearly positive
- frustrated: annoyed, impatient, dissatisfied without strong anger
- upset: clearly angry, agitated, strongly dissatisfied — often short clipped tone, complaints, repeated attempts to get attention
- distressed: overwhelmed, panicked, crying, highly emotional
- emotional_intensity low=subtle/mild, medium=clear and sustained, high=strong/escalated
- background_noise_severity none=no noise, low=audible but no interference, medium=occasional interference, high=materially impairs conversation
- background_noise_present: use `has_background_noise_signal` AND `non_speech_noise_rms` AND `spectral_flatness_in_silence`. If non_speech_noise_rms > 0.002 OR has_background_noise_signal=true, likely noise is present. High spectral_flatness_in_silence (>0.05) suggests static or broadband noise. clipping_detected=true with high clipping_ratio may indicate static bursts.
- speaker_overlap_present: use `speaker_overlap_score`. Score > 0.4 strongly suggests overlap. Score 0.25-0.4 is likely overlap. Below 0.2 is probably no overlap. Also check transcript for garbled/confused segments.
- long_silence_present: use `long_silence_max_seconds`. Flag as true only if > 5 seconds of continuous dead air — shorter pauses are normal turn-taking.
- audio_quality: judge ONLY on technical signal issues (distortion, clipping, echo, static, muffling, packet loss). Do NOT count background noise as impaired quality. clipping_ratio > 0.01 = severely impaired. clipping_ratio 0.001-0.01 = slightly impaired."""


def classify(transcript: str, acoustic_features: dict) -> tuple[dict, float]:
    """
    Returns (result_dict, cost_usd).
    """
    features_str = json.dumps(acoustic_features, indent=2)
    prompt = USER_TEMPLATE.format(transcript=transcript or "(no transcript)", features=features_str)

    client = _get_client()
    response = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    usage = response.usage
    cost_usd = (
        usage.prompt_tokens * GPT4O_MINI_INPUT_COST_PER_1M / 1_000_000
        + usage.completion_tokens * GPT4O_MINI_OUTPUT_COST_PER_1M / 1_000_000
    )

    result = json.loads(response.choices[0].message.content)
    return result, round(cost_usd, 6)