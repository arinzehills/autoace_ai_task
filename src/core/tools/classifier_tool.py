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
- upset signals: customer clearly not getting a response and escalating — "Hello? Hello? Hello?" repeated 3+ times means the customer IS upset trying to get attention (not neutral), "Are you a real person?", short terse bursts, agitated.
- frustrated signals: "why is this taking so long?", complaints without explosive anger, impatient sighing.
- distressed signals: crying, panic, "I don't know what to do", extreme overwhelm.
- satisfied signals: customer whose request is being actively fulfilled — booking an appointment cooperatively, completing a service request, politely engaging while needs are being met. Cooperative + successful outcome = satisfied. NOT neutral even if words are routine.
- neutral signals: pure information exchange with no clear emotional outcome — call ends incomplete or unresolved with no positive/negative signal.
- Repeated single-word utterances like "Hello? Hello?" from ONE person = upset, NOT speaker overlap.
- If the call switches languages mid-call, do NOT interpret transcription noise as emotional signals.
- long_silence_present: consider the call duration — a 7s pause in a 3-minute call may be an agent looking something up and is normal. Only flag true for silence that seems like dead air or technical failure relative to the call length.
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
- background_noise_present: check `non_speech_noise_rms` (> 0.001 = likely noise present), `has_background_noise_signal`, AND `spectral_flatness_in_silence` (> 0.05 = broadband/static noise in quiet periods) AND `spectral_flatness_in_speech` (> 0.03 = noise embedded in speech, e.g. static). `rms_kurtosis` > 10 often indicates impulsive background sounds (TV, traffic). Flag true if any combination suggests environmental noise.
- speaker_overlap_present: acoustic features cannot reliably detect overlap. Instead, look at the TRANSCRIPT for: garbled/nonsensical text that suggests two voices were recorded simultaneously, confusing mid-sentence switches, or text that looks like two voices blended (e.g. "12th of the 13th"). Also look at `rms_kurtosis` — very high values (>10) with background noise may indicate overlapping audio sources. Flag true if transcript shows confusion consistent with simultaneous speech.
- long_silence_present: use `long_silence_max_seconds`. Flag as true only if there is > 5 seconds of dead air that would feel unusually long relative to the total call duration. Brief pauses for typing or thinking are normal and should NOT be flagged.
- audio_quality: judge ONLY on technical signal issues (distortion, clipping, echo, static, muffling, packet loss). Do NOT count background noise as impaired quality. `clipping_ratio` > 0.01 = severely impaired. `clipping_ratio` 0.001-0.01 = slightly impaired. Otherwise clear.
- If the call involves a mid-call language switch (agent switches to Spanish or another language at customer request), do NOT interpret transcription artifacts from non-English speech as emotional frustration. Judge tone on behavioral signals only."""


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