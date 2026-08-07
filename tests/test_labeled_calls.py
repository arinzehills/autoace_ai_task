"""
Validation script — runs all 3 labeled calls and prints accuracy, F1, and confusion matrix.
Run: python tests/test_labeled_calls.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.processors.audio_processor import analyze

LABELED_DIR = Path("data/labeled")
MANIFEST = LABELED_DIR / "labels.csv"

import csv

def load_labels() -> dict[str, dict]:
    labels = {}
    with MANIFEST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["name"]] = json.loads(row["result_json"])
    return labels


def run_validation():
    labels = load_labels()
    files = sorted(LABELED_DIR.glob("*.ogg"))

    fields = [
        "emotional_tone",
        "emotional_intensity",
        "background_noise_present",
        "background_noise_severity",
        "audio_quality",
        "speaker_overlap_present",
        "long_silence_present",
    ]

    per_field_correct = {f: 0 for f in fields}
    tone_predictions, tone_actuals = [], []
    total = len(files)

    print(f"\n{'─' * 70}")
    print(f"  Running validation on {total} labeled calls")
    print(f"{'─' * 70}\n")

    for audio_path in files:
        name = audio_path.name
        ground_truth = labels.get(name)
        if not ground_truth:
            print(f"  ⚠  No label found for {name}, skipping")
            continue

        print(f"  Analyzing {name}...")
        result = analyze(str(audio_path))
        pred = result.to_output_dict()

        tone_predictions.append(pred["emotional_tone"])
        tone_actuals.append(ground_truth["emotional_tone"])

        print(f"  {'Field':<30} {'Expected':<15} {'Predicted':<15} {'Match'}")
        print(f"  {'─' * 65}")
        for field in fields:
            expected = ground_truth.get(field)
            predicted = pred.get(field)
            match = expected == predicted
            if match:
                per_field_correct[field] += 1
            icon = "✓" if match else "✗"
            print(f"  {field:<30} {str(expected):<15} {str(predicted):<15} {icon}")
        print()

    # Overall accuracy per field
    print(f"\n{'─' * 70}")
    print(f"  PER-FIELD ACCURACY ({total} calls)")
    print(f"{'─' * 70}")
    total_correct = 0
    total_checks = 0
    for field in fields:
        acc = per_field_correct[field] / total
        total_correct += per_field_correct[field]
        total_checks += total
        print(f"  {field:<35} {per_field_correct[field]}/{total}  ({acc:.0%})")

    overall = total_correct / total_checks
    print(f"\n  Overall accuracy: {total_correct}/{total_checks} = {overall:.0%}")

    # Tone confusion matrix
    tone_classes = ["neutral", "satisfied", "frustrated", "upset", "distressed"]
    present = sorted(set(tone_actuals + tone_predictions))

    print(f"\n{'─' * 70}")
    print(f"  EMOTIONAL TONE CONFUSION MATRIX")
    print(f"  (rows = actual, cols = predicted)")
    print(f"{'─' * 70}")
    header = f"  {'':>12}" + "".join(f"{c:>12}" for c in present)
    print(header)
    for actual in present:
        row = f"  {actual:>12}"
        for pred_cls in present:
            count = sum(
                1 for a, p in zip(tone_actuals, tone_predictions)
                if a == actual and p == pred_cls
            )
            row += f"{count:>12}"
        print(row)

    # Tone accuracy
    tone_correct = sum(a == p for a, p in zip(tone_actuals, tone_predictions))
    print(f"\n  Tone accuracy: {tone_correct}/{total} = {tone_correct/total:.0%}")
    print(f"{'─' * 70}\n")


if __name__ == "__main__":
    run_validation()