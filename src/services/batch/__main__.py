import sys
from src.services.batch import run_batch

if len(sys.argv) < 2:
    print("Usage: python -m src.services.batch <folder_or_zip>")
    sys.exit(1)

print(f"\nProcessing batch: {sys.argv[1]}\n")
batch = run_batch(sys.argv[1])

print(f"\n  {'─' * 80}")
print(f"  Batch complete   {batch.succeeded}/{batch.total} succeeded   {batch.failed} failed")
if batch.total_duration_seconds > 0:
    print(f"  Total cost:      ${batch.total_cost_usd:.6f}  |  Avg: ${batch.avg_cost_per_minute:.6f}/min")
print(f"  Total time:      {batch.total_processing_seconds:.1f}s")
if batch.output_csv:
    print(f"  Output:          {batch.output_csv}")
    print(f"                   {batch.output_json}")

sys.exit(0 if batch.failed == 0 else 1)