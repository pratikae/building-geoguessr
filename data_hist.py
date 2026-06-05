from collections import Counter
from datasets import load_from_disk
from dataset import DEFAULT_DATASET_PATH, LABEL_COLUMN_CANDIDATES, resolve_column

ds = load_from_disk(DEFAULT_DATASET_PATH)

counts = Counter()
for split_name, split in ds.items():
    if True:
        col = resolve_column(split, LABEL_COLUMN_CANDIDATES["country"])
        if col is None:
            raise RuntimeError(f"No country column found in split '{split_name}'")
        counts.update(split[col])

print(f"{'Country':<40} {'Count':>6}")
print("-" * 48)
for country, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"{str(country):<40} {count:>6}")

print(f"\nTotal: {sum(counts.values())} samples across {len(counts)} countries")
