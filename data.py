# pip install datasets
# pip install python-gazetteer
#
# on windows: $env:PYTHONUTF8="1"

from datasets import DatasetDict, concatenate_datasets, load_dataset
from gazetteer import Gazetteer

gz = Gazetteer()

def parse_longitude(text: str) -> float:
    text = text.replace("°", "").strip()
    if text.endswith("E"):
        return float(text[:-1])
    elif text.endswith("W"):
        return -float(text[:-1])
    raise ValueError(f"Invalid longitude format: {text}")


def parse_latitude(text: str) -> float:
    text = text.replace("°", "").strip()
    if text.endswith("N"):
        return float(text[:-1])
    elif text.endswith("S"):
        return -float(text[:-1])
    raise ValueError(f"Invalid latitude format: {text}")


def coordinates_to_country(longitude, latitude):
    for data in gz.search([(longitude, latitude)]):
        if data.result is not None:
            return data.result.admin1
        raise ValueError(f"No country found for ({longitude}, {latitude})")


def coordinates_to_state(longitude, latitude):
    for data in gz.search([(longitude, latitude)]):
        if data.result is not None:
            return data.result.admin2
        raise ValueError(f"No state found for ({longitude}, {latitude})")

ds = load_dataset("Morris0401/Year-Guessr-Dataset")
all_data = concatenate_datasets([ds[split] for split in ds.keys()])

success_count = 0
failed_count = 0

def map_country(example):
    if example["Country"] is None:
        try:
            example["Country"] = coordinates_to_country(
                parse_longitude(example["Longitude"]),
                parse_latitude(example["Latitude"]),
            )
            global success_count
            success_count += 1
        except ValueError:
            global failed_count
            failed_count += 1
    return example

# add all columns
all_data = all_data.map(map_country)
all_data = all_data.filter(lambda x: x["Country"] is not None)
print(f"Reverse geocoding: {success_count} succeeded, {failed_count} failed.")

all_data = all_data.map(lambda x, idx: {"id": idx}, with_indices=True)
all_data = all_data.map(
    lambda x: {"is_us": 1 if x["Country"] in ("United States", "United States of America") else 0}
)

def map_state(example):
    if example["is_us"] == 1:
        try:
            return {"State": coordinates_to_state(
                parse_longitude(example["Longitude"]),
                parse_latitude(example["Latitude"]),
            )}
        except ValueError:
            pass
    return {"State": ""}

all_data = all_data.map(map_state)

# split into train/val/test based on year
test_ds     = all_data.filter(lambda x: int(x["Year"]) > 1945)
trainval_ds = all_data.filter(lambda x: int(x["Year"]) <= 1945)
trainval_split = trainval_ds.train_test_split(test_size=0.1, seed=493, shuffle=True)

new_ds = DatasetDict({
    "train":      trainval_split["train"],
    "validation": trainval_split["test"],
    "test":       test_ds,
})

new_ds.save_to_disk("resplit_year_guessr_dataset")