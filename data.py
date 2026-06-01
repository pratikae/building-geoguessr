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

def is_us(country_name: str) -> bool:
    return country_name in ("United States", "United States of America")

ds = load_dataset("Morris0401/Year-Guessr-Dataset")
all_data = ds["wiki_dataset"]

success_count = 0
failed_count = 0

# add all columns

print("Parsing coordinates and performing reverse geocoding...")
coordinates = []
for example in all_data:
    try:
        lat = parse_latitude(example["Latitude"]) # type: ignore
        lon = parse_longitude(example["Longitude"]) # type: ignore
        coordinates.append((lon, lat))
    except ValueError:
        coordinates.append((None, None))

countries = []
states = []

print("Performing reverse geocoding...")
# batch reverse geocoding, record all countries and states to add

ctr = 0
for data in gz.search(coordinates): # gz.search() returns a generator
    if data.result is None:
        countries.append(None)
        states.append(None)
        failed_count += 1
    else:
        countries.append(data.result.admin1)
        if is_us(data.result.admin1):
            states.append(data.result.admin2)
        else:
            states.append(None)
        success_count += 1
    
    ctr += 1
    if ctr % 1000 == 0:
        print(f"Processed {ctr} examples...")

# map the new columns back to the dataset, 
# an existing Country column exists
# use those if not None,  otherwise use the reverse geocoded country

all_data = all_data.map(lambda x, idx: {"Country": x["Country"] if x["Country"] is not None else countries[idx],
                                        "State": states[idx]}, with_indices=True)

all_data = all_data.filter(lambda x: x["Country"] is not None)
print(f"Reverse geocoding: {success_count} succeeded, {failed_count} failed.")

all_data = all_data.map(lambda x, idx: {"id": idx}, with_indices=True)
all_data = all_data.map(
    lambda x: {"is_us": 1 if x["Country"] in ("United States", "United States of America") else 0}
)


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