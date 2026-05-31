# pip install datasets
# pip install python-gazetteer
# 
# on windows: $env:PYTHONUTF8="1"

from datasets import load_dataset, concatenate_datasets, DatasetDict
from gazetteer import Gazetteer 

gz = Gazetteer()

def coordinatesToCountry(longitude, latitude):
    coordinates = [(longitude, latitude)]
    for data in gz.search(coordinates): # gz.search() returns a generator
        if data.result is not None:
            return data.result.admin1
        raise ValueError(f"No country found for coordinates: {coordinates}")

# Load all existing splits
ds = load_dataset("Morris0401/Year-Guessr-Dataset")

# Combine all presplit data into one dataset
all_data = concatenate_datasets([ds[split] for split in ds.keys()])

def parse_longitude(text: str) -> float:
    # 73.8157417°E -> 73.8157417
    # Remove the degree symbol and the direction (E/W)
    # if the direction is W, we need to negate the longitude
    text = text.replace("°", "").strip()
    if text.endswith("E"):
        return float(text[:-1])
    elif text.endswith("W"):
        return -float(text[:-1])
    else:
        raise ValueError(f"Invalid longitude format: {text}")
    
def parse_latitude(text: str) -> float:
    # 18.5781108°N -> 18.5781108
    # Remove the degree symbol and the direction (N/S)
    # if the direction is S, we need to negate the latitude
    text = text.replace("°", "").strip()
    if text.endswith("N"):
        return float(text[:-1])
    elif text.endswith("S"):
        return -float(text[:-1])
    else:
        raise ValueError(f"Invalid latitude format: {text}")

success_reverse_geocoding = 0
failed_reverse_geocoding = 0

def add_country(example):
    if example["Country"] is None:
        try:
            example["Country"] = coordinatesToCountry(parse_longitude(example["Longitude"]), parse_latitude(example["Latitude"]) )
            global success_reverse_geocoding
            success_reverse_geocoding += 1
        except ValueError as e:
            global failed_reverse_geocoding
            failed_reverse_geocoding += 1
    return example

all_data = all_data.map(add_country)

# remove all entries where Country is still None
# we could not reverse geocode these entries, so we will remove them from the dataset
all_data = all_data.filter(lambda x: x["Country"] is not None)

# add an id column to the dataset, which is just the index of the entry in the dataset
all_data = all_data.map(lambda x, idx: {"id": idx}, with_indices=True)

# add an "is_us" column to the dataset, which is 1 if the country is United States, and 0 otherwise
# also allows "United States of America" as a valid country name for the United States
all_data = all_data.map(lambda x: {"is_us": 1 if x["Country"] == "United States" or x["Country"] == "United States of America" else 0})

print(f"Successfully reverse geocoded {success_reverse_geocoding} entries.")
print(f"Failed to reverse geocode {failed_reverse_geocoding} entries.")

# New test set: year > 1945
test_ds = all_data.filter(lambda x: int(x["Year"]) > 1945)

# Everything else goes to train/validation
trainval_ds = all_data.filter(lambda x: int(x["Year"]) <= 1945)

# Split remaining data into train/validation
trainval_split = trainval_ds.train_test_split(
    test_size=0.1,
    seed=493,
    shuffle=True,
)

new_ds = DatasetDict({
    "train": trainval_split["train"],
    "validation": trainval_split["test"],
    "test": test_ds,
})

# Save the new dataset
new_ds.save_to_disk("resplit_year_guessr_dataset")