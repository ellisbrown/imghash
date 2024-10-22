import os
import jsonlines
import json
import pickle
from collections import Counter
from tqdm import tqdm


path = "/mnt/disks/storage/data/finetune_data/jsons/665k.jsonl"
pickle_path = os.path.expanduser("~/llava_image_paths.pkl")


image_paths = set()
if os.path.exists(pickle_path):
    with open(pickle_path, "rb") as f:
        image_paths = pickle.load(f)
        print(f"Loaded {len(image_paths)} image paths")

else:
    with jsonlines.open(path) as reader:
        for line in tqdm(reader):
            if "image" in line:
                image_paths.add(line["image"])

# save as a pickle 
with open(pickle_path, "wb") as f:
    pickle.dump(image_paths, f)
    print(f"Saved {len(image_paths)} image paths")

print(f"Total number of image paths: {len(image_paths)}")

import matplotlib.pyplot as plt

# Initialize counters
image_counter = Counter()
total_lines = 0
empty_lines = 0

# Path to the JSONL file
path = "/mnt/disks/storage/data/finetune_data/jsons/665k.jsonl"

# Read the JSONL file and count image paths
with jsonlines.open(path) as reader:
    for line in tqdm(reader):
        total_lines += 1
        if "image" in line:
            image_counter[line["image"]] += 1
        else:
            empty_lines += 1
            continue

# Print total number of images and lines
print(f"Total number of image paths: {sum(image_counter.values())}")
print(f"Total number of lines: {total_lines}")
print(f"Number of empty lines: {empty_lines}")
# num unique images
print(f"Number of unique images: {len(image_counter)}")

# max number of occurrences, min number of occurrences
max_occurrences = max(image_counter.values())
min_occurrences = min(image_counter.values())
print(f"Max number of occurrences: {max_occurrences}")
print(f"Min number of occurrences: {min_occurrences}")

# Display histogram of image path counts
plt.hist(image_counter.values(), bins=50, log=True)
plt.xlabel('Number of occurrences')
plt.ylabel('Number of image paths')
plt.title('Histogram of Image Path Counts')
# plt.show()
# save histogram
plt.savefig("image_path_histogram_llava.png")
print("Saved histogram as image_path_histogram.png")