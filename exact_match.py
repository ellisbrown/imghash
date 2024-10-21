import os
import numpy as np
import pickle
from tqdm import tqdm
import json

# Parameters
HASH_SIZE = 8  # Should match the hash size used previously
TRAIN_HASH_FILES_DIR = os.path.expanduser("~/hash_files")  # Directory where train hashes are saved
TEST_HASH_FILES_DIR = os.path.expanduser("~/test_hash_files")  # Directory where test hashes are saved
RESULTS_DIR = os.path.expanduser("~/match_results_exact")  # Directory to save the results
IMAGE_PATHS_PICKLE = os.path.expanduser("~/image_paths.pkl")  # Path to the training image paths

# Create the results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load all training image paths
with open(IMAGE_PATHS_PICKLE, "rb") as f:
    image_paths = pickle.load(f)
image_paths = list(image_paths)  # Ensure it's a list
total_train_images = len(image_paths)
print(f"Loaded {total_train_images} training image paths")

# Load all train hashes into a mapping from hash to list of indices
train_hash_files = [
    os.path.join(TRAIN_HASH_FILES_DIR, f)
    for f in os.listdir(TRAIN_HASH_FILES_DIR)
    if f.startswith('hashes_') and f.endswith('.pkl')
]
train_hash_to_indices = {}  # Map from hash to list of training image indices

print(f"Found {len(train_hash_files)} train hash files.")

current_index = 0  # Keep track of the current index in image_paths

print("Loading train hashes and building hash-to-indices mapping...")
for hash_file in tqdm(sorted(train_hash_files), desc="Loading train hashes"):
    with open(hash_file, 'rb') as f:
        hashes = pickle.load(f)
        num_hashes = len(hashes)
        # Get the corresponding image indices
        indices = list(range(current_index, current_index + num_hashes))
        current_index += num_hashes
        # Build the mapping
        for h, idx in zip(hashes, indices):
            h_int = int(h)
            if h_int in train_hash_to_indices:
                train_hash_to_indices[h_int].append(idx)
            else:
                train_hash_to_indices[h_int] = [idx]
print(f"Total unique train hashes loaded: {len(train_hash_to_indices)}")

# Process each test dataset
test_hash_files = [
    os.path.join(TEST_HASH_FILES_DIR, f)
    for f in os.listdir(TEST_HASH_FILES_DIR)
    if f.endswith('_hashes.pkl')
]

# Initialize a dictionary to store results
match_results = {}

for hash_file in tqdm(sorted(test_hash_files), desc="Processing test datasets"):
    dataset_name = os.path.basename(hash_file).replace('_hashes.pkl', '')
    results_file = os.path.join(RESULTS_DIR, f"{dataset_name}_matches.json")
    print(f"\nProcessing test dataset: {dataset_name}")

    # Load test hashes
    with open(hash_file, 'rb') as f:
        test_hashes = pickle.load(f)
    total_test_images = len(test_hashes)
    print(f"Total test images: {total_test_images}")

    # Check for exact matches
    n_test_images_with_match = 0
    test_matches = {}
    for i, test_hash in enumerate(test_hashes):
        test_hash_int = int(test_hash)  # Ensure test_hash is an integer
        if test_hash_int in train_hash_to_indices:
            n_test_images_with_match += 1
            # Record the matching training indices and filenames
            matched_training_indices = train_hash_to_indices[test_hash_int]
            matched_training_filenames = [image_paths[idx] for idx in matched_training_indices]
            test_matches[i] = {
                'matched_training_indices': matched_training_indices,
                'matched_training_filenames': matched_training_filenames
            }

    print(f"Number of test images with exact hash matches in the training set: {n_test_images_with_match}")

    # Save the match results for this dataset
    match_results[dataset_name] = {
        'total_test_images': total_test_images,
        'n_test_images_with_match': n_test_images_with_match,
        'test_matches': test_matches  # A dictionary mapping test indices to matching training images
    }

    # Save the matches to a human-readable JSON file
    with open(results_file, 'w') as f:
        json.dump(match_results[dataset_name], f, indent=4)
    print(f"Saved match results to {results_file}")

print("\nSummary of Matches:")
for dataset_name, data in match_results.items():
    total_test_images = data['total_test_images']
    n_test_images_with_match = data['n_test_images_with_match']
    percentage = (100 * n_test_images_with_match / total_test_images) if total_test_images > 0 else 0
    print(f"{dataset_name}: {n_test_images_with_match}/{total_test_images} ({percentage:.2f}%) test images have exact hash matches in the training set.")
