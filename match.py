import os
import numpy as np
import faiss
import pickle
from tqdm import tqdm
import json

# Parameters
HASH_SIZE = 8  # Should match the hash size used previously
THRESHOLD = 0  # Hamming distance threshold for matching
INDEX_FILENAME = "faiss_index_train_phash.index"  # Path to the FAISS index file
IMAGE_PATHS_PICKLE = os.path.expanduser("~/image_paths.pkl")  # Path to the training image paths
TEST_HASH_FILES_DIR = os.path.expanduser("~/test_hash_files")  # Directory where test hashes are saved
RESULTS_DIR = os.path.expanduser("~/match_results")  # Directory to save the results

# Create the results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the FAISS index
index = faiss.read_index_binary(INDEX_FILENAME)
print(f"Loaded FAISS index with {index.ntotal} entries")

# Load the training image paths
with open(IMAGE_PATHS_PICKLE, "rb") as f:
    image_paths = pickle.load(f)
    print(f"Loaded {len(image_paths)} training image paths")
image_paths = list(image_paths)

# Ensure that the number of image paths matches the number of entries in the FAISS index
assert len(image_paths) == index.ntotal, "Mismatch between training images and FAISS index entries"

def hashes_to_binary_array(hashes, hash_size):
    """Convert integer hashes to binary arrays suitable for FAISS."""
    num_bits = hash_size * hash_size
    num_bytes = (num_bits + 7) // 8  # Calculate the number of bytes needed
    hashes_bin = np.zeros((len(hashes), num_bytes), dtype=np.uint8)
    for i, hash_int in enumerate(hashes):
        hash_bytes = int(hash_int).to_bytes(num_bytes, byteorder='big')
        hashes_bin[i] = np.frombuffer(hash_bytes, dtype=np.uint8)
    return hashes_bin

# List of test datasets to process
# test_hash_files = [f for f in os.listdir(TEST_HASH_FILES_DIR) if f.endswith('_hashes.pkl')]
test_hash_files = [os.path.join(TEST_HASH_FILES_DIR, f) for f in os.listdir(TEST_HASH_FILES_DIR) if f.endswith('_hashes.pkl')]

# Sanity check with handful of train hash files -- should have 100% match
train_hash_files_dir = "/home/ebrown/hash_files"
# train_hash_files = [f for f in os.listdir(train_hash_files_dir) if f.endswith('0.pkl')]
train_hash_files = [os.path.join(train_hash_files_dir, f) for f in os.listdir(train_hash_files_dir) if f.endswith('50.pkl')]

# Combine test and train hash files for processing
test_hash_files.extend(train_hash_files)


# Initialize a dictionary to store results
match_results = {}
for hash_file in tqdm(test_hash_files, desc="Processing test datasets"):
    dataset_name = os.path.basename(hash_file).replace('_hashes.pkl', '')
    results_file = os.path.join(RESULTS_DIR, f"{dataset_name}_matches.json")
    # if os.path.exists(results_file):
    #     print(f"Match results for dataset '{dataset_name}' already exist. Skipping.")
    #     continue
    
    print(f"\nProcessing test dataset: {dataset_name}")

    # Load test hashes
    # with open(os.path.join(TEST_HASH_FILES_DIR, hash_file), 'rb') as f:
    with open(hash_file, 'rb') as f:
        test_hashes = pickle.load(f)
    total_test_images = len(test_hashes)
    print(f"Total test images: {total_test_images}")

    # Convert test hashes to binary arrays
    test_hashes_bin = hashes_to_binary_array(test_hashes, HASH_SIZE)

    # Perform range search in the FAISS index
    print("Performing FAISS range search...")
    result = index.range_search(test_hashes_bin, THRESHOLD)
    lims, distances, indices = result

    # Count the number of test images with at least one match
    n_matches_per_query = np.diff(lims)
    n_test_images_with_match = np.sum(n_matches_per_query > 0)
    print(f"Number of test images with at least one match: {n_test_images_with_match}")

    # Record matches
    test_matches = {}
    for i in range(len(test_hashes)):
        start = lims[i]
        end = lims[i+1]
        if start == end:
            continue  # No matches for this test image
        matched_training_indices = indices[start:end]
        matched_training_filenames = [image_paths[idx] for idx in matched_training_indices]
        # Store the matches for this test image
        test_matches[i] = {
            'matched_training_indices': matched_training_indices.tolist(),
            'matched_training_filenames': matched_training_filenames
        }

    # Save the match results for this dataset
    match_results[dataset_name] = {
        'total_test_images': int(total_test_images),
        'n_test_images_with_match': int(n_test_images_with_match),
        'test_matches': test_matches
    }

    # Save the matches to a human-readable JSON file
    with open(results_file, 'w') as f:
        json.dump(match_results[dataset_name], f, indent=4)
    print(f"Saved match results to {results_file}")

print("\nSummary of Matches:")
for dataset_name, data in match_results.items():
    total_test_images = data['total_test_images']
    n_test_images_with_match = data['n_test_images_with_match']
    print(f"{dataset_name}: {n_test_images_with_match}/{total_test_images} ({100 * n_test_images_with_match / total_test_images:.2f}%) test images have matches in the training set.")
