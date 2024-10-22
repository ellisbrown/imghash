import os
import numpy as np
import imagehash
from PIL import Image
import faiss
from tqdm import tqdm, trange
import pickle


# Path to your pickle file containing image paths
pickle_path = os.path.expanduser("~/image_paths.pkl")

# Load image paths
with open(pickle_path, "rb") as f:
    image_paths = pickle.load(f)
    print(f"Loaded {len(image_paths)} image paths")

image_paths = list(image_paths)

# Base path where your images are stored
base_path = "/mnt/disks/storage/data/finetune_data/"


# Parameters
HASH_SIZE = 8        # Adjust as needed (e.g., 8, 16)
BATCH_SIZE = 10000   # Number of images to process in each batchse
INDEX_FILENAME = "faiss_index_train_phash.index"  # File to save the FAISS index


def preprocess_image(image):
    """Preprocess the image: convert to grayscale and resize."""
    image = image.convert('L')            # Convert to grayscale
    image = image.resize((256, 256))      # Resize to a standard size
    return image

def compute_hash(image, hash_size=HASH_SIZE):
    """Compute the perceptual hash of an image."""
    hash_obj = imagehash.phash(image, hash_size=hash_size)
    hash_int = int(str(hash_obj), 16)
    return hash_int


def process_batch(image_paths_batch, base_path, hash_size=HASH_SIZE):
    """Process a batch of images and compute their hashes."""
    hashes = []
    for image_rel_path in image_paths_batch:
        image_path = os.path.join(base_path, image_rel_path)
        try:
            with Image.open(image_path) as image:
                # Preprocess the image
                image = preprocess_image(image)
                # Compute the hash
                hash_int = compute_hash(image, hash_size=hash_size)
                hashes.append(hash_int)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    return hashes


def hashes_to_binary_array(hashes, hash_size):
    """Convert integer hashes to binary arrays suitable for FAISS."""
    num_bits = hash_size * hash_size
    num_bytes = (num_bits + 7) // 8  # Calculate the number of bytes needed
    hashes_bin = np.zeros((len(hashes), num_bytes), dtype=np.uint8)
    for i, hash_int in enumerate(hashes):
        hash_bytes = int(hash_int).to_bytes(num_bytes, byteorder='big')
        hashes_bin[i] = np.frombuffer(hash_bytes, dtype=np.uint8)
    return hashes_bin


num_bits = HASH_SIZE * HASH_SIZE
print(f"Creating a FAISS index with {num_bits}-bit binary hashes")
index = faiss.IndexBinaryFlat(num_bits)

total_images = len(image_paths)
num_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE
print(f"Total number of images: {total_images}")
print(f"Total number of batches: {num_batches}")


for batch_idx in trange(num_batches, desc="Processing batches"):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, total_images)
    image_paths_batch = image_paths[start_idx:end_idx]
    print(f"Processing batch {batch_idx + 1}/{num_batches}, images {start_idx} to {end_idx - 1}")

    # Compute hashes for the batch
    hashes = process_batch(image_paths_batch, base_path, hash_size=HASH_SIZE)

    # Convert hashes to binary arrays
    hashes_bin = hashes_to_binary_array(hashes, HASH_SIZE)

    # Add hashes to the FAISS index
    index.add(hashes_bin)

    # Optionally, save the index periodically
    # index_partial_filename = f"faiss_index_batch_{batch_idx + 1}.index"
    # faiss.write_index_binary(index, index_partial_filename)
    # print(f"Saved partial FAISS index to {index_partial_filename}")

# Save the FAISS index to disk
faiss.write_index_binary(index, INDEX_FILENAME)
print(f"Saved FAISS index to {INDEX_FILENAME}")
