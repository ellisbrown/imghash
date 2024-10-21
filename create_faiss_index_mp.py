import os
import numpy as np
import imagehash
from PIL import Image
import faiss
from tqdm import tqdm
import pickle
from multiprocessing import Pool, cpu_count
import multiprocessing

# Parameters
HASH_SIZE = 8        # Adjust as needed (e.g., 8, 16)
CHUNK_SIZE = 10000   # Number of images to process in each chunk
NUM_WORKERS = cpu_count()  # Number of parallel processes
INDEX_FILENAME = "faiss_index_train_phash.index"  # File to save the FAISS index

# Base path where your images are stored
base_path = "/mnt/disks/storage/data/finetune_data/"
pickle_path = os.path.expanduser("~/image_paths.pkl")

# Load image paths
with open(pickle_path, "rb") as f:
    image_paths = pickle.load(f)
    print(f"Loaded {len(image_paths)} image paths")

image_paths = list(image_paths)

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

def process_images(image_paths_chunk):
    """Process a chunk of images and compute their hashes."""
    hashes = []
    for image_rel_path in image_paths_chunk:
        image_path = os.path.join(base_path, image_rel_path)
        try:
            with Image.open(image_path) as image:
                # Preprocess the image
                image = preprocess_image(image)
                # Compute the hash
                hash_int = compute_hash(image, hash_size=HASH_SIZE)
                hashes.append(hash_int)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    return hashes

def process_and_save_chunk(args):
    """Worker function to process a chunk and save the hashes."""
    chunk_id, image_paths_chunk, hash_files_dir = args
    hash_file = os.path.join(hash_files_dir, f"hashes_{chunk_id}.pkl")
    if os.path.exists(hash_file):
        print(f"Worker {chunk_id}: Hash file already exists. Skipping processing.")
        return hash_file
    print(f"Worker {chunk_id}: Processing {len(image_paths_chunk)} images")
    hashes = process_images(image_paths_chunk)
    with open(hash_file, 'wb') as f:
        pickle.dump(hashes, f)
    print(f"Worker {chunk_id}: Saved hashes to {hash_file}")
    return hash_file

def main():
    total_images = len(image_paths)
    num_chunks = (total_images + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Total number of images: {total_images}")
    print(f"Number of chunks: {num_chunks}")
    print(f"Number of workers: {NUM_WORKERS}")

    # Directory to store hash files permanently
    hash_files_dir = os.path.expanduser("~/hash_files")
    os.makedirs(hash_files_dir, exist_ok=True)
    print(f"Directory for hash files: {hash_files_dir}")

    # Prepare arguments for workers
    args_list = []
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, total_images)
        image_paths_chunk = image_paths[start_idx:end_idx]
        args_list.append((chunk_id, image_paths_chunk, hash_files_dir))

    # Use multiprocessing Pool to process chunks in parallel
    with Pool(processes=NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(process_and_save_chunk, args_list), total=num_chunks))

    # Collect all hashes from hash files
    all_hashes = []
    print("Loading hashes from hash files")
    for chunk_id in tqdm(range(num_chunks)):
        hash_file = os.path.join(hash_files_dir, f"hashes_{chunk_id}.pkl")
        if os.path.exists(hash_file):
            with open(hash_file, 'rb') as f:
                hashes = pickle.load(f)
                all_hashes.extend(hashes)
        else:
            print(f"Warning: Hash file {hash_file} is missing. Skipping.")

    # Convert all hashes to binary arrays
    print("Converting hashes to binary arrays")
    hashes_bin = hashes_to_binary_array(all_hashes, HASH_SIZE)

    # Build the FAISS index
    num_bits = HASH_SIZE * HASH_SIZE
    print(f"Creating a FAISS index with {num_bits}-bit binary hashes")
    index = faiss.IndexBinaryFlat(num_bits)
    index.add(hashes_bin)
    print(f"Total number of hashes in index: {index.ntotal}")

    # Save the FAISS index to disk
    faiss.write_index_binary(index, INDEX_FILENAME)
    print(f"Saved FAISS index to {INDEX_FILENAME}")

def hashes_to_binary_array(hashes, hash_size):
    """Convert integer hashes to binary arrays suitable for FAISS."""
    num_bits = hash_size * hash_size
    num_bytes = (num_bits + 7) // 8  # Calculate the number of bytes needed
    hashes_bin = np.zeros((len(hashes), num_bytes), dtype=np.uint8)
    for i, hash_int in enumerate(hashes):
        hash_bytes = int(hash_int).to_bytes(num_bytes, byteorder='big')
        hashes_bin[i] = np.frombuffer(hash_bytes, dtype=np.uint8)
    return hashes_bin

if __name__ == '__main__':
    main()
