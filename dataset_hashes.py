import os
import numpy as np
import imagehash
from PIL import Image
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import pickle
from multiprocessing import Pool, cpu_count

# Parameters
HASH_SIZE = 8        # Same as your training data
NUM_WORKERS = cpu_count()  # Adjust as needed
HASH_FILES_DIR = os.path.expanduser("~/test_hash_files")  # Directory to save hash pickles

# Create the directory if it doesn't exist
os.makedirs(HASH_FILES_DIR, exist_ok=True)

# Test dataset mapping (update this mapping as needed)
test_dataset_map = dict(
    # name=("dataset_name", "version", "split", "image_key"
    ade=("SaiCharithaAkula21/benchmark_ade_manual", None, "train", "image"),
    ai2d=("lmms-lab/ai2d", None, "test", "image"),
    blink=("BLINK-Benchmark/BLINK", ["Counting", "IQ_Test", "Object_Localization", "Relative_Depth", "Relative_Reflectance", "Spatial_Relation"], "val", "image_1"),
    chartqa=("lmms-lab/ChartQA", None, "test", "image"),
    coco=("SaiCharithaAkula21/benchmark_coco_filtered", None, "train", "image"),
    docvqa=("lmms-lab/DocVQA", "DocVQA", "test", "image"),
    gqa=("lmms-lab/GQA", "testdev_balanced_images", "testdev", "image"),
    infovqa=("lmms-lab/DocVQA", "InfographicVQA", "test", "image"),
    mathvista=("AI4Math/MathVista", None, "testmini", "decoded_image"),
    mmbench_cn=("lmms-lab/MMBench_CN", "default", "dev", "image"),
    mmbench_en=("lmms-lab/MMBench_EN", None, "dev", "image"),
    mme=("lmms-lab/MME", None, "test", "image"),
    mmmu=("lmms-lab/MMMU", None, "validation", "image_1"),
    mmstar=("Lin-Chen/MMStar", None, "val", "image"),
    mmvet=("lmms-lab/MMVet", None, "test", "image"),
    mmvp=("MMVP/MMVP", None, "train", "image"),
    ocrbench=("echo840/OCRBench", None, "test", "image"),
    cvbench=("nyu-visionx/CV-Bench", None, "test", "image"),
    pope=("lmms-lab/POPE", None, "test", "image"),
    realworldqa=("lmms-lab/RealWorldQA", None, "test", "image"),
    sqa=("derek-thomas/ScienceQA", None, "test", "image"),
    seed=("lmms-lab/SEED-Bench", None, "test", "image"),
    stvqa=("lmms-lab/ST-VQA", None, "test", "image"),
    synthdog=("naver-clova-ix/synthdog-en", None, "validation", "image"),
    textvqa=("lmms-lab/textvqa", None, "validation", "image"),
    vizwiz=("lmms-lab/VizWiz-VQA", None, ("val", "test"), "image"),
    vstar=("craigwu/vstar_bench", None, "test", "image"),
)

def get_dataset(name):
    dataset_name, version, split = test_dataset_map[name][:3]

    if isinstance(version, list):
        datasets = []
        for v in version:
            datasets.append(load_dataset(dataset_name, v, split=split))
        return concatenate_datasets(datasets)

    if isinstance(split, tuple):
        datasets = []
        for s in split:
            datasets.append(load_dataset(dataset_name, version, split=s))
        return concatenate_datasets(datasets)

    return load_dataset(dataset_name, version, split=split)

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

def process_image(args):
    """Process a single image and compute its hash."""
    data, image_key, repo_id = args
    try:
        img = data[image_key]
        key = image_key
        # if isinstance(img, str):
        #     if repo_id is not None:
        #         sub_folder, image_id = img.split('/')
        #         path = hf_hub_download(
        #             repo_id=repo_id,
        #             filename=image_id,
        #             subfolder=sub_folder,
        #             repo_type="dataset"
        #         )
        #         image = Image.open(path).convert('RGB')
        #     else:
        #         image = Image.open(img).convert('RGB')
        # elif isinstance(img, Image.Image):
        #     image = img
        # else:
        #     print(f"Unsupported image type: {type(img)}")
        #     return None
        if img is None:
            if name != "sqa":
                raise ValueError("Image is None")
            img = dataset[1][key]
        if isinstance(img, list):
            if name != "seed":
                raise ValueError("Image is a list")
            img = img[0]
        if isinstance(img, str):
            if name != "vstar":
                # print("Image is a string")
                raise ValueError("Image is a string")
            sub_folder, image_id = img.split('/')
            path = hf_hub_download(repo_id="craigwu/vstar_bench", filename=image_id, subfolder=sub_folder, repo_type="dataset")
            img = Image.open(path)

        image = img.convert('RGB')

        # Preprocess the image
        image = preprocess_image(image)

        # Compute the hash
        hash_int = compute_hash(image, hash_size=HASH_SIZE)
        return hash_int
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def process_dataset(dataset_name):
    """Process a test dataset and save its hashes."""
    print(f"\nProcessing dataset: {dataset_name}")
    hash_file = os.path.join(HASH_FILES_DIR, f"{dataset_name}_hashes.pkl")
    if os.path.exists(hash_file):
        print(f"Hashes for dataset '{dataset_name}' already exist. Skipping.")
        return

    dataset = get_dataset(dataset_name)
    image_key = test_dataset_map[dataset_name][-1]
    repo_id = None
    if dataset_name == 'vstar':
        repo_id = 'craigwu/vstar_bench'

    # Prepare arguments for multiprocessing
    args_list = [(data, image_key, repo_id) for data in dataset]

    # Use multiprocessing Pool to process images in parallel
    with Pool(processes=NUM_WORKERS) as pool:
        hashes = list(tqdm(pool.imap_unordered(process_image, args_list), total=len(args_list)))

    # Filter out None values (failed processing)
    hashes = [h for h in hashes if h is not None]

    # Save hashes to a pickle file
    with open(hash_file, 'wb') as f:
        pickle.dump(hashes, f)

    print(f"Saved hashes for dataset '{dataset_name}' to {hash_file}")
    print(f"Total images processed: {len(dataset)}")
    print(f"Total hashes saved: {len(hashes)}")

if __name__ == '__main__':
    # List of test datasets to process
    test_datasets = list(test_dataset_map.keys())[::-1]
    test_datasets = list(test_dataset_map.keys())

    for dataset_name in test_datasets:
        # Check if hashes already exist
        hash_file = os.path.join(HASH_FILES_DIR, f"{dataset_name}_hashes.pkl")
        if os.path.exists(hash_file):
            print(f"Hashes for dataset '{dataset_name}' already exist. Skipping.")
            continue

        try:
            process_dataset(dataset_name)
        except Exception as e:
            print(f"Error processing dataset '{dataset_name}': {e}")
