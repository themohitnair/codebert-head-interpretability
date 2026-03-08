from datasets import load_dataset

ds = load_dataset(
    "sentence-transformers/codesearchnet",
    cache_dir="./hf_cache"
)

print(len(ds))