from datasets import load_dataset

ds = load_dataset("sentence-transformers/codesearchnet")

print(len(ds))