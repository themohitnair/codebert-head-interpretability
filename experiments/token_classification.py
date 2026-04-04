from codebert_head_interpretability.dataset import get_dataset


def main():
    dataset = get_dataset("codesearchnet", language="python")
    ds = dataset.load(split="train")

    for example in dataset.to_examples(ds, max_examples=1):
        print("Code:", example.code)


if __name__ == "__main__":
    main()
