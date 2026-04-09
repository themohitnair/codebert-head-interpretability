import os
import matplotlib.pyplot as plt
import numpy as np
from codebert_head_interpretability.utils.maths import compute_entropy


def _show_or_save_plot(save_path: str | None) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_category_heatmap(
    global_stats,
    count_per_head,
    category="identifier",
    save_path=None,
):
    layers = 12
    heads = 12
    heatmap = np.zeros((layers, heads))

    for (layer, head), cat_dict in global_stats.items():
        total = count_per_head[(layer, head)]
        val = cat_dict.get(category, 0) / total
        heatmap[layer, head] = val

    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap, aspect="auto", cmap="Blues")
    plt.colorbar(label=f"{category} attention")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(f"Attention Heatmap for '{category}'")
    plt.xticks(range(heads))
    plt.yticks(range(layers))
    plt.tight_layout()

    _show_or_save_plot(save_path)


def plot_top_category(global_stats, count_per_head, save_path=None):
    labels, values = [], []

    for (layer, head), cat_dict in sorted(global_stats.items()):
        total = count_per_head[(layer, head)]
        avg = {k: v / total for k, v in cat_dict.items()}
        top_cat = max(avg, key=avg.get)
        labels.append(f"L{layer}H{head}")
        values.append(avg[top_cat])

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("Top Category Score")
    plt.title("Top Attention Category per Head")
    plt.tight_layout()

    _show_or_save_plot(save_path)


def plot_entropy(global_stats, count_per_head, save_path=None):
    labels, entropies = [], []

    for (layer, head), cat_dict in sorted(global_stats.items()):
        total = count_per_head[(layer, head)]
        avg = {k: v / total for k, v in cat_dict.items()}
        ent = compute_entropy(avg)
        labels.append(f"L{layer}H{head}")
        entropies.append(ent)

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(entropies)), entropies)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("Entropy")
    plt.title("Entropy per Head (Lower = More Focused)")
    plt.tight_layout()

    _show_or_save_plot(save_path)
