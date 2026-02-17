"""
Curate a COCO subset of images containing exactly two distinct,
frequently co-occurring categories ("Common Pairs").

The idea: CLIP should *easily* handle high-frequency pairs it has seen
millions of times during pretraining.  Any failure on these pairs is
therefore a clear structural GAP, not a data-frequency issue.

Output
------
datasets/coco_common_pairs.json
    {
        "metadata": { ... },
        "pairs": { "person+car": { ... }, ... },
        "images": [ { "image_id", "file_name", "pair", "categories" }, ... ]
    }
"""

import json
import os
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────
COCO_ROOT = Path("/home/molefe/datasets/coco2017")
ANNOTATIONS = COCO_ROOT / "annotations_trainval2017" / "annotations" / "instances_train2017.json"
IMAGE_DIR = COCO_ROOT / "train2017" / "train2017"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets"
OUTPUT_FILE = OUTPUT_DIR / "coco_common_pairs.json"

# ── tunables ─────────────────────────────────────────────────────────
TOP_K_PAIRS = 30          # keep the top-K most frequent co-occurring pairs
TARGET_PER_PAIR = 500     # aim for ~500 images per pair
MIN_PER_PAIR = 100        # drop pairs with fewer than this many candidates
SEED = 42

random.seed(SEED)


def load_annotations():
    print(f"Loading {ANNOTATIONS} …")
    with open(ANNOTATIONS) as f:
        data = json.load(f)
    return data


def build_category_map(data):
    """Map category id → name."""
    return {c["id"]: c["name"] for c in data["categories"]}


def build_image_categories(data):
    """Map image_id → set of category ids present in that image."""
    img_cats = defaultdict(set)
    for ann in data["annotations"]:
        img_cats[ann["image_id"]].add(ann["category_id"])
    return img_cats


def count_pair_cooccurrences(img_cats):
    """Count how many images each unordered category pair shares."""
    pair_counts = Counter()
    for cats in img_cats.values():
        for pair in combinations(sorted(cats), 2):
            pair_counts[pair] += 1
    return pair_counts


def select_top_pairs(pair_counts, cat_map, top_k, min_count):
    """Return top-K pairs that each have >= min_count images."""
    top = []
    for pair, count in pair_counts.most_common():
        if count < min_count:
            break
        top.append((pair, count))
        if len(top) == top_k:
            break

    print(f"\nTop {len(top)} co-occurring pairs (min {min_count} images):")
    print("-" * 60)
    for (c1, c2), count in top:
        print(f"  {cat_map[c1]:>20s} + {cat_map[c2]:<20s}  {count:>6,d} images")
    print("-" * 60)
    return top


def filter_exact_two(img_cats, target_pairs_set):
    """
    For each image, check if it has *exactly two* distinct categories
    AND that pair is in our target set.

    Returns {pair: [image_id, …]}
    """
    pair_images = defaultdict(list)
    for img_id, cats in img_cats.items():
        if len(cats) != 2:
            continue
        pair = tuple(sorted(cats))
        if pair in target_pairs_set:
            pair_images[pair].append(img_id)
    return pair_images


def balanced_sample(pair_images, target_per_pair):
    """Sample up to target_per_pair images from each pair."""
    sampled = {}
    for pair, img_ids in pair_images.items():
        if len(img_ids) <= target_per_pair:
            sampled[pair] = img_ids
        else:
            sampled[pair] = random.sample(img_ids, target_per_pair)
    return sampled


def build_image_map(data):
    """image_id → {"file_name": …, "width": …, "height": …}"""
    return {img["id"]: img for img in data["images"]}


def save_output(sampled, cat_map, image_map, pair_counts_dict):
    pairs_meta = {}
    images_list = []

    for pair, img_ids in sorted(sampled.items()):
        c1, c2 = pair
        pair_name = f"{cat_map[c1]}+{cat_map[c2]}"
        pairs_meta[pair_name] = {
            "category_ids": list(pair),
            "category_names": [cat_map[c1], cat_map[c2]],
            "total_cooccurrences": pair_counts_dict[pair],
            "exact_two_candidates": len(pair_images_global[pair]),
            "sampled": len(img_ids),
        }
        for img_id in img_ids:
            info = image_map[img_id]
            images_list.append({
                "image_id": img_id,
                "file_name": info["file_name"],
                "pair": pair_name,
                "categories": [cat_map[c1], cat_map[c2]],
                "width": info["width"],
                "height": info["height"],
            })

    output = {
        "metadata": {
            "description": (
                "COCO train2017 subset: images with exactly two distinct, "
                "frequently co-occurring categories. Designed for CLIP "
                "compositional-understanding diagnostics."
            ),
            "source": "instances_train2017.json",
            "image_dir": str(IMAGE_DIR),
            "num_pairs": len(pairs_meta),
            "num_images": len(images_list),
            "top_k_pairs": TOP_K_PAIRS,
            "target_per_pair": TARGET_PER_PAIR,
            "seed": SEED,
        },
        "pairs": pairs_meta,
        "images": images_list,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(images_list)} images across {len(pairs_meta)} pairs")
    print(f"  → {OUTPUT_FILE}")


# ── main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_annotations()
    cat_map = build_category_map(data)
    img_cats = build_image_categories(data)
    print(f"  {len(data['images']):,d} images, {len(data['annotations']):,d} annotations, "
          f"{len(cat_map)} categories")

    # Step 1: count all pairwise co-occurrences
    pair_counts = count_pair_cooccurrences(img_cats)
    print(f"\n{len(pair_counts):,d} distinct category pairs found across all images")

    # Step 2: pick the top-K most frequent pairs
    top_pairs = select_top_pairs(pair_counts, cat_map, TOP_K_PAIRS, MIN_PER_PAIR)
    target_set = {pair for pair, _ in top_pairs}
    pair_counts_dict = dict(pair_counts)

    # Step 3: filter to images with *exactly* two categories from target set
    pair_images_global = filter_exact_two(img_cats, target_set)
    total_candidates = sum(len(v) for v in pair_images_global.values())
    pairs_with_images = {p: ids for p, ids in pair_images_global.items() if ids}
    print(f"\n{total_candidates:,d} images have exactly 2 categories forming a common pair")
    print(f"  covering {len(pairs_with_images)}/{len(target_set)} of our target pairs")

    # Step 4: balanced sample
    sampled = balanced_sample(pairs_with_images, TARGET_PER_PAIR)
    total_sampled = sum(len(v) for v in sampled.values())
    print(f"\nAfter balanced sampling ({TARGET_PER_PAIR}/pair): {total_sampled:,d} images")

    # Step 5: save
    image_map = build_image_map(data)
    save_output(sampled, cat_map, image_map, pair_counts_dict)

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Pair':<35s} {'Candidates':>12s} {'Sampled':>10s}")
    print("=" * 70)
    for pair in sorted(sampled.keys()):
        c1, c2 = pair
        name = f"{cat_map[c1]} + {cat_map[c2]}"
        print(f"  {name:<33s} {len(pair_images_global[pair]):>10,d} {len(sampled[pair]):>10,d}")
    print("=" * 70)
    print(f"  {'TOTAL':<33s} {total_candidates:>10,d} {total_sampled:>10,d}")
