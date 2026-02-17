"""
Verify extracted CLIP embeddings: shapes, norms, cosine similarities.

Usage
-----
    python scripts/verify_clip_embeddings.py
    python scripts/verify_clip_embeddings.py --embed_dir clip_embeddings/coco_common_pairs
"""

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBED_DIR = PROJECT_ROOT / "clip_embeddings" / "coco_common_pairs"


def parse_args():
    p = argparse.ArgumentParser(description="Verify CLIP embeddings.")
    p.add_argument("--embed_dir", type=Path, default=DEFAULT_EMBED_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    d = args.embed_dir
    text_dir = d / "text_embeddings"

    # ── Load ──
    image_embeds = np.load(d / "image_embeddings.npy")
    comp = np.load(text_dir / "compositional.npy")
    ind_a = np.load(text_dir / "individual_a.npy")
    ind_b = np.load(text_dir / "individual_b.npy")

    with open(text_dir / "prompts.json") as f:
        prompts = json.load(f)
    with open(d / "meta.json") as f:
        meta = json.load(f)

    manifest = []
    with open(d / "image_manifest.jsonl") as f:
        for line in f:
            if line.strip():
                manifest.append(json.loads(line))

    pair_keys = list(prompts.keys())

    # ── 1. Shape & dtype report ──
    print("=" * 70)
    print("  SHAPE & DTYPE REPORT")
    print("=" * 70)
    print(f"  Image embeddings : {image_embeds.shape}  dtype={image_embeds.dtype}")
    print(f"  Compositional    : {comp.shape}  dtype={comp.dtype}")
    print(f"  Individual A     : {ind_a.shape}  dtype={ind_a.dtype}")
    print(f"  Individual B     : {ind_b.shape}  dtype={ind_b.dtype}")
    print(f"  Manifest records : {len(manifest)}")
    print(f"  Model            : {meta['model_id']}")
    print(f"  Embed dim        : {meta['embed_dim']}")

    ok = True
    if image_embeds.shape[0] != len(manifest):
        print("  [FAIL] Image embed rows != manifest records")
        ok = False
    if image_embeds.shape[1] != meta["embed_dim"]:
        print("  [FAIL] Image embed dim mismatch")
        ok = False
    if comp.shape[0] != len(pair_keys):
        print("  [FAIL] Compositional rows != num pairs")
        ok = False

    # ── 2. Norm check (should be ~1.0 after L2 normalisation) ──
    print(f"\n{'=' * 70}")
    print("  L2 NORM CHECK")
    print("=" * 70)
    img_norms = np.linalg.norm(image_embeds, axis=1)
    comp_norms = np.linalg.norm(comp, axis=1)
    ind_a_norms = np.linalg.norm(ind_a, axis=1)
    ind_b_norms = np.linalg.norm(ind_b, axis=1)

    for name, norms in [("Image", img_norms), ("Compositional", comp_norms),
                        ("Individual A", ind_a_norms), ("Individual B", ind_b_norms)]:
        print(f"  {name:16s}: min={norms.min():.6f}  max={norms.max():.6f}  "
              f"mean={norms.mean():.6f}")
        if abs(norms.mean() - 1.0) > 0.01:
            print(f"    [WARN] Mean norm far from 1.0")

    # ── 3. NaN / Inf check ──
    print(f"\n{'=' * 70}")
    print("  NaN / Inf CHECK")
    print("=" * 70)
    for name, arr in [("Image", image_embeds), ("Compositional", comp),
                      ("Individual A", ind_a), ("Individual B", ind_b)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        status = "OK" if (n_nan == 0 and n_inf == 0) else "FAIL"
        print(f"  {name:16s}: NaN={n_nan}  Inf={n_inf}  [{status}]")
        if n_nan > 0 or n_inf > 0:
            ok = False

    # ── 4. Per-pair cosine similarity analysis ──
    print(f"\n{'=' * 70}")
    print("  PER-PAIR COSINE SIMILARITY ANALYSIS")
    print("=" * 70)
    print(f"\n  cos(comp, mean_img) = cosine between 'a {'{cat_a}'} and a {'{cat_b}'}' text")
    print(f"                        embedding and the mean image embedding for that pair")
    print(f"  cos(a, mean_img)    = cosine between 'a {'{cat_a}'}' and mean image embedding")
    print(f"  cos(b, mean_img)    = cosine between 'a {'{cat_b}'}' and mean image embedding")
    print(f"  cos(a, b)           = cosine between individual concept text embeddings")
    print()

    header = (f"  {'Pair':<28s} {'N':>5s}  {'cos(comp,img)':>13s}  "
              f"{'cos(a,img)':>10s}  {'cos(b,img)':>10s}  {'cos(a,b)':>8s}")
    print(header)
    print(f"  {'-' * 85}")

    cos_comp_all, cos_a_all, cos_b_all, cos_ab_all = [], [], [], []

    for pi, pk in enumerate(pair_keys):
        # Gather image indices for this pair
        pair_mask = np.array([m["pair"] == pk for m in manifest])
        pair_imgs = image_embeds[pair_mask]
        n_imgs = pair_imgs.shape[0]

        if n_imgs == 0:
            print(f"  {pk:<28s} {'0':>5s}  (no images)")
            continue

        # Mean image embedding (re-normalise)
        mean_img = pair_imgs.mean(axis=0)
        mean_img = mean_img / np.linalg.norm(mean_img)

        cos_comp = float(comp[pi] @ mean_img)
        cos_a = float(ind_a[pi] @ mean_img)
        cos_b = float(ind_b[pi] @ mean_img)
        cos_ab = float(ind_a[pi] @ ind_b[pi])

        cos_comp_all.append(cos_comp)
        cos_a_all.append(cos_a)
        cos_b_all.append(cos_b)
        cos_ab_all.append(cos_ab)

        print(f"  {pk:<28s} {n_imgs:>5d}  {cos_comp:>+13.4f}  "
              f"{cos_a:>+10.4f}  {cos_b:>+10.4f}  {cos_ab:>+8.4f}")

    print(f"  {'-' * 85}")
    print(f"  {'MEAN':<28s} {'':>5s}  {np.mean(cos_comp_all):>+13.4f}  "
          f"{np.mean(cos_a_all):>+10.4f}  {np.mean(cos_b_all):>+10.4f}  "
          f"{np.mean(cos_ab_all):>+8.4f}")

    # ── 5. Semantic dilution indicator ──
    print(f"\n{'=' * 70}")
    print("  SEMANTIC DILUTION INDICATOR")
    print("=" * 70)
    print(f"\n  For each pair, compare cos(compositional, mean_img) vs")
    print(f"  max(cos(individual_a, mean_img), cos(individual_b, mean_img)).")
    print(f"  If the compositional prompt is CLOSER to the real images than")
    print(f"  either individual, composition is working. If it is FARTHER,")
    print(f"  the 'AND' is diluting the signal.\n")

    n_diluted = 0
    for pi, pk in enumerate(pair_keys):
        pair_mask = np.array([m["pair"] == pk for m in manifest])
        if pair_mask.sum() == 0:
            continue
        pair_imgs = image_embeds[pair_mask]
        mean_img = pair_imgs.mean(axis=0)
        mean_img = mean_img / np.linalg.norm(mean_img)

        cos_comp = float(comp[pi] @ mean_img)
        cos_best_ind = max(float(ind_a[pi] @ mean_img), float(ind_b[pi] @ mean_img))
        delta = cos_comp - cos_best_ind
        label = "OK" if delta >= 0 else "DILUTED"
        if delta < 0:
            n_diluted += 1
        print(f"  {pk:<28s}  comp={cos_comp:+.4f}  best_ind={cos_best_ind:+.4f}  "
              f"delta={delta:+.4f}  [{label}]")

    print(f"\n  Diluted pairs: {n_diluted}/{len(pair_keys)}")

    # ── 6. Intra-pair variance ──
    print(f"\n{'=' * 70}")
    print("  INTRA-PAIR IMAGE EMBEDDING VARIANCE")
    print("=" * 70)
    print(f"\n  Mean pairwise cosine similarity among images of the same pair.")
    print(f"  High = tight cluster, Low = spread out.\n")

    for pi, pk in enumerate(pair_keys):
        pair_mask = np.array([m["pair"] == pk for m in manifest])
        pair_imgs = image_embeds[pair_mask]
        n_imgs = pair_imgs.shape[0]
        if n_imgs < 2:
            continue
        # Sample up to 200 pairs for efficiency
        if n_imgs > 200:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_imgs, 200, replace=False)
            sample = pair_imgs[idx]
        else:
            sample = pair_imgs
        sim_matrix = sample @ sample.T
        # Extract upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(sim_matrix.shape[0], k=1)
        pairwise = sim_matrix[triu_idx]
        print(f"  {pk:<28s}  N={n_imgs:>4d}  mean_cos={pairwise.mean():.4f}  "
              f"std={pairwise.std():.4f}")

    # ── Final ──
    print(f"\n{'=' * 70}")
    print(f"  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
