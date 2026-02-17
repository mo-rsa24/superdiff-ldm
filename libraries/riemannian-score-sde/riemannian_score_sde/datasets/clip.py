import json
from pathlib import Path

import numpy as np

from score_sde.datasets import TensorDataset


class CLIPEmbeddingDataset(TensorDataset):
    """Load pre-extracted CLIP image embeddings (L2-normalised, on S^{d-1}).

    Filtering modes (set via Hydra config):
      pair=null, category=null  →  all 5,722 images
      pair="person+car"         →  only person+car images
      category="person"         →  all images containing "person"
    """

    def __init__(self, embeddings_path, manifest_path=None, pair=None,
                 category=None, **kwargs):
        embeddings_path = Path(embeddings_path)
        data = np.load(embeddings_path).astype(np.float32)

        if manifest_path is not None and (pair is not None or category is not None):
            manifest_path = Path(manifest_path)
            with open(manifest_path) as f:
                records = [json.loads(line) for line in f if line.strip()]

            if pair is not None:
                # Exact pair match: "person+car"
                mask = np.array([r["pair"] == pair for r in records])
            else:
                # Single category: all images containing e.g. "person"
                mask = np.array([category in r["categories"] for r in records])

            data = data[mask]

        # Safety: re-normalise to unit sphere
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        data = data / np.clip(norms, 1e-8, None)

        super().__init__(data)
