# VinBigData.py - Multi-head Salient sepVAE Dataset
# Returns triplets of images (Normal, Cardiomegaly, Pleural Effusion) for head-nulling loss training.
# Each image is a 512x512 grayscale DICOM, normalized to [-1, 1] for CheSS backbone compatibility.

import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
from PIL import Image


class VinBigDataTripletDataset(Dataset):
    """
    VinBigDataTripletDataset for Multi-head Salient sepVAE training.

    Returns triplets of images from three distinct pathology pools:
    - Normal (class_id: 14)
    - Cardiomegaly (class_id: 3)
    - Pleural Effusion (class_id: 10)

    Directory structure:
        /datasets/mmolefe/vinbigdata/train/  -> *.dicom files
        /datasets/mmolefe/vinbigdata/train.csv -> image_id to class_id mapping

    Each __getitem__ returns:
        {
            'x_norm': torch.Tensor (1, 512, 512),
            'x_disease1': torch.Tensor (1, 512, 512),  # Pleural Effusion
            'x_disease2': torch.Tensor (1, 512, 512),  # Cardiomegaly
            'disease_labels': torch.Tensor (3,)  # [0, 1, 2] for head-nulling gating
        }
    """

    def __init__(
        self,
        dicom_dir: str = "/datasets/mmolefe/vinbigdata/train",
        csv_path: str = "/datasets/mmolefe/vinbigdata/train.csv",
        img_size: int = 512,
        window_center: float = 40.0,
        window_width: float = 400.0,
    ):
        """
        Args:
            dicom_dir: Path to folder containing .dicom files
            csv_path: Path to train.csv with image_id and class_id columns
            img_size: Target image size (default: 512 for CheSS backbone)
            window_center: Default window center for VOI LUT
            window_width: Default window width for VOI LUT
        """
        self.dicom_dir = Path(dicom_dir)
        self.img_size = int(img_size)
        self.window_center = window_center
        self.window_width = window_width

        if not self.dicom_dir.exists():
            raise FileNotFoundError(f"DICOM directory not found: {self.dicom_dir}")
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load CSV and pre-calculate class pools
        df = pd.read_csv(csv_path)

        # Group by class_id
        self.normal_ids = self._filter_class(df, class_id=14)
        self.cardio_ids = self._filter_class(df, class_id=3)
        self.effusion_ids = self._filter_class(df, class_id=10)

        # Validation
        if len(self.normal_ids) == 0:
            raise RuntimeError("No Normal (class_id=14) images found in CSV")
        if len(self.cardio_ids) == 0:
            raise RuntimeError("No Cardiomegaly (class_id=3) images found in CSV")
        if len(self.effusion_ids) == 0:
            raise RuntimeError("No Pleural Effusion (class_id=10) images found in CSV")

        print(f"[VinBigData Dataset Initialized]")
        print(f"  Normal: {len(self.normal_ids)} images")
        print(f"  Cardiomegaly: {len(self.cardio_ids)} images")
        print(f"  Pleural Effusion: {len(self.effusion_ids)} images")
        print(f"  Total triplets: {len(self)}")

    def _filter_class(self, df: pd.DataFrame, class_id: int) -> List[str]:
        """Extract unique image_ids for a given class_id."""
        filtered = df[df['class_id'] == class_id]['image_id'].unique().tolist()
        return filtered

    def __len__(self) -> int:
        """
        Dataset length is determined by the smallest class pool to ensure
        balanced sampling across all three pathologies.
        """
        return min(len(self.normal_ids), len(self.cardio_ids), len(self.effusion_ids))

    def _load_dicom(self, image_id: str) -> np.ndarray:
        """
        Load a DICOM file and apply VOI LUT windowing.

        Args:
            image_id: Image identifier (without extension)

        Returns:
            np.ndarray: Windowed pixel array (H, W) in [0, 1] range
        """
        dicom_path = self.dicom_dir / f"{image_id}.dicom"

        if not dicom_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

        # Load DICOM
        dcm = pydicom.dcmread(str(dicom_path))
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Apply VOI LUT (Windowing)
        # Use metadata if available, otherwise use defaults
        window_center = float(getattr(dcm, 'WindowCenter', self.window_center))
        window_width = float(getattr(dcm, 'WindowWidth', self.window_width))

        # Handle list values (some DICOMs store multiple windows)
        if isinstance(window_center, (list, pydicom.multival.MultiValue)):
            window_center = float(window_center[0])
        if isinstance(window_width, (list, pydicom.multival.MultiValue)):
            window_width = float(window_width[0])

        # Apply windowing: pixels outside [center - width/2, center + width/2] are clipped
        lower = window_center - window_width / 2.0
        upper = window_center + window_width / 2.0
        windowed = np.clip(pixel_array, lower, upper)

        # Normalize to [0, 1]
        windowed = (windowed - lower) / (upper - lower + 1e-8)

        return windowed

    def _preprocess_image(self, pixel_array: np.ndarray) -> torch.Tensor:
        """
        Resize and normalize image to [-1, 1] range for sepVAE.

        Args:
            pixel_array: (H, W) array in [0, 1] range

        Returns:
            torch.Tensor: (1, img_size, img_size) in [-1, 1] range
        """
        # Convert to PIL for resizing
        img = Image.fromarray((pixel_array * 255).astype(np.uint8), mode='L')
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)

        # Convert back to array and normalize to [-1, 1]
        x = np.asarray(img, dtype=np.float32) / 255.0  # [0, 1]
        x = x * 2.0 - 1.0  # [-1, 1]
        x = x[None, ...]  # (1, H, W)

        return torch.from_numpy(x)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a triplet of images for head-nulling loss training.

        Args:
            idx: Index (determines which Normal image to use as anchor)

        Returns:
            dict: {
                'x_norm': Normal image,
                'x_disease1': Pleural Effusion image,
                'x_disease2': Cardiomegaly image,
                'disease_labels': [0, 1, 2] tensor for head gating
            }
        """
        # Use idx for Normal (anchor), random sample for diseases
        norm_id = self.normal_ids[idx % len(self.normal_ids)]
        effusion_id = self.effusion_ids[np.random.randint(0, len(self.effusion_ids))]
        cardio_id = self.cardio_ids[np.random.randint(0, len(self.cardio_ids))]

        # Load and preprocess
        x_norm = self._preprocess_image(self._load_dicom(norm_id))
        x_effusion = self._preprocess_image(self._load_dicom(effusion_id))
        x_cardio = self._preprocess_image(self._load_dicom(cardio_id))

        # Disease labels for head-nulling loss gating
        # 0: Normal, 1: Disease1 (Effusion), 2: Disease2 (Cardiomegaly)
        disease_labels = torch.tensor([0, 1, 2], dtype=torch.long)

        return {
            'x_norm': x_norm,
            'x_disease1': x_effusion,
            'x_disease2': x_cardio,
            'disease_labels': disease_labels
        }


def jax_collate_fn(batch_list):
    """
    JAX-compatible collate function for VinBigData triplets.

    This function stacks triplets and creates proper disease labels for training.
    Input batch contains B triplets, output contains 3*B images with labels.

    Args:
        batch_list: List of dicts from __getitem__, each with:
                   - 'x_norm': Normal image (1, 512, 512)
                   - 'x_disease1': Pleural Effusion image (1, 512, 512)
                   - 'x_disease2': Cardiomegaly image (1, 512, 512)
                   - 'disease_labels': [0, 1, 2] (unused)

    Returns:
        Dict with:
            - 'x_norm': (B, 1, 512, 512) Normal images
            - 'x_disease1': (B, 1, 512, 512) Pleural Effusion images
            - 'x_disease2': (B, 1, 512, 512) Cardiomegaly images
            - 'disease_labels': (3*B,) labels [0,0,...,1,1,...,2,2,...]
                               Where: 0=Normal, 1=Pleural Effusion, 2=Cardiomegaly

    Example:
        With batch_size=4:
        - Input: 4 triplets
        - Output: 12 images with labels [0,0,0,0, 1,1,1,1, 2,2,2,2]
    """
    B = len(batch_list)

    # Stack images
    x_norm = torch.stack([b['x_norm'] for b in batch_list])  # (B, 1, 512, 512)
    x_disease1 = torch.stack([b['x_disease1'] for b in batch_list])  # Effusion
    x_disease2 = torch.stack([b['x_disease2'] for b in batch_list])  # Cardiomegaly

    # Create labels: [0, 0, ..., 1, 1, ..., 2, 2, ...]
    # 0: Normal, 1: Pleural Effusion, 2: Cardiomegaly
    labels = torch.cat([
        torch.zeros(B, dtype=torch.long),  # Normal
        torch.ones(B, dtype=torch.long),   # Pleural Effusion
        torch.full((B,), 2, dtype=torch.long)  # Cardiomegaly
    ])

    return {
        'x_norm': x_norm,
        'x_disease1': x_disease1,
        'x_disease2': x_disease2,
        'disease_labels': labels  # (3*B,)
    }


# Example usage and validation
if __name__ == "__main__":
    # Test dataset initialization
    dataset = VinBigDataTripletDataset()
    print(f"\nDataset length: {len(dataset)}")

    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"  x_norm shape: {sample['x_norm'].shape}, range: [{sample['x_norm'].min():.3f}, {sample['x_norm'].max():.3f}]")
    print(f"  x_disease1 shape: {sample['x_disease1'].shape}, range: [{sample['x_disease1'].min():.3f}, {sample['x_disease1'].max():.3f}]")
    print(f"  x_disease2 shape: {sample['x_disease2'].shape}, range: [{sample['x_disease2'].min():.3f}, {sample['x_disease2'].max():.3f}]")
    print(f"  disease_labels: {sample['disease_labels']}")

    # Test with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"\nBatch shapes (default collation):")
    print(f"  x_norm: {batch['x_norm'].shape}")
    print(f"  x_disease1: {batch['x_disease1'].shape}")
    print(f"  x_disease2: {batch['x_disease2'].shape}")
    print(f"  disease_labels: {batch['disease_labels'].shape}")

    # Test with JAX collation
    loader_jax = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=jax_collate_fn)
    batch_jax = next(iter(loader_jax))
    print(f"\nBatch shapes (JAX collation):")
    print(f"  x_norm: {batch_jax['x_norm'].shape}")
    print(f"  x_disease1: {batch_jax['x_disease1'].shape}")
    print(f"  x_disease2: {batch_jax['x_disease2'].shape}")
    print(f"  disease_labels: {batch_jax['disease_labels'].shape}")
    print(f"  disease_labels values: {batch_jax['disease_labels']}")
