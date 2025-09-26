import random
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset
from PIL import Image

# If you use a HF processor elsewhere, import there (don’t hard-wire globals here)

def augment_suffix(suffix: str) -> str:
    parts = suffix.split(' ; ')
    random.shuffle(parts)
    return ' ; '.join(parts)

class VLMDataset(Dataset):
    """
    Example multimodal dataset: returns (image, label_dict)
    label_dict must include keys used in collate_fn: 'image' (path), 'prefix', 'suffix'
    """
    def __init__(self, samples: List[Dict[str, Any]], image_root=None, transforms=None):
        self.samples = samples
        self.image_root = image_root
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        rec = self.samples[idx]
        img_path = rec["image"] if self.image_root is None else f"{self.image_root}/{rec['image']}"
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, rec

def make_collate_fn(processor, torch_dtype, device):
    """
    Binds processor/dtype/device and returns a collate_fn compatible with HF Trainer.
    """
    def collate_fn(batch):
        images, labels = zip(*batch)
        prefixes = ["<image>" + lb["prefix"] for lb in labels]
        suffixes = [augment_suffix(lb["suffix"]) for lb in labels]
        inputs = processor(
            text=prefixes,
            images=images,
            suffix=suffixes,
            return_tensors="pt",
            padding="longest"
        ).to(torch_dtype).to(device)
        return inputs
    return collate_fn
