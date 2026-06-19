"""
dataset.py — CityscapesDataset + augmentations
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────────────────────
# Label mapping
# ─────────────────────────────────────────────────────────────────────────────
CITYSCAPES_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,   8: 1,   9: 255, 10: 255,
    11: 2,  12: 3,  13: 4,
    14: 255, 15: 255, 16: 255,
    17: 5,  18: 255,
    19: 6,  20: 7,
    21: 8,  22: 9,  23: 10,
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
    29: 255, 30: 255,
    31: 16, 32: 17, 33: 18,
}

CITYSCAPES_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70],
    [102, 102, 156], [190, 153, 153], [153, 153, 153],
    [250, 170,  30], [220, 220,   0], [107, 142,  35],
    [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70],
    [  0,  60, 100], [  0,  80, 100], [  0,   0, 230],
    [119,  11,  32],
], dtype=np.uint8)

_lut = np.full(256, 255, dtype=np.uint8)
for _raw, _tid in CITYSCAPES_ID_TO_TRAINID.items():
    _lut[_raw] = _tid
CITYSCAPES_LUT = _lut

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

CITYSCAPES_CLASS_WEIGHTS = [
    0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
    1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
    1.0865, 1.0955, 1.0865, 1.1529, 1.0507,
]
CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_IGNORE_INDEX = 255
ADE20K_NUM_CLASSES  = 150
ADE20K_IGNORE_INDEX = 255

ADE20K_CLASS_NAMES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road",
    "bed", "windowpane", "grass", "cabinet", "sidewalk", "person",
    "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea",
    "mirror", "rug", "field", "armchair", "seat", "fence", "desk",
    "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers",
    "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator",
    "grandstand", "path", "stairs", "runway", "case", "pool table",
    "pillow", "screen door", "stairway", "river", "bridge", "bookcase",
    "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island",
    "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier",
    "awning", "streetlight", "booth", "television", "airplane",
    "dirt track", "apparel", "pole", "land", "bannister",
    "escalator", "ottoman", "bottle", "buffet", "poster", "stage",
    "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket",
    "waterfall", "tent", "bag", "minibike", "cradle", "oven",
    "ball", "food", "step", "tank", "trade name", "microwave",
    "pot", "animal", "bicycle", "lake", "dishwasher", "screen",
    "blanket", "sculpture", "hood", "sconce", "vase", "traffic light",
    "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor",
    "bulletin board", "shower", "radiator", "glass", "clock", "flag",
]
_rng = np.random.RandomState(42)
ADE20K_PALETTE = _rng.randint(0, 256, size=(ADE20K_NUM_CLASSES, 3), dtype=np.uint8)
# LUT 256 phần tử: index 0 → ignore_index, index 1..150 → 0..149, còn lại → ignore_index
_lut = np.full(256, ADE20K_IGNORE_INDEX, dtype=np.uint8)
for raw_id in range(1, ADE20K_NUM_CLASSES + 1):
    _lut[raw_id] = raw_id - 1
ADE20K_LUT = _lut

# ─────────────────────────────────────────────────────────────────────────────
# Cityscapes Dataset
# ─────────────────────────────────────────────────────────────────────────────
class CityscapesDataset(Dataset):
    def __init__(
        self,
        root_dir:     str,
        label_dir:    str | None = None,
        transform=None,
        train_height: int =  512,
        train_width: int = 1024,
    ):
        super().__init__()
        self.root_dir     = root_dir
        self.label_dir    = label_dir
        self.transform    = transform
        self.train_height = train_height
        self.train_width  = train_width

        self.list_img = []
        for folder in sorted(f for f in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, f))):
            for fname in sorted(os.listdir(os.path.join(root_dir, folder))):
                self.list_img.append(os.path.join(folder, fname))

        self.list_label_img = []
        if label_dir is not None:
            for folder in sorted(f for f in os.listdir(label_dir)
                                  if os.path.isdir(os.path.join(label_dir, f))):
                for fname in sorted(os.listdir(os.path.join(label_dir, folder))):
                    if fname.endswith("labelIds.png"):
                        self.list_label_img.append(os.path.join(folder, fname))

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.root_dir, self.list_img[idx])
        image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.train_width, self.train_height))

        has_label = self.label_dir is not None and len(self.list_label_img) > 0

        if has_label:
            mask_path = os.path.join(self.label_dir, self.list_label_img[idx])
            mask_raw  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_raw  = cv2.resize(
                mask_raw, (self.train_width, self.train_height),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = CITYSCAPES_LUT[mask_raw]

        if self.transform is not None:
            if has_label:
                out   = self.transform(image=image, mask=mask)
                image = out["image"]
                mask  = out["mask"].long()
            else:
                image = self.transform(image=image)["image"]
                return image
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            if has_label:
                mask = torch.from_numpy(mask).long()

        return image, mask

    @staticmethod
    def decode_segmap(mask: "np.ndarray | torch.Tensor") -> np.ndarray:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = np.asarray(mask, dtype=np.int32)
        h, w = mask.shape
        rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        for train_id, colour in enumerate(CITYSCAPES_PALETTE):
            rgb[mask == train_id] = colour
        return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────────────────────────────────────

def build_cityscapes_transforms(split: str, train_height: int = 512, train_width: int = 1024) -> A.Compose:
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=(-0.5, 0.5), p=0.5),
            A.PadIfNeeded(
                min_height   = train_height,
                min_width    = train_width,
                border_mode  = cv2.BORDER_CONSTANT,
                fill         = 0,
                fill_mask    = CITYSCAPES_IGNORE_INDEX,
            ),
            A.RandomCrop(height=train_height, width=train_width),
            A.ColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.4, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# UnNormalize (dùng cho visualize)
# ─────────────────────────────────────────────────────────────────────────────
class UnNormalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std  = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

# ─────────────────────────────────────────────────────────────────────────────
# ADE20K Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ADE20KDataset(Dataset):
    """
    Dataset cho ADE20K (ADEChallengeData2016).

    Parameters
    ----------
    img_dir   : path tới images/training hoặc images/validation
    label_dir : path tới annotations/training hoặc annotations/validation
                (None nếu chỉ predict, không có ground truth)
    transform : albumentations Compose
    """

    def __init__(
        self,
        img_dir:      str,
        label_dir:    str | None = None,
        transform=None,
        train_height: int = 512,
        train_width:  int = 512,
    ):
        super().__init__()
        self.img_dir      = img_dir
        self.label_dir    = label_dir
        self.transform    = transform
        self.train_height = train_height
        self.train_width  = train_width

        # ADE20K KHÔNG có sub-folder theo city như Cityscapes — ảnh nằm phẳng
        self.list_img = sorted(
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        self.has_label = label_dir is not None
        if self.has_label:
            # Annotation cùng tên file, đổi đuôi .jpg → .png
            self.list_label = [
                os.path.splitext(f)[0] + ".png" for f in self.list_img
            ]
            missing = [f for f in self.list_label
                      if not os.path.exists(os.path.join(label_dir, f))]
            if missing:
                print(f"[ADE20KDataset] Cảnh báo: {len(missing)} annotation bị thiếu, "
                     f"ví dụ: {missing[:3]}")

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.list_img[idx])
        image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.train_width, self.train_height))

        if self.has_label:
            lbl_path = os.path.join(self.label_dir, self.list_label[idx])
            mask_raw = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
            mask_raw = cv2.resize(
                mask_raw, (self.train_width, self.train_height),
                interpolation=cv2.INTER_NEAREST,
            )
            # Quan trọng: shift 1-based → 0-based, 0 (unlabeled) → ignore_index
            mask = ADE20K_LUT[mask_raw]

        if self.transform is not None:
            if self.has_label:
                out   = self.transform(image=image, mask=mask)
                image = out["image"]
                mask  = out["mask"].long()
            else:
                image = self.transform(image=image)["image"]
                return image
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            if self.has_label:
                mask = torch.from_numpy(mask).long()

        return image, mask

    @staticmethod
    def decode_segmap(mask: "np.ndarray | torch.Tensor") -> np.ndarray:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = np.asarray(mask, dtype=np.int32)
        h, w = mask.shape
        rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        for train_id, colour in enumerate(ADE20K_PALETTE):
            rgb[mask == train_id] = colour
        return rgb


def build_ade20k_transforms(split: str, train_height: int = 512, train_width: int = 512) -> A.Compose:
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=(-0.5, 0.5), p=0.5),
            A.PadIfNeeded(
                min_height   = train_height,
                min_width    = train_width,
                border_mode  = cv2.BORDER_CONSTANT,
                fill         = 0,
                fill_mask    = ADE20K_IGNORE_INDEX,
            ),
            A.RandomCrop(height=train_height, width=train_width),
            A.ColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.4, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])