import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.transforms import functional as F

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image

MIN_BOX_SIZE = 1

# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    # BEFORE: images, targets, _ = zip(*batch)
    # FIX: batch contains 2 items, we tried to unpack 3
    images, targets = zip(*batch)
    return list(images), list(targets)


# -----------------------------
# Dataset
# -----------------------------
@dataclass
class Sample:
    image_path: Path
    ann_path: Path
    order_name: str


class ArTaxOrVoTTDataset(Dataset):
    """
    Reads VoTT JSON per image. Expected keys:
      - data["asset"]["name"] -> image filename
      - data["regions"][i]["boundingBox"] -> {left, top, width, height}
      - data["regions"][i]["tags"] -> ["OrderName", ...] (optional; we fallback to folder name)
    """

    def __init__(
        self,
        samples: List[Sample],
        class_to_idx: Dict[str, int],
        train: bool = True,
        hflip_p: float = 0.5,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.train = train
        self.hflip_p = hflip_p

    def __len__(self) -> int:
        return len(self.samples)

    def _read_target(self, ann_path: Path, fallback_label: str) -> Tuple[Tensor, Tensor]:
        data = json.loads(ann_path.read_text(encoding="utf-8"))

        boxes: List[List[float]] = []
        labels: List[int] = []

        for reg in data.get("regions", []):
            bb = reg.get("boundingBox", None)
            if bb is None:
                continue

            left = float(bb["left"])
            top = float(bb["top"])
            width = float(bb["width"])
            height = float(bb["height"])
            x1, y1 = left, top
            # BEFORE: x2, y2 = left + height, top + width
            # FIX: we previously added width to height and vice versa, needed to switch places
            x2, y2 = left + width, top + height

            tag_list = reg.get("tags", []) or []
            # BEFORE: label_name = tag_list[0] if (len(tag_list) > 0) else fallback_label; 
            # if label_name not in self.class_to_idx: label_name = fallback_label
            # FIX: this is not a bug for the current dataset, however it is a guard against errors that may arise:
            # correct tag=label is taken from json tag list 1st item, which may not be the correct class (as the folder name or from class
            # list at all), so we look through all tags for the instance, find first that exists and valid, and if not fallback to folder name
            class_tags = [t for t in tag_list if t in self.class_to_idx]
            label_name = class_tags[0] if class_tags else fallback_label

            # BEFORE: boxes.append([x1, y2, x2, y1])
            # FIX: coordinate pairs were mixed up, bbox is xyxy per same pair
            box = [x1, y1, x2, y2]
            boxes.append(box)
            labels.append(self.class_to_idx[label_name])

        # BEFORE: if len(boxes) == 0: raise ValueError(f"No boxes found in {ann_path}")
        # FIX: image without instances is valid background, we don't need to raise an error about it, if image is bg return (0, 4) bbox for batch consistency
        return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(labels, dtype=torch.int64))


    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        s = self.samples[idx]

        data = json.loads(s.ann_path.read_text(encoding="utf-8"))
        img_name = data.get("asset", {}).get("name", s.image_path.name)
        img_path = s.image_path.parent / img_name

        img = Image.open(img_path).convert("RGB")
        # BEFORE: h, w = img.size
        # FIX: PIL returns w,h, it is cv2 that returns h, w
        w, h = img.size

        boxes, labels = self._read_target(s.ann_path, fallback_label=s.order_name)

        # Basic augmentation: horizontal flip
        if self.train and random.random() < self.hflip_p:
            # BEFORE: img = F.vflip(img); boxes[:, [1, 3]] = torch.tensor([h, h], dtype=torch.float32) - boxes[:, [3, 1]]
            # FIX: it was assumed that as the variable named hflip_p, but vflip is done - the hflip 
            # makes sense better, as it is more "natural" for insects to appear looking at the opposite 
            # direction than to appear upside-down
            img = F.hflip(img)
            boxes[:, [0, 2]] = torch.tensor([w, w], dtype=torch.float32) - boxes[:, [2, 0]]

        img_t = F.to_tensor(img)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

        # BEFORE: None
        # FIX: bboxes can have width/height ~ 0 in incorrect labels from the
        # start, as well as coords that both fall out of the image and therefore get
        # clipped to the same w/h value - bbox starts to have width/height 0 - 
        # such instances should be filtered out 
        valid_bbox_mask = (
            ((boxes[:, 2] - boxes[:, 0]) > MIN_BOX_SIZE)
            & ((boxes[:, 3] - boxes[:, 1]) > MIN_BOX_SIZE)
        )
        boxes, labels = boxes[valid_bbox_mask], labels[valid_bbox_mask]


        target: Dict[str, Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }
        return img_t, target


class ArTaxOrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,
        num_workers: int = 4,
        val_split: float = 0.1,
        seed: int = 42,
        pin_memory: bool = False
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.pin_memory = pin_memory

        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None

    def _discover_orders(self, artaxor_root: Path) -> List[str]:
        # class folders are the "orders"
        orders = []
        for d in artaxor_root.iterdir():
            if d.is_dir() and (d / "annotations").exists():
                orders.append(d.name)
        orders = sorted(orders)
        if not orders:
            raise FileNotFoundError(f"No order folders with 'annotations' found under {artaxor_root}")
        return orders

    def _build_samples(self, artaxor_root: Path, orders: List[str]) -> List[Sample]:
        samples: List[Sample] = []
        for order in orders:
            order_dir = artaxor_root / order
            ann_dir = order_dir / "annotations"
            if not ann_dir.exists():
                continue
            # BEFORE: for ann_path in ann_dir.rglob("*.json"):
            # FIX: .rglob return files in order depending on the filesystem, then shuffle will shuffle at the set seed
            # but from the flexible starting set, so for consistency across runs we sort files first, so it 
            # will always be the same set
            for ann_path in sorted(ann_dir.rglob("*.json")):
                samples.append(Sample(image_path=order_dir / "dummy.jpg", ann_path=ann_path, order_name=order))
        if not samples:
            raise FileNotFoundError(f"No JSON annotations found under {artaxor_root}")
        return samples

    def setup(self, stage: str | None = None) -> None:
        artaxor_root = self.data_dir / "ArTaxOr"
        if not artaxor_root.exists():
            # BEFORE: if (self.data_dir / "annotation").exists(): artaxor_root = self.data_dir
            # FIX: if we point to "ArTaxOr" directly the if checks for the unexisting annotations folder,
            # so instead we look through subdirs (which give class ims/annots), and check whether annotations exist there
            
            # Sometimes users point directly at ArTaxOr
            if any((d / "annotations").exists() for d in self.data_dir.iterdir() if d.is_dir()):
                artaxor_root = self.data_dir
            else:
                raise FileNotFoundError(f"Expected ArTaxOr folder under {self.data_dir}")

        orders = self._discover_orders(artaxor_root)
        self.class_to_idx = {name: i + 1 for i, name in enumerate(orders)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        all_samples = self._build_samples(artaxor_root, orders)

        rng = random.Random(self.seed)
        rng.shuffle(all_samples)
        n_val = max(1, int(len(all_samples) * self.val_split))
        val_samples = all_samples[:n_val]
        train_samples = all_samples[n_val:]

        self.train_ds = ArTaxOrVoTTDataset(train_samples, self.class_to_idx, train=True)
        self.val_ds = ArTaxOrVoTTDataset(val_samples, self.class_to_idx, train=False)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )


# -----------------------------
# Lightning module
# -----------------------------
class FasterRCNNLit(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-6,
        weight_decay: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        self.model = model

        self.map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def forward(self, images: List[Tensor], targets: List[Dict[str, Tensor]] | None = None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)  # returns losses in train mode
        loss = sum(loss_dict.values())
        # FIX: added batch_size=len(images) to self.log to remove warning 
        self.log("train/loss", loss, batch_size=len(images), prog_bar=True, on_step=True, on_epoch=True)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, batch_size=len(images), prog_bar=False, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        self.map_metric.update(preds, targets)

    def on_validation_epoch_end(self):
        metrics = self.map_metric.compute()
        self.log("val/map", metrics["map"], prog_bar=True)
        self.log("val/map_50", metrics["map_50"], prog_bar=True)
        self.map_metric.reset()

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1) # was gamma=0.01, changed for better training
        return {"optimizer": opt, "lr_scheduler": sch}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path containing ArTaxOr/...")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4) # was 1e-6, changed for better training
    # BEFORE: None
    # FIX: we had weight_decay field in FasterRCNNLit init, however 
    # we did not have argparse param for it and never passed it to the module
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # BEFORE: None
    # FIX: we set accelerator=auto, and pin_memory=True, which raises warning on CPU/MPS, script still runs,
    # however for better performance we let the user set the accelerator and pin memory only on GPU
    parser.add_argument("--accelerator", default="auto", choices=["auto", "cpu", "gpu", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    pin_memory = torch.cuda.is_available() and args.accelerator in ("auto", "gpu")
    dm = ArTaxOrDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        pin_memory=pin_memory
    )
    dm.setup()

    num_classes = len(dm.class_to_idx) + 1  # + background (0)
    lit = FasterRCNNLit(num_classes=num_classes, lr=args.lr, weight_decay=args.weight_decay)

    # BEFORE: ckpt = ModelCheckpoint(monitor="val/map_50", mode="max", save_top_k=1, filename="frcnn-{epoch:02d}-{val_map50:.4f}")
    # FIX: checkpoint named after val_map50, when the metric is val/map_50, also added better naming consistency: set explicitly, not auto-inserted params
    ckpt = ModelCheckpoint(monitor="val/map_50", mode="max", save_top_k=1, filename="frcnn-epoch{epoch:02d}-map50_{val/map_50:.4f}", auto_insert_metric_name=False)
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=1 if args.accelerator != "cpu" else "auto",
        callbacks=[ckpt, lrmon],
        log_every_n_steps=20
    )
    trainer.fit(lit, datamodule=dm)
    print("Best checkpoint:", ckpt.best_model_path)


# python Lab1/object_detection_pipeline.py --data_dir Lab1/sample_data --batch_size 2 --num_workers 1 --val_split 0.3 --max_epochs 8 --seed 42 --lr 1e-4
if __name__ == "__main__":
    main()