# Einstellungen des Trainings
from pathlib import Path

checkpoint = "PekingU/rtdetr_v2_r18vd"
image_size = 640
DATASET_PATH = ""
OUTPUT_DIR = "rtdetr-v2-r18-finetune-1"

# Prozessor laden
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(
    checkpoint,
    do_resize=True,
    size={"width": image_size, "height": image_size},
    use_fast=True,
)


# Data Augmentation für das Training
import albumentations as A

train_augmentation_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-5, 5), shear=(-5, 5), p=0.7),
        A.OneOf([A.MotionBlur(blur_limit=3, p=1.0), A.GaussianBlur(blur_limit=3, p=1.0), A.GaussNoise(var_limit=(5.0, 30.0), p=1.0)], p=0.2),
        A.RandomFog(p=0.1),              
        A.ColorJitter(p=0.1),     
        A.ToGray(p=0.1),
        A.RandomGamma(gamma_limit=(80,120), p=0.1)
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25, min_width=1, min_height=1),
)

# keine Augmentation für die Validierung
validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1, min_width=1, min_height=1),
)


# Dataset-Klasse für das Laden der Bilder und Labels für Trainings-, Validierungs- und Testdatensatz
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, image_processor, transform=None):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.image_processor = image_processor
        self.transform = transform

        self.image_files = [f for f in sorted(img_dir.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        self.label_files = [f for f in sorted(lbl_dir.iterdir()) if f.suffix.lower() in {".txt"}]

        self.images = [Image.open(self.image_files[element_idx]).convert("RGB") for element_idx in range(len(self.image_files))]
        
        print(f"Dataset: {len(self.image_files)} Bilder gefunden")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.images[idx]
        W, H = image.size

        label_path = self.label_files[idx]
        
        # Labels laden und konvertieren
        boxes, categories = read_yolo_to_coco(label_path, W, H)
        
        # Zu numpy für Augmentationen
        image_array = np.array(image)
        
        # Augmentationen anwenden
        if self.transform and len(boxes) > 0:
            try:
                transformed = self.transform(
                    image=image_array, 
                    bboxes=boxes, 
                    category=categories
                )
                image_array = transformed["image"]
                boxes = transformed["bboxes"]
                categories = transformed["category"]
            except Exception as e:
                print(f"Augmentation failed for {label_path.name}: {e}")
                # Fallback: Original verwenden

        # Format für image processor
        formatted_annotations = {
            "image_id": idx,
            "annotations": [
                {
                    "image_id": idx,
                    "category_id": cat,
                    "bbox": list(box),
                    "iscrowd": 0,
                    "area": box[2] * box[3],
                } for cat, box in zip(categories, boxes)
            ]
        }
        
        # Image processor anwenden
        result = self.image_processor(
            images=image_array, 
            annotations=formatted_annotations, 
            return_tensors="pt"
        )
        
        # Batch-Dimension entfernen
        return {
            "pixel_values": result["pixel_values"].squeeze(0), 
            "labels": result["labels"][0],  
        }

# Ersetze die kaputte ImagesDataset:
train_dataset = ImageDataset(
    img_dir=DATASET_PATH / "images" / "train", 
    lbl_dir=DATASET_PATH / "labels" / "train", 
    image_processor=image_processor, 
    transform=train_augmentation_and_transform
)

validation_dataset = ImageDataset(
    img_dir=DATASET_PATH / "images" / "val", 
    lbl_dir=DATASET_PATH / "labels" / "val", 
    image_processor=image_processor, 
    transform=validation_transform
)

test_dataset = ImageDataset(
    img_dir=DATASET_PATH / "images" / "test", 
    lbl_dir=DATASET_PATH / "labels" / "test", 
    image_processor=image_processor, 
    transform=validation_transform
)


# DataLoader für das Training
import torch

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


# Modell laden
from transformers import AutoModelForObjectDetection

model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)


# Hyperparameter für das Training definieren
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import torch

use_cuda = torch.cuda.is_available()

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    fp16=use_cuda,    # nur auf GPU
    learning_rate=1e-5,
    weight_decay=0.05,
    num_train_epochs=220,
    warmup_ratio=0.10,
    lr_scheduler_type="cosine",
    eval_strategy="steps",# gültige Werte: "no" | "steps" | "epoch"
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    save_total_limit=500,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_strategy="steps",
    logging_first_step=True,
    logging_steps=50,
    disable_tqdm=False, # zeige Progressbar in Notebooks
    load_best_model_at_end=True,
    remove_unused_columns=False, # wichtig für Object Detection
    dataloader_num_workers=0,       
    dataloader_pin_memory=use_cuda,
    report_to="none",
    seed=42
)

# Trainier initialisieren
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=collate_fn,
    processing_class=image_processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)] # wenn sich Leistung nicht mehr verbessert
)

# Training starten
trainer.train()