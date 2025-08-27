# Klassen zu Id's
map_classes = {
    "part": 0,
}


# Einstellungen und Variablen festlegen
import os

OUTPUT_FOLDER = ""
CAMERA = ""
DATASET_NAME = ""
OUTPUT_DIR = ""
height, width = 720, 1280
SPLIT_RATIO = [0.7, 0.2, 0.1] # train, val, test

all_folders = os.listdir(OUTPUT_FOLDER)
annotation_paths = [f"annot_path/{folder}/Replicator/{camera}/object_detection" for folder in all_folders]
yolo_rgb_paths = [f"annot_path/{folder}/Replicator/{camera}/rgb" for folder in all_folders]



# .txt Dateien aufräumen und in das YOLO-Format übertragen
# .txt Dateien müssen aufgeräumt werden, da es Fehler bei den Annotationen durch den Replicator gibt
from collections import Counter
import shutil

os.makedirs(OUTPUT_DIR, exist_ok=True)
if os.path.exists("yolo_annotated"):
    shutil.rmtree("yolo_annotated")
os.makedirs("yolo_annotated", exist_ok=True)

for idx, annot_path in enumerate(annotation_paths):
    for file in os.listdir(annot_path):
        if not file.endswith(".txt"):
            continue
        
        file_path = os.path.join(annotation_paths[idx], file)
        with open(file_path, "r") as f:
            raw_lines = [ln.strip() for ln in f if ln.strip()]

        counts = Counter(raw_lines)  # Zählen, wie oft jede Zeile in der Datei vorkommt

        output_lines = set()
        for line in raw_lines:
            if counts[line] <= 1:
                continue    # Zeilen, welche nur einmal vorkommen, werden gefiltert

            items = line.split(" ")
            items[0] = items[0].replace("k09,", "").replace("k09", "").replace("sheet", "")

            if items[0] in list(map_classes.keys()):
                [class_id, x_min, y_min, x_max, y_max] = [map_classes[items[0]], int(items[4]), int(items[5]), int(items[6]), int(items[7])]
                
                x_center = ((x_min) + (x_max)) / 2 / width
                y_center = ((y_min) + (y_max)) / 2 / height
                item_width = ((x_max) - (x_min)) / width
                item_height = ((y_max) - (y_min)) / height

                output_data = " ".join([str(class_id), str(x_center), str(y_center), str(item_width), str(item_height)])

                output_lines.add(output_data + "\n")
        
        file_path = os.path.join(OUTPUT_DIR, "data_" + str(idx))
        os.makedirs(file_path, exist_ok=True)
        with open(os.path.join(file_path, file), "w") as out_file:
            out_file.writelines(list(output_lines))


# Aufteilen in ein Trainings-, Validierungs- und Testdatensatz
import random

for i, yolo_rgb_path in enumerate(yolo_rgb_paths):
    yolo_annoted_path = os.path.join(OUTPUT_DIR, f"data_{i}") 

    all_images = [f for f in os.listdir(yolo_rgb_path) if f.endswith(".png")]
    all_labels = [f for f in os.listdir(yolo_annoted_path) if f.endswith(".txt")]

    if len(all_images) != len(all_labels):
        print(f"Warning: Mismatch in number of images and labels in run {i}. Images: {len(all_images)}, Labels: {len(all_labels)}")

    # shuffle the images with their corresponding annotations
    pairs = [(img, label) for (img, label) in zip(all_images, all_labels)]
    random.shuffle(pairs)

    print(pairs)

    n = len(all_images)

    train, val = int(n * SPLIT_RATIO[0]), int(n * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))

    splits = {
        "train": pairs[:train],
        "val": pairs[train:val],
        "test": pairs[val:]
    }

    for split, files in splits.items():
        os.makedirs(f"{DATASET_NAME}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_NAME}/labels/{split}", exist_ok=True)

        for img_file, label_file in files:
            shutil.copy(f"{yolo_rgb_path}/{img_file}", f"{DATASET_NAME}/images/{split}/{i}_{img_file}")
            shutil.copy(f"{yolo_annoted_path}/{label_file}", f"{DATASET_NAME}/labels/{split}/{i}_{label_file}")


# .yaml Datei erstellen
import yaml

class_names = list(map_classes.keys())

data = {
    'path': f'./{DATASET_NAME}',
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': len(class_names),
    'names': class_names
}

with open(f'{DATASET_NAME}/data.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)