from ultralytics import YOLO
from pathlib import Path

# Vergleich verschiedener Modelle auf realen und synthetischen Datensätzen
MODEL_N_PATH = Path("")
MODEL_S_PATH = Path("")
MODEL_M_PATH = Path("")
DATASET_REAL_PATH = Path("")
DATASET_SYNTHETIC_PATH = Path("")

test_model_paths = [MODEL_N_PATH, MODEL_S_PATH, MODEL_M_PATH]
test_model_names = ["Model N", "Model S", "Model M"]

for model_path, model_name in zip(test_model_paths, test_model_names):
    for dataset_path in [DATASET_REAL_PATH, DATASET_SYNTHETIC_PATH]:
        model = YOLO(model_path / "weights" / "best.pt")

        metrics = model.val(
            data=dataset_path / "data.yaml",
            split="test", 
            imgsz=640,
            conf=0.1,           
            save_json=True,        
            save_txt=True,         
            plots=True          
        )
        print(f"----------- {model_name} ----------")
        print("mAP50-95:", metrics.box.map)       # mean AP (IoU 0.5:0.95)
        print("mAP50:", metrics.box.map50)        # mean AP (IoU 0.5)
        print("Precision:", metrics.box.mp)       # mean Precision
        print("Recall:", metrics.box.mr)          # mean Recall
