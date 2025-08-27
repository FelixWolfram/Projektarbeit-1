import shutil
import csv
from pathlib import Path
import torch
import json
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# -------------------------------------------------
# Konfiguration
# -------------------------------------------------
REAL_DIR = Path("")
SYNTHETIC_DIR = Path("")
CHECKPOINT_ROOT = Path("") # Modell-Checkpoint Ordner
if not CHECKPOINT_ROOT.exists():
    raise ValueError(f"Checkpoint-Ordner nicht gefunden: {CHECKPOINT_ROOT}")

IOU_THR = 0.50  # Schwelle für als korrekt klassifierte Bounding Box
CONF_THR = 0.1   # Anzeige/Export-Schwelle für Predictions
NMS_IOU_THR = 0.5  # IoU-Threshold für Non-Maximum Suppression


img_dir_real   = REAL_DIR / "images"
label_dir_real = REAL_DIR / "labels"
img_dir_synthetic   = SYNTHETIC_DIR / "images" / "test"
label_dir_synthetic = SYNTHETIC_DIR / "labels" / "test"

# -------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------
# aktuellsten Checkpoint finden
def find_best_checkpoint(root: Path, metric="eval_loss"):
    if not root.exists():
        return None
        
    candidates = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not candidates:
        return None
    
    best_checkpoint = None
    best_metric_value = float('inf')
        
    for checkpoint_dir in candidates:
        trainer_state_file = checkpoint_dir / "trainer_state.json"
        
        if not trainer_state_file.exists():
            print(f"  {checkpoint_dir.name}: Keine trainer_state.json gefunden")
            continue
            
        with open(trainer_state_file, 'r') as f:
            trainer_state = json.load(f)
        
        # Suche die Metrik in log\_history
        log_history = trainer_state.get('log_history', [])
        metric_values = []
        
        for entry in log_history:
            if metric in entry:
                metric_values.append(entry[metric])
        
        if not metric_values:
            print(f"  {checkpoint_dir.name}: Metrik '{metric}' nicht gefunden")
            continue
        
        # aktuellsten Wert dieser Metrik nehmen
        current_value = metric_values[-1]

        # Prüfe ob dieser Checkpoint besser ist
        if (current_value < best_metric_value):
            best_metric_value = current_value
            best_checkpoint = checkpoint_dir
                            
    return str(best_checkpoint)

# yolo format zu x1y1x2y2
def read_yolo_to_xyxy(label_path: Path, img_w: int, img_h: int):
    boxes_xyxy, labels = [], []
    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                c, cx, cy, bw, bh = map(float, parts)
                # normierte YOLO (cx,cy,w,h) -> Pixel-xyxy
                x = (cx - bw / 2.0) * img_w
                y = (cy - bh / 2.0) * img_h
                w = bw * img_w
                h = bh * img_h
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(img_w - 1, x + w), min(img_h - 1, y + h)
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                boxes_xyxy.append([x1, y1, x2, y2])
                labels.append(int(c))
    return boxes_xyxy, labels

# x1y1x2y2 zu cx,cy,w,h (normiert)
def xyxy_to_yolo_norm(x1, y1, x2, y2, img_w, img_h):
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    # normalisieren
    return cx / img_w, cy / img_h, w / img_w, h / img_h

# Boxen zeichnen
def draw_boxes(img: Image.Image, boxes_xyxy, labels=None, scores=None, id2label=None, color=(0, 255, 0)):
    draw = ImageDraw.Draw(img)
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in b]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        txt = ""
        if labels is not None:
            lab_id = int(labels[i])
            lab = id2label.get(lab_id, str(lab_id)) if id2label else str(lab_id)
            txt = lab
        if scores is not None:
            sc = float(scores[i])
            txt = f"{txt} {sc:.2f}" if txt else f"{sc:.2f}"
        if txt:
            draw.text((x1 + 3, max(0, y1 - 12)), txt, fill=color)

# IoU zweier Boxen (x1y1x2y2) berechnen
def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
    bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def non_max_suppression(boxes, scores, labels, iou_threshold=0.5):
    # Non-Maximum Suppression: Entfernt überlappende Boxes mit niedrigeren Confidence-Scores

    # Konvertiere zu Listen falls nötig
    boxes = [list(box) for box in boxes]
    scores = list(scores)
    labels = list(labels)
    
    # Indizes nach Score sortieren (höchste zuerst)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    suppressed = set()
    
    for i in indices:
        if i in suppressed:
            continue
            
        keep.append(i)
        
        # Vergleiche mit allen verbleibenden Boxes
        for j in indices:
            if j == i or j in suppressed:
                continue
                
            # Prüfe nur Boxes derselben Klasse
            # aktuell nur eine Klasse implementiert, für zukünftige Erweiterungen jedoch relevant
            if labels[i] == labels[j]:
                iou = box_iou_xyxy(boxes[i], boxes[j])
                if iou > iou_threshold:
                    suppressed.add(j)
    
    # Behalte nur nicht-unterdrückte Detections
    filtered_boxes = [boxes[i] for i in keep]
    filtered_scores = [scores[i] for i in keep]
    filtered_labels = [labels[i] for i in keep]
    
    return filtered_boxes, filtered_scores, filtered_labels

# Boxen abgleichen (IoU >= Schwelle)
def match_detections(pred_boxes, gt_boxes, iou_thr=0.5):
    matches = []
    used_pred = set()
    used_gt = set()
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            iou = box_iou_xyxy(pb, gb)
            if iou >= iou_thr:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    for iou, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, iou))
    tp_idx = {pi for (pi, _, _) in matches}
    fp_idx = set(range(len(pred_boxes))) - tp_idx
    fn_cnt = len(gt_boxes) - len(tp_idx)
    return matches, tp_idx, fp_idx, fn_cnt

# Vergleichsbild zeichnen (GT rot, Preds grün/gelb)
def draw_compare(img: Image.Image, gt_boxes, pred_boxes, pred_scores, matches, id2label=None):
    # GT rot
    draw_boxes(img, gt_boxes, labels=[0]*len(gt_boxes), scores=None, id2label=id2label, color=(255, 0, 0))
    # Preds: grün (TP) / gelb (FP)
    tp_pred_idx = {pi for (pi, _, _) in matches}
    for i, b in enumerate(pred_boxes):
        col = (0, 255, 0) if i in tp_pred_idx else (255, 255, 0)
        draw_boxes(img, [b], labels=[0], scores=[pred_scores[i]], id2label=id2label, color=col)
    # IoU-Text an TPs
    dr = ImageDraw.Draw(img)
    for (pi, _, iou) in matches:
        x1, y1, _, _ = pred_boxes[pi]
        dr.text((int(x1)+3, max(0, int(y1)-24)), f"IoU {iou:.2f}", fill=(0, 255, 0))

# -------------------------------------------------
# Modell + Prozessor laden
# -------------------------------------------------
load_from = find_best_checkpoint(CHECKPOINT_ROOT)
processor = AutoImageProcessor.from_pretrained(load_from)
model = AutoModelForObjectDetection.from_pretrained(load_from)
id2label = getattr(model.config, "id2label", None) or {0: "object"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
print(f"Lade Modell aus: {load_from} | Device: {device}")

for data_type in ["real", "synthetic"]:
    # Ordnerstruktur definieren
    out_base = Path("runs/test_daten") / load_from.split("\\")[0] / data_type
    pred_images_dir = out_base / "pred" / "images"
    pred_labels_dir = out_base / "pred" / "labels"
    gt_images_dir   = out_base / "gt" / "images"
    gt_labels_dir   = out_base / "gt" / "labels"
    copy_images_dir = out_base / "images"
    pred_csv_path   = out_base / "predictions.csv"
    compare_images_dir = out_base / "compare_images"
    metrics_txt_path = out_base / "metrics.txt"

    # wenn Ordner schon existiert, löschen
    if out_base.exists():
        print(f"Entferne alten Test-Ordner: {out_base}")
        shutil.rmtree(out_base)

    # je nach Datentyp die Quellordner setzen
    img_dir, label_dir = img_dir_synthetic, label_dir_synthetic
    if data_type == "real":
        img_dir = img_dir_real
        label_dir = label_dir_real

    # -------------------------------------------------
    # Ordner vorbereiten
    # -------------------------------------------------
    for data_type in ["real", "synthetic"]:
        for d in [pred_images_dir, pred_labels_dir, gt_images_dir, gt_labels_dir, copy_images_dir, compare_images_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # Testbilder kopieren (wenn Ordner leer)
    if not any(copy_images_dir.iterdir()):
        for fn in sorted(img_dir.iterdir()):
            if fn.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                shutil.copy2(fn, copy_images_dir / fn.name)

    # -------------------------------------------------
    # Durchlauf: GT zeichnen + Predictions exportieren
    # -------------------------------------------------
    # Metriken und Variablen initialisieren
    rows = []
    n_images = 0 
    n_gt_total = 0
    n_pred_total = 0
    TP = 0
    FP = 0
    FN = 0
    sum_iou_tp = 0.0
    n_tp_for_miou = 0
    ap_records = []
    all_pred_boxes = []
    all_pred_scores = []  
    all_gt_boxes = []

    img_files = [f for f in sorted(img_dir.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    for img_path in img_files:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        stem = img_path.stem
        n_images += 1

        # --- Ground Truth laden und Bild mit GT speichern ---
        gt_txt = label_dir / f"{stem}.txt"
        gt_boxes, gt_labels = read_yolo_to_xyxy(gt_txt, w, h)
        n_gt_total += len(gt_boxes)

        gt_img = image.copy()
        if gt_boxes:
            draw_boxes(gt_img, gt_boxes, labels=gt_labels, scores=None, id2label=id2label, color=(255, 0, 0))
        gt_img.save(gt_images_dir / f"{stem}.jpg")

        if gt_txt.exists():
            shutil.copy2(gt_txt, gt_labels_dir / gt_txt.name)

        # --- Prediction ---
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        processed = processor.post_process_object_detection(
            outputs,
            threshold=CONF_THR,
            target_sizes=torch.tensor([[h, w]], device=device)
        )[0]

        preds_img = image.copy()
        pred_boxes = processed["boxes"].detach().cpu().tolist()
        pred_scores = processed["scores"].detach().cpu().tolist()
        pred_labels = processed["labels"].detach().cpu().tolist()
        
        # Non-Maximum Suppression anwenden
        pred_boxes_nms, pred_scores_nms, pred_labels_nms = non_max_suppression(
            pred_boxes, pred_scores, pred_labels, iou_threshold=NMS_IOU_THR
        )
                
        # Verwende NMS-gefilterte Predictions weiter
        pred_boxes = pred_boxes_nms
        pred_scores = pred_scores_nms  
        pred_labels = pred_labels_nms
        
        n_pred_total += len(pred_boxes)

        all_pred_boxes.append(pred_boxes)
        all_pred_scores.append(pred_scores)
        all_gt_boxes.append(gt_boxes)

        # annotierte Predictions speichern
        draw_boxes(preds_img, pred_boxes, labels=pred_labels, scores=pred_scores, id2label=id2label, color=(0, 255, 0))
        preds_img.save(pred_images_dir / f"{stem}.jpg")

        # YOLO-Pred-Labels speichern: 
        out_txt = pred_labels_dir / f"{stem}.txt"
        with open(out_txt, "w", newline="\n") as f:
            for c, score, (x1, y1, x2, y2) in zip(pred_labels, pred_scores, pred_boxes):
                cx, cy, bw, bh = xyxy_to_yolo_norm(x1, y1, x2, y2, w, h)
                f.write(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(score):.6f}\n")
                rows.append([
                    img_path.name, int(c), id2label.get(int(c), str(int(c))),
                    float(score),
                    float(x1), float(y1), float(x2), float(y2),
                    float(cx), float(cy), float(bw), float(bh)
                ])

        # --- Vergleiche + Metriken ---
        matches, tp_idx, fp_idx, fn_cnt = match_detections(pred_boxes, gt_boxes, iou_thr=IOU_THR)
        TP += len(tp_idx)
        FP += len(fp_idx)
        FN += fn_cnt
        for (_, _, iou) in matches:
            sum_iou_tp += iou
            n_tp_for_miou += 1
        for i, s in enumerate(pred_scores):
            ap_records.append((float(s), 1 if i in tp_idx else 0))

        # Vergleichsbild speichern: GT rot, TP grün, FP gelb + IoU
        cmp_img = image.copy()
        draw_compare(cmp_img, gt_boxes, pred_boxes, pred_scores, matches, id2label=id2label)
        compare_images_dir.mkdir(parents=True, exist_ok=True)
        (compare_images_dir / f"{stem}.jpg").parent.mkdir(parents=True, exist_ok=True)
        cmp_img.save(compare_images_dir / f"{stem}.jpg")

    # CSV speichern
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2", "cx", "cy", "w", "h"])
        writer.writerows(rows)

    # Metriken berechnen und metrics.txt schreiben
    def compute_pr_ap(records, n_gt):
        if n_gt == 0 or not records:
            return 0.0, [], []
        recs = sorted(records, key=lambda x: x[0], reverse=True)
        tp_cum = 0
        fp_cum = 0
        precisions = []
        recalls = []
        for _, is_tp in recs:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / max(1, (tp_cum + fp_cum)))
            recalls.append(tp_cum / max(1, n_gt))
        
        mrec = [0.0] + recalls + [1.0]
        mpre = [0.0] + precisions + [0.0]
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        ap = 0.0
        for i in range(1, len(mrec)):
            ap += (mrec[i] - mrec[i-1]) * mpre[i]
        return ap, precisions, recalls
    
    # Funktion, um die AP für mehrere IoU zu berechnen (für AP50-95)
    def compute_ap_multiple_iou(pred_boxes, pred_scores, gt_boxes_all, iou_thresholds):
        aps = []
        
        for iou_thr in iou_thresholds:
            ap_records = []
            for _, (pred_b, pred_s, gt_b) in enumerate(zip(pred_boxes, pred_scores, gt_boxes_all)):
                if not pred_b:  # Keine Predictions
                    continue
                    
                _, tp_idx, _, _ = match_detections(pred_b, gt_b, iou_thr=iou_thr)
                
                for i, s in enumerate(pred_s):
                    ap_records.append((float(s), 1 if i in tp_idx else 0))
            
            total_gt = sum(len(gt_b) for gt_b in gt_boxes_all)
            ap, _, _ = compute_pr_ap(ap_records, total_gt)
            aps.append(ap)
        
        return aps

    precision = TP / max(1, (TP + FP))
    recall    = TP / max(1, n_gt_total)
    f1        = 2 * precision * recall / max(1e-12, (precision + recall))
    miou      = (sum_iou_tp / n_tp_for_miou) if n_tp_for_miou > 0 else 0.0
    ap50, _, _ = compute_pr_ap(ap_records, n_gt_total)

    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5, 0.55, 0.6, ..., 0.95
    aps_multiple = compute_ap_multiple_iou(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thresholds)
    ap50_95 = sum(aps_multiple) / len(aps_multiple) if aps_multiple else 0.0

    # Metriken in Datei schreiben
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Images: {n_images}\n")
        f.write(f"GT boxes: {n_gt_total}\n")
        f.write(f"Pred boxes: {n_pred_total}\n")
        f.write(f"IoU threshold: {IOU_THR}\n")
        f.write(f"NMS IoU threshold: {NMS_IOU_THR}\n")
        f.write(f"Confidence threshold: {CONF_THR}\n")
        f.write(f"TP: {TP}\nFP: {FP}\nFN: {FN}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1:        {f1:.4f}\n")
        f.write(f"mIoU(TP):  {miou:.4f}\n")
        f.write(f"AP@{IOU_THR}: {ap50:.4f}\n")
        f.write(f"AP@0.50-0.95: {ap50_95:.4f}\n")
    
print(f"Fertig. Export unter: {out_base}")
print(f"- Predictions: {pred_images_dir} (Bilder), {pred_labels_dir} (YOLO-TXT mit conf)")
print(f"- Ground Truth: {gt_images_dir} (Bilder), {gt_labels_dir} (GT-TXT Kopie)")
print(f"- Compare: {compare_images_dir} (GT=rot, TP=grün, FP=gelb, IoU-Text an TPs)")
print(f"- CSV: {pred_csv_path}")
print(f"- Metrics: {metrics_txt_path}")