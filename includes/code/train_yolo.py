from ultralytics import YOLO

yolo_model_name = 'yolo11n.pt' # yolo11n.pt, yolo11s.pt, yolo11m.pt
model = YOLO(yolo_model_name)
dataset_name = "dataset_random_1077"
output_dir_name = "1 - exp_200_1077_imgs_n"


model.train(
    data=f"{dataset_name}/data.yaml",
    epochs=200,
    patience=20,
    imgsz=640,
    batch=16,
    project="runs/train",     
    name=output_dir_name,              
    workers=4,                
    verbose=True,
    device=0,         

    degrees=40,     # rotation random between -180 and 180 degrees
    hsv_h=0.5,      # hue
    hsv_s=0.8,      # saturation
    hsv_v=0.6,      # brightness
    translate=0.2,  # translates images
    scale=0.4,      # scale between 0.6 and 1.4
    shear=6,        # shear along x and y axis
    lr0=0.01,       # learning rate at the beginning (default 0.01)
    perspective=0.0005, # simulates different perspective
    flipud=0.5,     # 50\% chance to flip the image vertically
    fliplr=0.5,     # 50\% chance to flip the image horizontally
    bgr=0.2,        # 20\% chance to flip color channels from rgb to bgr
    erasing=0.3,    # 30\% chance that a certain part of the image is erased
    weight_decay=0.003, # regularization to avoid overfitting
    dropout=0.1,   # 10\% of activations are randomly set to zero during training

    single_cls=True, 
    classes=[0],
    mosaic = 0.1,
    mixup = 0.0,
    cutmix = 0.2,
    copy_paste = 0.0
)