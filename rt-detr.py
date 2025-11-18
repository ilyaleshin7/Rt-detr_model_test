from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
results = model("path/to/bus.jpg")


