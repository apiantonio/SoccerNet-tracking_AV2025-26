
@software{easychamp_player_detection_2025,
  author = {EasyChamp},
  title = {EasyChamp Player Detection YOLOv8},
  year = {2025},
  url = {https://huggingface.co/aabyzov/easychamp-player-detection-yolov8},
  publisher = {Hugging Face}
}

```python
from ultralytics import YOLO

# Load model from Hugging Face
model = YOLO('https://huggingface.co/aabyzov/easychamp-player-detection-yolov8/resolve/main/player_detection_best.pt')
```