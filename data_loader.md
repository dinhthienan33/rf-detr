
dataset/
└── observing_train/
    ├── annotations/
    │   └── annotations.json
    └── samples/
        ├── Backpack_0/
        │   ├── object_images/
        │   │   ├── img_1.jpg
        │   │   ├── img_2.jpg
        │   │   └── img_3.jpg
        │   └── drone_video.mp4
        ├── Backpack_1/
        ├── Jacket_0/
        ├── Jacket_1/
        └── ...
└── public_test/
    ├── public_test/
       └── samples/
           ├── Blackbox_0/
           │   ├── object_images/
           │   │   ├── img_1.jpg
           │   │   ├── img_2.jpg
           │   │   └── img_3.jpg
           │   └── drone_video.mp4
           ├── Blackbox_1/
           ├── ...

## ⚙️ 1️⃣ Cấu trúc `annotations.json` (giả định chuẩn AeroEyes)

Thông thường file này chứa dạng:

```json
[
  {
    "video_id": "Backpack_0",
    "annotations": [
      {
        "frame_id": 10,
        "bboxes": [{"x1": 420, "y1": 300, "x2": 550, "y2": 420}]
      },
      ...
    ]
  },
  ...
]
```

Nếu khác, bạn chỉ cần chỉnh phần `load_annotations()` mình viết bên dưới.

---

## 🧩 2️⃣ Code Dataset & Dataloader

```python
import os
import json
import cv2
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class AeroEyesDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_frames_per_video=200):
        """
        Args:
            root_dir (str): đường dẫn tới dataset/observing_train
            transform: torchvision transform cho ảnh
            max_frames_per_video: số frame trích ra tối đa mỗi video
        """
        self.root_dir = root_dir
        self.samples_dir = os.path.join(root_dir, "samples")
        self.ann_path = os.path.join(root_dir, "annotations", "annotations.json")
        self.transform = transform
        self.max_frames = max_frames_per_video

        # load annotation mapping
        self.annotations = self.load_annotations(self.ann_path)
        # build sample list
        self.sample_list = self.build_samples()

    def load_annotations(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # map video_id -> {frame_id -> bbox}
        ann_dict = {}
        for item in data:
            vid = item["video_id"]
            boxes = {}
            for ann_block in item.get("annotations", []):
                frame_id = ann_block["frame_id"]
                bboxes = ann_block.get("bboxes", [])
                boxes[frame_id] = bboxes
            ann_dict[vid] = boxes
        return ann_dict

    def build_samples(self):
        """
        Tạo list các sample:
        [
          {
            'video_id': 'Backpack_0',
            'ref_imgs': [...],
            'frame_id': 45,
            'frame_path': '<temp extracted frame path hoặc video>',
            'bbox': [x1, y1, x2, y2],
            'conf': 1
          }
        ]
        """
        samples = []
        for video_name in sorted(os.listdir(self.samples_dir)):
            video_dir = os.path.join(self.samples_dir, video_name)
            video_path = os.path.join(video_dir, "drone_video.mp4")
            ref_dir = os.path.join(video_dir, "object_images")

            if not os.path.exists(video_path) or not os.path.exists(ref_dir):
                continue

            ref_imgs = sorted([
                os.path.join(ref_dir, f)
                for f in os.listdir(ref_dir)
                if f.lower().endswith((".jpg", ".png"))
            ])

            ann_frames = list(self.annotations.get(video_name, {}).keys())
            ann_frames = [int(f) for f in ann_frames]
            ann_frames.sort()

            for frame_id in ann_frames:
                bboxes = self.annotations[video_name][frame_id]
                if len(bboxes) == 0:
                    continue
                # chỉ lấy bbox đầu tiên (vì 1 object duy nhất)
                bbox = bboxes[0]
                bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                samples.append({
                    "video_id": video_name,
                    "ref_imgs": ref_imgs,
                    "video_path": video_path,
                    "frame_id": frame_id,
                    "bbox": bbox,
                    "conf": 1
                })

            # thêm negative sample (không chứa object)
            if len(ann_frames) > 5:
                for neg_id in random.sample(ann_frames, min(3, len(ann_frames))):
                    samples.append({
                        "video_id": video_name,
                        "ref_imgs": ref_imgs,
                        "video_path": video_path,
                        "frame_id": neg_id + 5,  # frame lệch 5 frame
                        "bbox": [0, 0, 0, 0],
                        "conf": 0
                    })
        return samples

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]

        # Load 3 reference images
        ref_imgs = [Image.open(p).convert("RGB") for p in sample["ref_imgs"]]

        # Load video frame
        frame = self.load_frame(sample["video_path"], sample["frame_id"])

        # Apply transforms
        if self.transform:
            ref_imgs = [self.transform(img) for img in ref_imgs]
            frame = self.transform(frame)
        else:
            ref_imgs = [transforms.ToTensor()(img) for img in ref_imgs]
            frame = transforms.ToTensor()(frame)

        bbox = torch.tensor(sample["bbox"], dtype=torch.float32)
        conf = torch.tensor([sample["conf"]], dtype=torch.float32)
        return ref_imgs, frame, bbox, conf

    def load_frame(self, video_path, frame_id):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Cannot read frame {frame_id} from {video_path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
```

---

## 🧠 3️⃣ Sử dụng với DataLoader

```python
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

train_dataset = AeroEyesDataset(
    root_dir="dataset/observing_train",
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

for ref_imgs, frame, bbox, conf in train_loader:
    print(len(ref_imgs), ref_imgs[0].shape, frame.shape, bbox.shape, conf.shape)
    break
```

---

## 🧠 4️⃣ Kết quả `__getitem__`

Mỗi sample trả về:

```python
(
  [tensor_ref1, tensor_ref2, tensor_ref3],  # 3 ảnh mẫu
  tensor_frame,                             # frame ảnh
  tensor([x1, y1, x2, y2]),                 # bbox ground-truth
  tensor([1.0])                             # 1 nếu có object, 0 nếu negative
)
```
