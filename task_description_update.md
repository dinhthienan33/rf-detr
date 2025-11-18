Đây là một file `README.md` chi tiết, mô tả pipeline và các bước kỹ thuật để tùy chỉnh kho mã nguồn `roboflow/rf-detr` cho bài toán One-Shot Object Detection của bạn.

-----

# README: Tùy chỉnh RF-DETR cho Phát hiện Một Mẫu (One-Shot Object Detection)

## 1\. 🎯 Mô tả Dự án

Dự án này tùy chỉnh kho mã nguồn **RF-DETR** (từ Roboflow) để giải quyết bài toán **Phát hiện Một Mẫu (One-Shot Object Detection - OSOD)**.

**Bài toán:** Huấn luyện một mô hình có khả năng tìm thấy các đối tượng trong một `ảnh mục tiêu` (target image), dựa trên một `ảnh tham khảo` (reference image). Điều kiện đặc biệt là các đối tượng trong tập `test` **hoàn toàn mới** và không xuất hiện trong tập `train`.

**Giải pháp:** Chúng ta sẽ "phẫu thuật" RF-DETR, biến nó từ một mô hình **Phân loại (Classifier)** thành một mô hình **So khớp (Matcher)**. Kiến trúc này được gọi là **Siamese DETR**.

-----

## 2\. 🏛️ Kiến trúc Cốt lõi (Siamese DETR)

Chúng ta sẽ *không* dùng hai mô hình riêng biệt. Chúng ta sẽ tạo một mô hình End-to-End duy nhất với kiến trúc mạng Siamese (Siamese Network):

  * **Chia sẻ Trọng số (Weight Sharing):** Đây là mấu chốt. Cả hai nhánh (tham khảo và mục tiêu) sẽ sử dụng **chung** một bộ `Backbone` và `Encoder`. Điều này buộc các đặc trưng (features) phải nằm trong cùng một không gian (feature space), giúp việc so sánh trở nên khả thi.
  * **Nhánh Tham khảo (Reference Branch):**
      * Đầu vào: `ref_image` (ảnh đã crop).
      * Quy trình: `ref_image` → `Backbone` → `Encoder` → `Pooling` (ví dụ: Global Average Pooling).
      * Đầu ra: Một vector "dấu vân tay" `v_ref`.
  * **Nhánh Mục tiêu (Target Branch):**
      * Đầu vào: `target_image` (ảnh gốc, đầy đủ).
      * Quy trình: `target_image` → `Backbone` → `Encoder` → `Decoder`.
      * Đầu ra: 300 "ứng viên" (candidates).
  * **Các "Đầu" (Heads):**
      * **GIỮ LẠI:** `Decoder` (để tìm và tách biệt đối tượng) và `Regression Head` (để vẽ bbox).
      * **VỨT BỎ:** `Classification Head` (vì chúng ta không phân loại 80 lớp COCO).
      * **THAY THẾ:** Bằng một **Hàm Mất mát So khớp (Matching Loss)**, ví dụ như `Contrastive Loss`.

-----

## 3\. 🔧 Pipeline và Hướng dẫn Chỉnh sửa Code

Đây là các bước chi tiết để "phẫu thuật" kho mã nguồn `roboflow/rf-detr`.

### Bước 1: Chuẩn bị Dữ liệu

Đây là bước quan trọng nhất. Bạn cần tạo một `Dataset` và `DataLoader` tùy chỉnh.

  * **Định dạng:** Dataset của bạn phải trả về một bộ (tuple) cho mỗi `__getitem__`:
    `tuple(ref_image, target_image, ground_truth_dict)`
  * **`ref_image`:** Ảnh đã được crop chỉ chứa vật thể tham khảo.
  * **`target_image`:** Ảnh gốc, đầy đủ, chứa vật thể đó.
  * **`ground_truth_dict`:** Một dictionary chứa thông tin "sự thật nền" của `target_image`.
      * `"boxes"`: `torch.Tensor` (N, 4) chứa tọa độ `[x_center, y_center, w, h]` của N vật thể trong `target_image`.
      * `"labels"`: `torch.Tensor` (N) chứa ID của vật thể *tham khảo* (ví dụ: `0` nếu là vật thể đang tham chiếu, `1` cho các vật thể khác).
  * **Augmentation (Gia tăng dữ liệu):**
      * Phải áp dụng augmentation (xoay, lật, đổi màu, che...) **ĐỘC LẬP** cho cả `ref_image` và `target_image`.
      * Điều này "dạy" cho mô hình tính "bất biến" (invariance) - nhận ra vật thể dù nó thay đổi.

> ✅ *Cập nhật 2025-11-13:* Đã tạo `rfdetr/datasets/oneshot.py` với lớp `OneShotDataset` sinh tuple `(ref_image, target_image, ground_truth_dict)` cùng pipeline augment độc lập cho hai nhánh; `TrainConfig` và `utils.collate_fn` đã được mở rộng để hỗ trợ chế độ `dataset_file="oneshot"`.

### Bước 2: Chỉnh sửa Mô hình (Model)

Bạn cần tạo một file mô hình mới (ví dụ: `rfdetr/models/siamese_detr.py`) và định nghĩa lớp `SiameseDETR` của mình.

> ✅ *Cập nhật 2025-11-13:* Đã bổ sung `rfdetr/models/siamese_detr.py` cùng cờ cấu hình `use_siamese`; `build_model` tự động gói `LWDETR` thành `SiameseDETR` khi bật chế độ này.

```python
# Trong file rfdetr/models/siamese_detr.py (TỰ TẠO)

import torch
import torch.nn as nn
from .detr import RFDETR # Giả sử đây là lớp RF-DETR gốc

class SiameseDETR(nn.Module):
    def __init__(self, detr_model: RFDETR):
        super().__init__()
        
        # 1. Kế thừa các thành phần từ RF-DETR
        # Chúng ta sẽ chia sẻ trọng số (weight-sharing)
        self.backbone = detr_model.backbone
        self.encoder = detr_model.encoder
        self.decoder = detr_model.decoder
        self.regression_head = detr_model.bbox_embed # Giả sử tên là bbox_embed
        
        # 2. VỨT BỎ classification head
        # del detr_model.class_embed
        
        # 3. Thêm một lớp Pooling cho nhánh tham khảo
        self.ref_pool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling

    def forward(self, ref_image: torch.Tensor, target_image: torch.Tensor):
        """
        Quy trình "Siamese"
        """
        
        # == Nhánh Tham khảo (Reference Branch) ==
        # 1. Chạy qua Backbone + Encoder
        ref_features_map = self.encoder(self.backbone(ref_image)["features"][-1])
        # 2. Pooling để lấy "dấu vân tay"
        #    Kích thước từ [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        v_ref = self.ref_pool(ref_features_map).squeeze(-1).squeeze(-1)
        
        # == Nhánh Mục tiêu (Target Branch) ==
        # 1. Chạy qua Backbone + Encoder
        target_features_map = self.encoder(self.backbone(target_image)["features"][-1])
        # 2. Chạy qua Decoder để lấy các "ứng viên"
        #    Lưu ý: Bạn cần xem code gốc của RF-DETR để gọi hàm này chính xác
        decoder_output = self.decoder(target_features_map, ...) 
        
        # 3. Lấy Object Embeddings (các rổ đầy) từ Decoder
        #    Kích thước [B, Num_Queries, Hidden_Dim]
        object_embeddings = decoder_output.last_hidden_state 
        
        # 4. Lấy Bounding Boxes dự đoán
        #    Kích thước [B, Num_Queries, 4]
        predicted_bboxes = self.regression_head(object_embeddings)
        
        # == Trả về kết quả ==
        # Chúng ta không dùng softmax vì đây không phải là phân loại
        outputs = {
            "v_ref": v_ref, # Dấu vân tay [B, C]
            "object_embeddings": object_embeddings, # Ứng viên [B, Num_Queries, C]
            "predicted_bboxes": predicted_bboxes # Bbox dự đoán [B, Num_Queries, 4]
        }
        return outputs
```

### Bước 3: Chỉnh sửa Bộ ghép cặp (Matcher)

Đây là **phần khó nhất**. Bạn phải "dạy" cho `HungarianMatcher` cách ghép cặp dựa trên "so khớp" thay vì "phân loại".

  * **Vị trí file (dự đoán):** `rfdetr/models/matcher.py`
  * **Logic cũ (Cần sửa):** `cost_class = out_prob.unsqueeze(1) ...`
  * **Logic Mới (Cần thêm):**

<!-- end list -->

```python
# Trong file rfdetr/models/matcher.py (SỬA LẠI)

class HungarianMatcher(nn.Module):
    
    def forward(self, outputs, targets):
        """
        outputs: Đầu ra từ mô hình SiameseDETR (chứa "v_ref", "object_embeddings", "predicted_bboxes")
        targets: Ground truth (chứa "boxes", "labels")
        """
        
        # Lấy các giá trị cần thiết
        v_ref = outputs["v_ref"] # [B, C]
        obj_embeds = outputs["object_embeddings"] # [B, Num_Queries, C]
        pred_bboxes = outputs["predicted_bboxes"] # [B, Num_Queries, 4]
        
        # ... (Lấy gt_boxes và gt_labels từ targets) ...
        
        # == Tính toán Cost Matrix ==
        
        # 1. Cost Bbox (Giữ nguyên)
        #    Dùng L1 và GIU loss
        cost_bbox = torch.cdist(pred_bboxes.flatten(0, 1), gt_boxes.flatten(0, 1), p=1)
        cost_giou = -generalized_box_iou(...)
        
        # 2. Cost "So khớp" (THAY THẾ CHO cost_class)
        #    Chúng ta dùng 1 - Cosine Similarity
        #    Mục tiêu: Kéo similarity của cặp ĐÚNG về 1 (cost = 0)
        
        # Chuẩn hóa L2 (cần thiết cho Cosine Similarity ổn định)
        v_ref_norm = F.normalize(v_ref, p=2, dim=1)
        obj_embeds_norm = F.normalize(obj_embeds, p=2, dim=2)
        
        # Tính ma trận similarity: [B, Num_Queries, Num_GT]
        sim_matrix = torch.einsum('bc,bqc->bq', v_ref_norm, obj_embeds_norm)
        
        # Ở đây chúng ta giả định gt_label = 0 là vật thể cần tìm
        # Tạo cost matrix: 
        # Nếu là vật thể cần tìm (label=0), cost = 1 - sim
        # Nếu là vật thể khác (label=1), cost = sim (phạt nếu giống)
        cost_match = torch.where(gt_labels == 0, 1 - sim_matrix, sim_matrix)
        
        # == Tổng Cost ==
        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_match * cost_match
        
        # Chạy Hungarian algorithm
        indices = linear_sum_assignment(C.cpu())
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

```

*Lưu ý: Logic trên là logic đơn giản hóa. Bạn có thể cần một `Matcher` phức tạp hơn, ví dụ như chỉ so khớp `v_ref` với `gt_label == 0` và dùng `cost_bbox` cho các vật thể còn lại.*

### Bước 4: Chỉnh sửa Hàm Mất mát (Criterion)

  * **Vị trí file (dự đoán):** `rfdetr/models/criterion.py`
  * **Logic:**
    1.  Gọi `Matcher` (đã sửa ở Bước 3) để lấy các cặp (indices).
    2.  **BỎ:** `loss_ce` (Cross-Entropy loss).
    3.  **GIỮ:** `loss_bbox` và `loss_giou` (chỉ tính cho các cặp được ghép).
    4.  **THÊM:** `loss_match` (Contrastive Loss).

<!-- end list -->

```python
# Trong file rfdetr/models/criterion.py (SỬA LẠI)

class SetCriterion(nn.Module):
    
    def __init__(self, ..., matcher, ...):
        super().__init__()
        self.matcher = matcher
        # Thêm margin cho contrastive loss
        self.margin = 0.5 
        
    def loss_match(self, outputs, targets, indices):
        """
        Tính Contrastive Loss
        """
        v_ref = F.normalize(outputs["v_ref"], p=2, dim=1) # [B, C]
        obj_embeds = F.normalize(outputs["object_embeddings"], p=2, dim=2) # [B, Num_Queries, C]

        loss_total = 0
        
        # Lặp qua từng batch
        for i, (idx_pred, idx_gt) in enumerate(indices):
            
            # 1. Lấy các cặp Positive (đã được Matcher ghép)
            positive_embeds = obj_embeds[i, idx_pred]
            positive_sim = torch.einsum('c,qc->q', v_ref[i], positive_embeds)
            loss_positive = (1 - positive_sim).pow(2).mean() # Kéo về 1

            # 2. Lấy các cặp Negative (không được ghép)
            mask_negative = torch.ones(obj_embeds.shape[1], dtype=torch.bool)
            mask_negative[idx_pred] = False
            negative_embeds = obj_embeds[i, mask_negative]
            
            negative_sim = torch.einsum('c,qc->q', v_ref[i], negative_embeds)
            
            # Contrastive loss: max(0, sim - margin)
            loss_negative = F.relu(negative_sim - self.margin).pow(2).mean() # Đẩy ra xa (dưới margin)
            
            loss_total += loss_positive + loss_negative
            
        return loss_total / len(indices)

    def forward(self, outputs, targets):
        # 1. Chạy Matcher để lấy cặp
        indices = self.matcher(outputs, targets)
        
        # 2. Tính các loss
        loss_bbox = self.loss_boxes(outputs, targets, indices, num_boxes)
        loss_giou = self.loss_giou(outputs, targets, indices, num_boxes)
        loss_match = self.loss_match(outputs, targets, indices)
        
        losses = {
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
            'loss_match': loss_match,
        }
        return losses
```

### Bước 5: Huấn luyện và Suy luận

  * **Huấn luyện:**
      * Chạy vòng lặp huấn luyện bình thường. `DataLoader` của bạn (Bước 1) sẽ cung cấp `(ref, target, gts)`.
      * Mô hình (Bước 2) sẽ xử lý.
      * Hàm loss (Bước 4) sẽ tính toán và backpropagate.
  * **Suy luận (Inference):**
    1.  Chạy `ref_image` qua `Backbone + Encoder + Pool` để lấy `v_ref` (chỉ chạy 1 lần).
    2.  Chạy `target_image` qua `Backbone + Encoder + Decoder` để lấy `object_embeddings` và `predicted_bboxes`.
    3.  Tính `similarity_scores = cosine_similarity(v_ref, object_embeddings)`.
    4.  Lọc kết quả: Giữ lại các `(bbox, score)` nào có `score > threshold` (ví dụ: 0.8).
    5.  Bạn **không cần NMS**, vì `Decoder` đã xử lý việc này.

-----

## 4\. ⚠️ Rủi ro và Thách thức

1.  **Độ khó Kỹ thuật:** Đây là một "ca phẫu thuật" phức tạp. Bạn phải hiểu sâu về code của RF-DETR, đặc biệt là `Matcher` và `Criterion`.
2.  **Cân bằng Loss:** Bạn sẽ có 3 hàm loss (`bbox`, `giou`, `match`). Việc tìm ra các hệ số `lambda` (ví dụ: `loss_total = 1*loss_match + 5*loss_bbox + 2*loss_giou`) để chúng hội tụ đúng cách là rất khó và tốn thời gian.
3.  **Thiết kế Matcher:** Logic `Matcher` ở trên là đơn giản hóa. Một `Matcher` thực tế cho OSOD có thể cần phức tạp hơn để xử lý trường hợp nhiều vật thể tham khảo, hoặc không có vật thể nào.