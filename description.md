Kế hoạch “reference-conditioned queries” — trỏ đúng chỗ cần sửa trong repo và cách vá tối thiểu.

# Điểm móc trong repo

* `rfdetr/detr.py`

  * `RFDETR.get_model(config)` tạo `Model(**config.dict())`. Đây là lối vào để thêm tham số cấu hình mới (ví dụ `ref_conditioning`, `use_box_only`) và để bạn truyền ref-image vào train/predict. ([Roboflow][1])
  * `RFDETR.predict(...)`: vòng tiền xử lý ảnh, nơi bạn có thể mở rộng API để nhận thêm `ref_images` đi kèm mỗi frame. ([Roboflow][1])
* Phần “đầu ra phân loại” nằm trong mô hình con (head) — upstream issues chỉ rõ đầu phân loại là `class_embed` trong `rfdetr/models/lwdetr.py`. Bạn sẽ không đụng vào decoder cốt lõi, chỉ thay cách khởi tạo query và có thể bypass `class_embed`. ([GitHub][2])

# Sửa 1 — mở API để đưa reference vào pipeline

**Mục tiêu:** thêm ref vào `train/predict` và đẩy xuống mô hình.

1. Mở rộng API:

* Trong `rfdetr/detr.py`, đổi chữ ký:

  * `RFDETR.predict(images, ..., ref_images=None, **kwargs)` — giữ tương thích, nếu `ref_images is None` dùng queries học được như cũ. Bạn đặt ref theo từng ảnh: list cùng độ dài với `images`. ([Roboflow][1])
  * Tương tự trong `train(...)` → khi build `DataLoader`, batch trả về `(ref_img, img, targets)`.

2. Tiền xử lý:

* Ở loop chuẩn hoá trong `predict(...)`, hiện repo convert ảnh về `torch.Tensor` [0..1], shape `(C,H,W)`. Làm y hệt cho `ref_img`. Giữ nguyên kiểm tra kênh/chuẩn hoá. Đưa cả hai xuống `self.model.model` qua kwargs mới, ví dụ `self.model.model(ref_img=ref_tensor, img=img_tensor, ...)`. ([Roboflow][1])

# Sửa 2 — thêm “reference-conditioned queries” trong mô hình

**Mục tiêu:** thay cách tạo `query_embed` để phụ thuộc ref, không đụng vào backbone/encoder/decoder gốc.

Vị trí thay:

* File mô hình lõi: `rfdetr/models/lwdetr.py` (class LWDETR). Tại đây có:

  * `self.class_embed = nn.Linear(hidden_dim, num_classes+1)` (đầu phân loại) và `self.bbox_embed` (đầu hồi quy hộp).
  * `self.query_embed = nn.Embedding(num_queries, hidden_dim)` để khởi tạo queries.
  * Hàm `forward(...)` gọi backbone→encoder→decoder với `query_embed`. (Đường dẫn và biến danh định y như upstream issue mô tả). ([GitHub][2])

Các bước vá tối thiểu:

1. **Adapter cho reference**
   Thêm các module sau trong `__init__`:

```python
# giả sử có self.hidden_dim và self.backbone đã sẵn
self.ref_pool = torch.nn.AdaptiveAvgPool2d((1,1))
self.ref_proj = torch.nn.Sequential(
    torch.nn.Linear(self.backbone_out_channels, self.hidden_dim),
    torch.nn.LayerNorm(self.hidden_dim),
)
# tuỳ chọn: cross-attn nhẹ trước decoder để “truyền” ref vào queries
self.ref_cross_attn = torch.nn.MultiheadAttention(
    embed_dim=self.hidden_dim, num_heads=8, batch_first=True
)
```

2. **Tạo ref-queries**
   Trong `forward`, nhận thêm `ref_img: Tensor | None`:

```python
def forward(self, img, *, ref_img=None, targets=None, **kwargs):
    # features cho frame
    feats, pos, mask = self.backbone(img)  # y như trước

    # queries học sẵn (giữ để tương thích checkpoint)
    q_learned = self.query_embed.weight.unsqueeze(0).repeat(img.size(0), 1, 1)  # [B, Q, D]

    if ref_img is not None:
        # encode ref dùng backbone dùng chung để chia sẻ biểu diễn
        ref_feats, _, _ = self.backbone(ref_img)
        ref_map = ref_feats[-1]              # lấy stage cuối [B, C, H, W]
        ref_vec = self.ref_pool(ref_map).flatten(1)        # [B, C]
        q_ref = self.ref_proj(ref_vec).unsqueeze(1)        # [B, 1, D]
        # trộn: cho q_learned “ngửi” ref một lượt cross-attn
        q_cond, _ = self.ref_cross_attn(q_learned, q_ref, q_ref)  # [B, Q, D]
        query_embed = q_learned + q_cond
    else:
        query_embed = q_learned

    # gọi transformer decoder như cũ, chỉ thay query_embed truyền vào
    hs, references = self.transformer(
        feats, mask, pos, query_embed=query_embed
    )
    # heads: bbox như cũ, cls có thể giữ để tương thích nhưng không dùng trong loss
    outputs_classes = self.class_embed(hs)         # hoặc bỏ qua trong loss
    outputs_coords  = self.bbox_embed(hs).sigmoid()
    out = {"pred_logits": outputs_classes[:, -1], "pred_boxes": outputs_coords[:, -1]}
    return out
```

3. **Cấu hình mô hình**

* Thêm cờ cấu hình ở `ModelConfig` (đi qua `RFDETR.get_model(config)`):
  `ref_conditioning: Literal["none","gap_xattn"]="gap_xattn"`,
  `use_box_only: bool=True` để điều khiển loss sau này. Thêm kwargs này được chuyển nguyên vẹn vào `Model(**config.dict())`. ([Roboflow][1])

# Sửa 3 — vô hiệu hoá phân loại, giữ box-only

**Mục tiêu:** không cần nhãn lớp. Vẫn train hồi quy box.

* Không chạm vào kiến trúc decoder. Chỉ đổi loss/matcher để bỏ thành phần class.
* Trong huấn luyện, để tương thích checkpoint bạn có thể **giữ `class_embed` nhưng weight=0**.

Nơi áp dụng:

* Bộ huấn luyện/loss của RF-DETR được gọi bên trong `self.model.train(...)` từ `RFDETR.train_from_config`. Bạn không cần đổi call site; chỉ cần thêm tham số `use_box_only=True` vào `ModelConfig` và ở lớp `Model` (hoặc criterion bên trong nó) **bỏ tính `loss_ce`** và **khử `cost_class` trong Hungarian** (giữ `L1 + GIoU`). Upstream xác nhận head phân loại là `class_embed` — để lại nhưng không đóng góp loss. ([Roboflow][1])

Pseudo-patch cho criterion/matcher bên trong mô hình:

```python
if self.use_box_only:
    # matcher
    cost_bbox = lam_l1 * box_l1_cost(pred_boxes, tgt_boxes) + lam_giou * giou_cost(pred_boxes, tgt_boxes)
    indices = hungarian(cost_bbox)  # không có cost_class
    # losses
    loss_bbox = l1(pred_boxes[indices], tgt_boxes) + lam_giou*(1-giou(pred_boxes[indices], tgt_boxes))
    losses = {"loss_bbox": loss_bbox}
else:
    # nhánh cũ (cls + box)
```

# Sửa 4 — dữ liệu và target

* Dataloader phải trả `ref_img` cùng `img` và **targets chỉ chứa boxes** (không `labels`). Khi `use_box_only=True`, criterion của bạn **không đọc `labels`**.
* Nếu ref là crop “đúng vật thể”, bạn có thể khởi động bằng pseudo-label: `tgt_box = full_image_box` cho ảnh ref; với frame thật, dùng Hungarian trên score decoder để chọn box tốt nhất rồi tự-nhãn dần (self-training) — phần này nằm ở dataset/loop huấn luyện chứ không đòi sửa kiến trúc.

# Vì sao đây là “DETR-way” chứ không phá kiến trúc

* Bạn **không đụng** backbone, encoder, decoder; chỉ thay **nguồn queries** (`self.query_embed`) bằng phiên bản đã **điều kiện hoá ref**; vẫn dùng cùng call vào transformer. Đây là đúng triết lý DETR: queries chứa “ý định tìm gì”; ref cung cấp “ý định”.
* Các file/entry points cần đụng tới đều xác định: `rfdetr/detr.py` cho API & config → `Model(**config.dict())`; trong mô hình con (`rfdetr/models/lwdetr.py`) nơi có `class_embed`/`bbox_embed`/`query_embed` và `forward`. ([Roboflow][1])

# Gợi ý thực thi an toàn

* Giữ nguyên `class_embed` để load checkpoint, nhưng **đặt weight cls = 0** trong loss.
* Freeze backbone + encoder 5–10 epoch đầu để regression không “trôi”.
* Duy trì `num_queries` như cũ; không cần tăng khi thêm conditioning. (Có issue bàn về tăng num_queries → lỗi giới hạn; tránh đổi tuỳ khi chưa kiểm soát đầy đủ.) ([GitHub][3])

Kết quả: mô hình vẫn là DETR, inference vẫn đi qua `RFDETR.predict`, nhưng nay mỗi ảnh có thể kèm `ref_img`; decoder queries đã được “tiêm” ngữ nghĩa từ ref để hồi quy bbox **mà không cần nhãn lớp**.

[1]: https://roboflow.github.io/rf-detr/reference/rfdetr/ "RF-DETR - RF-DETR"
[2]: https://github.com/roboflow/rf-detr/issues/330?utm_source=chatgpt.com "train_from_config initializes one less class_embed neuron ..."
[3]: https://github.com/roboflow/rf-detr/issues/419?utm_source=chatgpt.com "Increasing object limit of 300 · Issue #419 · roboflow/rf-detr"
