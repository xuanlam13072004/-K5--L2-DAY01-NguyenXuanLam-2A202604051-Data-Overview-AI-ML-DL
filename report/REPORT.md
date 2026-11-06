# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
    "class_id": 468,
    "class_name": "cab",
    "rank": 1,
    "score": 0.510915
    "taxonomy_name": "ImageNet-1K",
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    Record này nói về cả bước ảnh, vì kết quả không có tọa độ về các vật thể trong ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
    Danh sách tên lớp đi kèm taxonomy ImageNet-1K của model yolo11n-cls.pt.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    Cần class_id vì để phân loại trong trường hợp tên lớp bị trùng hoặc viết khác, còn class_name thì dễ đọc hơn còn taxonomy_name thì để biết tên lớp đó thuộc taxonomy nào vì ví dụ như cùng 1 "class_name": "cab" có thể có nhiều taxonomy khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
    Guideline cần nói rõ quy định cách chọn một lớp. Ví dụ: chọn chủ thể chiếm diện tích lớn nhất hoặc ở trung tâm ảnh.
6. Vì sao không được lấy `score` của model làm ground truth?
    Vì đó chỉ là mức độ tin tưởng của model dựa trên taxonomy để dự đoán nên không thể lấy score của model làm ground truth

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
    "class_name": "bowl",
    "score": 0.499948,
    "bbox_xyxy": [
      59.29,
      288.92,
      133.46,
      328.98
    ],
    "bbox_width": 74.18,
    "bbox_height": 40.06
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
    Box dùng định dạng xyxy theo pixel, gốc tọa độ (0, 0) ở góc trên bên trái ảnh, x tăng sang phải, y tăng xuống dưới. Góc trên-trái ở điểm (59.29, 288.92): cách mép trái ảnh khoảng 59 px và cách mép trên khoảng 289 px, góc dưới-phải ở điểm (133.46, 328.98) cách mép trái khoảng 133 px và cách mép trên khoảng 329 px. Chiều rộng = 133.46 − 59.29 ≈ 74 px; chiều cao = 328.98 − 288.92 ≈ 40 px.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
    Đối với ngưỡng 0.20 có 17 prediction và với ngưỡng 0.60 là 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    khi hạ ngưỡng tức là có thêm dữ liệu ít chắc chắn reviewer cần phải kiểm tra lại các kết quả ví dụ như tên đúng không, box có vừa khít không, có box trùng không hoặc có box thừa không.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    Box phải chạm vào điểm trái nhất, phải nhất, trên nhất và dưới nhất của phần vật thể nhìn thấy được. Box bao trọn cả các bộ phận gắn liền như quai cốc hay cán thìa, không chừa khoảng nền thừa và không cắt mất phần nào của vật. Nếu vật bị che, chỉ vẽ phần còn nhìn thấy, không đoán phần bị che. Nếu vật bị cắt ở mép ảnh, box dừng tại mép ảnh. Mỗi vật thể có đúng một box riêng; các vật cùng lớp đặt cạnh nhau phải vẽ thành từng box riêng, không gộp chung.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    Em chọn cây lăn bột nó bị bột che mất phần giữa và sẽ lên hỏi Lab Coach, nếu theo guideline thì 2 đầu của câu lăn bột có thể bị xác định thành 2 vật thể.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
    "instance_id": "kitchen-001",
    "class_name": "person",
    "score": 0.899318,
    "polygon_point_count": 348,
    "polygon_xy": [
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    So với box, polygon cho biết chính xác hơn hình dạng của vật thể chứ không chỉ là ô vuông bao quanh vật.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    instance_id dùng để phân biệt các vật thể được xác định trong ảnh, nó không phải là class ID vì class ID cho biết vật thuộc loại gì, nó không phải tracking ID vì tracking ID dùng trong video để giữ cùng một id cho cùng một vật qua nhiều khung hình.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    Đường viền mask phải đi sát ranh giới giữa vật và nền, chỉ bao những pixel nhìn thấy được thuộc về vật. Không lấn ra nền, không đoán phần bị che. Khoét bỏ phần nền nằm giữa vật như khe giữa hai chân, khoảng trống giữa tay và thân, lỗ quai cốc. Bộ phận gắn liền (tóc, quần áo, giày với người; quai với cốc) tính vào mask của vật đó. Vật đang cầm hoặc vật chạm nhau phải tách thành mask riêng; mỗi pixel chỉ thuộc một instance.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    Em chọn cây lăn bột nó bị bột che mất phần giữa và sẽ lên hỏi Lab Coach, nếu theo guideline thì 2 đầu của câu lăn bột không có liên kết với nhau nên rts có thể nó sẽ được xác định thành 2 vật thể.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh (class_id, class_name, ImageNet-1K) | traffic: top-5 đều là xe (cab 0.51, minibus 0.16…), model phân vân; detection lại thấy bus 0.91 | Chọn một lớp cho chủ thể chính theo guideline, không chép rank 1 | Nhãn đúng chủ thể chính chưa, các ảnh tương tự có gán thống nhất không |
| Phát hiện vật thể | Mỗi vật một box bbox_xyxy (pixel) + lớp COCO-80 | kitchen: ngưỡng 0.20 có 17 box, 0.60 còn 6; bowl score 0.50 sát ngưỡng; 5 bowl, 2 oven có thể trùng | Vẽ box ôm sát phần nhìn thấy, mỗi vật một box, gán cả vật model bỏ sót | Box lệch hoặc trùng, sai lớp, vật bị bỏ sót |
| Instance segmentation | Mỗi vật một polygon + lớp + instance_id riêng | kitchen-001 (person 0.90): polygon chỉ ~57% diện tích box; nồi dưới gầm bàn bị che phần lớn, COCO-80 không có lớp “nồi” | Vẽ mask sát biên, khoét nền, mỗi vật một instance; ca che nhiều thì escalation | Mask lấn nền không, mask có chồng nhau không, ca escalation đã xử lý chưa |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
    Chỉ dùng ba ảnh COCO công khai mà notebook đã cố định và kiểm tra checksum, Tôi không tải lên Colab hay GitHub bất kỳ ảnh cá nhân nào, và không ghi họ tên, MSSV, email hay số điện thoại vào REPORT.md hoặc output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
    Dừng ngay, không chạy tiếp, không tải lên và không commit dữ liệu đó, rồi báo cho Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
