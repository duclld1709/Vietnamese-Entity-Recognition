# Câu hỏi phỏng vấn: Vietnamese NER & Personal-Data Anonymiser

Bộ 15 câu hỏi một nhà tuyển dụng (NLP/AI Engineer, Data Scientist, Backend ML) có thể hỏi về dự án
này, kèm gợi ý trả lời bám theo code thật trong repo. Mỗi câu có ba phần: người phỏng vấn muốn kiểm
tra điều gì, gợi ý trả lời, và câu hỏi nối tiếp nên chuẩn bị.

> Mẹo chung: dự án mạnh ở ba điểm. (1) Có **use case kinh doanh rõ** (Nghị định 13/2023, gửi văn bản
> cho LLM an toàn). (2) **Kết hợp model với rule** một cách có nguyên tắc. (3) Bạn **tự phát hiện
> metric của v1 bị thổi phồng**. Hãy dẫn câu chuyện về ba điểm này.

---

## Q1. Giới thiệu dự án trong 2 phút.

**Muốn kiểm tra:** khả năng tóm tắt, gắn kỹ thuật với giá trị kinh doanh.

**Gợi ý trả lời:**
- **Bài toán:** doanh nghiệp Việt Nam muốn dùng LLM (ChatGPT, Claude…) cho hợp đồng, hồ sơ khách hàng,
  nhưng Nghị định 13/2023/NĐ-CP buộc họ chịu trách nhiệm khi chuyển dữ liệu cá nhân cho bên thứ ba.
  Các công cụ có sẵn như Presidio hay cloud DLP làm tốt với tên tiếng Anh và định dạng Mỹ, nhưng bỏ sót
  "Nguyễn Văn Nam", `0912.345.678`, số CCCD 12 chữ số.
- **Giải pháp lai:**
  - Model **PhoBERT + Linear-CRF** tìm PER/ORG/LOC (dữ liệu VLSP 2016).
  - **Rule có kiểm tra hợp lệ** cho dữ liệu có cấu trúc: SĐT, CCCD/CMND, email, STK, MST, thẻ (Luhn),
    biển số, ngày tháng, URL (`vietner/pii.py`).
- **Ẩn danh có thể đảo ngược:** thay thực thể bằng `[PER_1]`, gửi cho LLM, rồi `deanonymize` câu trả
  lời để đặt lại tên thật.
- **Sản phẩm:** thư viện, CLI, FastAPI, demo Streamlit trên Hugging Face Spaces, Docker, CI với 68 test.

---

## Q2. Tại sao kết hợp model với rule thay vì chỉ dùng một trong hai?

**Muốn kiểm tra:** tư duy chọn công cụ đúng cho từng phần bài toán.

**Gợi ý trả lời:**
- **Tên người, tổ chức, địa danh** không có định dạng cố định và phụ thuộc ngữ cảnh ("Hà Nội" có thể
  là LOC hay một phần của ORG), nên cần model học từ dữ liệu.
- **SĐT, CCCD, thẻ** có định dạng cố định và có thể **kiểm tra hợp lệ**: đầu số di động 03/05/07/08/09,
  mã tỉnh ở 3 chữ số đầu của CCCD (Thông tư 07/2016), checksum Luhn cho thẻ, ngày và tháng hợp lệ.
  Rule ở đây chính xác hơn model thống kê và không cần dữ liệu gán nhãn (VLSP cũng không có nhãn này).
- **Chuỗi số mơ hồ** (STK, MST, CMND 9 số) chỉ được báo khi có **từ khoá ngữ cảnh** trong 40 ký tự phía
  trước (`stk`, `mst`, `cmnd`…, tham số `context_required=True`). Nếu không, một dãy 10 số bất kỳ sẽ bị
  gắn nhầm. Có ngữ cảnh còn nâng score lên 0.99.

**Nối tiếp:** "Từ khoá ở xa hơn 40 ký tự thì sao?" Sẽ bỏ sót (recall giảm). Đó là trade-off chủ động
để giữ precision; có thể tinh chỉnh bằng một tập test thực tế.

---

## Q3. Khi model và rule cùng bắt một đoạn chồng nhau, chọn cái nào?

**Muốn kiểm tra:** chi tiết thiết kế, thuật toán đơn giản nhưng đúng.

**Gợi ý trả lời (`resolve_overlaps` trong `pii.py`):**
- Sắp xếp các ứng viên theo: **score cao hơn → span dài hơn → ưu tiên rule → vị trí**, sau đó chọn
  tham lam các span không chồng lấn.
- Score: rule đã kiểm tra hợp lệ là 0.85–0.99; model là hằng số `MODEL_SCORE = 0.8`, vì CRF không cho
  xác suất đã calibrate. Vì vậy rule hợp lệ luôn thắng model khi chồng nhau.
- Ví dụ: model gắn "0912345678" là một phần của ORG, còn rule PHONE hợp lệ (0.9). PHONE thắng.

**Nối tiếp:** "Sao không lấy marginal probability từ CRF?" Có thể tính bằng forward-backward, nhưng
v1 không có, và cần calibrate. Đây là hướng cải tiến hợp lý.

---

## Q4. Làm sao trả về đúng vị trí ký tự (offset) khi PhoBERT cần văn bản đã tách từ?

**Muốn kiểm tra:** xử lý tiếng Việt thực tế, cẩn thận với chuỗi.

**Gợi ý trả lời (`vietner/text.py`):**
- PhoBERT cần đầu vào đã tách từ (`Hà_Nội`). Tôi tách bằng pyvi, nhưng ẩn danh cần thay thế **tại
  chỗ** trong văn bản gốc, nên phải biết `[start, end)` của từng từ.
- `align()` dựa trên giả định pyvi chỉ **chèn khoảng trắng** và **nối âm tiết bằng `_`**. Với mỗi
  token, tách theo `_` rồi tìm lần lượt từng âm tiết trong văn bản gốc từ con trỏ hiện tại, và chỉ cho
  phép khoảng trắng nằm giữa. Nếu giả định bị phá vỡ thì trả `None`, và `segment()` **fallback về tách
  theo khoảng trắng** (kèm log warning) thay vì crash.
- Trường hợp đặc biệt: nếu văn bản gốc thực sự chứa dấu `_` thì giữ nguyên token.
- **Chuẩn hoá NFC** trước mọi xử lý: văn bản từ macOS hoặc PDF hay ở dạng NFD (dấu tách rời), vốn
  không có trong vocab của PhoBERT.
- Văn bản được tách câu (`split_sentences`) trước khi tách từ, và offset của câu được cộng dồn.

---

## Q5. Biểu diễn mỗi từ thế nào khi PhoBERT tách từ thành nhiều subword (BPE)?

**Muốn kiểm tra:** hiểu tokenizer, kỹ năng PyTorch.

**Gợi ý trả lời (`vietner/encoder.py`):**
- Mỗi từ được BPE-tokenize **riêng** (tokenizer của PhoBERT tách theo khoảng trắng, nên kết quả giống
  khi tokenize cả câu), có `lru_cache` 200k từ. Cách này **đảm bảo đúng một vector cho mỗi từ**.
- `batch_inputs` dựng `input_ids`, `attention_mask` và `word_index` (subword thuộc từ nào; −1 cho
  `<s>`, `</s>`, padding).
- `mean_pool` dùng `index_add_` để cộng hidden state của các subword cùng từ, chia cho số subword.
  Slot 0 gom các vị trí bị bỏ qua. Toàn bộ là vector hoá, không có vòng lặp Python theo từ.
- So với v1: v1 tokenize cả câu rồi gom lại theo hậu tố `@@`. Gặp từ ngoài vocab (ra `<unk>`, mất
  `@@`) thì số vector lệch số nhãn, và v1 **âm thầm bỏ cả câu** (mất dữ liệu train), còn lúc inference
  thì `zip` cắt cụt kết quả.

---

## Q6. Văn bản dài hơn giới hạn 256 token của PhoBERT thì xử lý thế nào?

**Muốn kiểm tra:** hiểu giới hạn của transformer, xử lý trong production.

**Gợi ý trả lời:**
- PhoBERT có 258 vị trí, tức 256 token cộng `<s>` và `</s>`. v1 crash khi gặp câu dài.
- v2 xử lý hai tầng:
  1. Tách văn bản thành câu (`split_sentences`, dựa trên dấu câu kết thúc hoặc xuống dòng).
  2. `WordEncoder.chunk` chia chuỗi từ thành các đoạn liên tiếp **tối đa 254 subword**, cắt **tại ranh
     giới từ** (không cắt giữa từ). Một từ đơn lẻ dài hơn giới hạn thì bị cắt lấy các subword đầu.
- `encode()` gom các chunk của mọi đầu vào, **sắp theo độ dài** (length bucketing) để mỗi batch ít
  padding, rồi ghép lại đúng thứ tự.
- Lúc train, `chunk_for_encoder` cũng cắt câu dài theo cách tương tự (cắt cả nhãn).
- Giới hạn API: `MAX_CHARS = 20_000` ký tự mỗi văn bản, tối đa 64 văn bản mỗi request.

**Nối tiếp:** "Cắt chunk có làm mất ngữ cảnh ở ranh giới không?" Có, một thực thể nằm ngay ranh giới
có thể bị tách. Có thể dùng sliding window có overlap, nhưng chunk 254 subword hiếm khi bị cắt trong
câu bình thường.

---

## Q7. Tại sao dùng CRF thay vì softmax trên từng token?

**Muốn kiểm tra:** nền tảng sequence labeling.

**Gợi ý trả lời:**
- Softmax quyết định nhãn **độc lập** cho từng token, nên có thể sinh chuỗi vô lý như `O → I-PER`
  hay `B-PER → I-LOC`.
- CRF học **ma trận chuyển nhãn** (`transitions`, `start_transitions`, `end_transitions`) và giải mã
  **chuỗi nhãn tốt nhất toàn cục** bằng Viterbi, nên ranh giới thực thể nhất quán hơn.
- Train bằng negative log-likelihood: điểm của chuỗi vàng trừ log-partition, tính bằng **thuật toán
  forward** với `logsumexp`, độ phức tạp O(T·K²).
- CRF được **vendor** từ pytorch-crf 0.7.2 (MIT, thư viện không còn được bảo trì). Thay đổi duy nhất
  là chuyển mask sang `bool` để chạy với PyTorch mới (`torch.where` không nhận `uint8`).
- Repo có notebook `Softmax_PhoBERT.ipynb` của v1 để so sánh.

---

## Q8. Chế độ `probe` và `finetune` khác nhau thế nào? Khi nào dùng cái nào?

**Muốn kiểm tra:** hiểu trade-off transfer learning, chi phí tính toán.

**Gợi ý trả lời (`vietner/models.py`, `training/train.py`):**
- **`probe`** (công thức v1): PhoBERT **đóng băng**, chỉ train Linear-CRF head.
  - Rất nhanh: `train_probe` **encode mọi câu một lần**, cache đặc trưng, rồi train head nhiều epoch
    trên tensor có sẵn. Mỗi epoch không phải chạy lại PhoBERT.
  - Checkpoint chỉ khoảng **24 KB** (chỉ lưu head; `save` bỏ trạng thái encoder khi ở chế độ probe).
- **`finetune`**: train toàn bộ, end-to-end.
  - AdamW với **hai learning rate**: encoder 3e-5, head 1e-3; linear warmup 10%; gradient clipping.
  - AMP dùng bf16 nếu GPU hỗ trợ (không cần GradScaler), ngược lại fp16 + GradScaler.
  - **Loss CRF tính ở float32** (`feats.float()`), vì `logsumexp` ở half precision dễ mất độ chính xác.
- Thường fine-tune cho F1 cao hơn nhưng tốn GPU và checkpoint nặng khoảng 500 MB. Probe phù hợp khi
  tài nguyên hạn chế hoặc cần đổi head nhanh.

---

## Q9. F1 = 0.91 của v1 có đúng không?

**Muốn kiểm tra:** hiểu metric NER, trung thực. **Đây là câu quan trọng nhất.**

**Gợi ý trả lời:**
- Con số đó là **macro F1 cấp token, có tính cả nhãn `O`**. Ba vấn đề:
  1. Khoảng **94% token là `O`** (51.036 / 54.492 trong log test), và `O` có F1 = 1.00. Lấy macro
     cộng một lớp dễ gần như hoàn hảo vào là thổi phồng con số.
  2. Metric cấp token **không kiểm tra ranh giới**: đoán đúng 2/3 token của "Nguyễn Văn Nam" vẫn được
     điểm từng phần, trong khi với ẩn danh thì bỏ sót "Nam" là lộ thông tin.
  3. Dashboard v1 còn **hard-code** một bảng báo cáo không khớp log train (lưu lại ở
     `reports/v1_reported.json`).
- v2 dùng **entity-level P/R/F1** chuẩn CoNLL/VLSP (`entity_report` trong `metrics.py`): thực thể chỉ
  được tính đúng khi **cả span lẫn nhãn khớp chính xác**. Kết quả được đối chiếu với `seqeval` trong
  test. Metric cũ vẫn được giữ (`token_report`) để so sánh.
- Kết quả entity-level sẽ công bố sau khi chạy evaluation; UI đọc trực tiếp từ `metrics.json`, không
  gõ tay.

---

## Q10. Bạn đảm bảo test set không bị rò rỉ thế nào, mà vẫn so sánh được với v1?

**Muốn kiểm tra:** data leakage, khả năng tái lập.

**Gợi ý trả lời (`training/data.py`):**
- **Tái tạo split của v1:** v1 bỏ các câu mà việc gom `@@` không khớp số nhãn, rồi gọi
  `train_test_split(random_state=42)` hai lần (70/15/15). Các câu bị bỏ **chỉ phụ thuộc tokenizer**,
  nên `v1_kept` mô phỏng lại đúng logic đó (kể cả round-trip qua id khiến subword OOV thành `<unk>`).
  Nhờ vậy `split_v1` dựng lại **chính xác** test set cũ mà không cần embedding gốc, và model cũ với
  model mới được chấm trên cùng các câu.
- **Khử trùng lặp:** VLSP có câu lặp y hệt. `remove_test_overlap` xoá khỏi train/val mọi câu có chuỗi
  token trùng với test (và câu train trùng val). Số câu bị xoá được ghi vào `metrics.json`.

**Nối tiếp:** "Sao không dùng split mới hoàn toàn sạch?" Có hỗ trợ (`--split random`), nhưng mục tiêu
là so sánh công bằng với checkpoint đã công bố.

---

## Q11. Phân tích lỗi (error analysis) bạn làm thế nào?

**Muốn kiểm tra:** không dừng ở một con số F1.

**Gợi ý trả lời (`error_breakdown` trong `metrics.py`):**
- Mỗi lỗi được phân loại:
  - `wrong_type`: span đúng, nhãn sai (ví dụ ORG thành LOC).
  - `wrong_boundary`: span chồng lấn, cùng nhãn (thừa hoặc thiếu từ).
  - `wrong_both`: chồng lấn, khác nhãn.
  - `missed`: thực thể vàng không có dự đoán nào chồng lấn.
  - `spurious`: dự đoán không chồng lấn thực thể vàng nào.
- Mọi câu sai được ghi vào `errors.jsonl` (văn bản, span vàng, span dự đoán) để đọc tay.
- Ý nghĩa với sản phẩm ẩn danh: `missed` và `wrong_boundary` thiếu từ là **nguy hiểm** (lộ dữ liệu),
  còn `wrong_type` gần như vô hại (vẫn bị che, chỉ sai nhãn placeholder).
- Theo log v1, ORG yếu nhất (B-ORG F1 0.82): tên tổ chức dài và hay chứa địa danh.

---

## Q12. Ba chiến lược ẩn danh hoạt động thế nào? Làm sao đảo ngược được?

**Muốn kiểm tra:** thiết kế API hướng người dùng, edge case.

**Gợi ý trả lời (`vietner/anonymize.py`):**
- **`tag`:** placeholder `[PER_1]`, **nhất quán**: cùng một thực thể (khoá `(label, casefold + chuẩn hoá
  khoảng trắng)`) luôn ra cùng một tag. "Nguyễn Văn Nam" xuất hiện 3 lần đều thành `[PER_1]`, giúp LLM
  hiểu đó là một người. Trả về `mapping` để đảo ngược.
- **`mask`:** giữ văn bản đọc được. SĐT giữ 2 số đầu (nhà mạng) và 2 số cuối, CCCD và STK giữ 3 số cuối,
  thẻ giữ 4 số cuối, email giữ ký tự đầu và domain, tên giữ chữ cái đầu mỗi từ.
- **`redact`:** `[PER]`, không đảo ngược được.
- **`deanonymize`:** regex ghép mọi khoá, **sắp theo độ dài giảm dần**, rồi thay một lượt.
- Edge case: nếu văn bản đầu vào **đã chứa** dạng `[LABEL_N]` thì trả warning, vì đảo ngược có thể nhập
  nhằng.
- Mặc định không che MISC (quốc tịch, sự kiện), vì hiếm khi là dữ liệu cá nhân.

**Nối tiếp:** "Mapping có phải dữ liệu nhạy cảm không?" Có. Nó chứa toàn bộ dữ liệu gốc, nên phải lưu ở
phía client hoặc server nội bộ, không bao giờ gửi kèm cho LLM.

---

## Q13. Nếu không tải được PhoBERT thì service xử lý thế nào?

**Muốn kiểm tra:** resilience, graceful degradation.

**Gợi ý trả lời (`vietner/pipeline.py`, `vietner/api.py`):**
- `VietNER.from_checkpoint(strict=False)` bắt exception khi nạp model và trả về instance **chỉ chạy
  rule** (`tagger=None`), lưu `unavailable_reason`.
- `/health` trả `"degraded"` thay vì `"ok"`, và **mọi response kèm warning** rằng chỉ có thực thể từ
  rule. Người dùng không bị đánh lừa rằng tên người đã được che.
- Model được nạp **một lần** trong `lifespan` của FastAPI. v1 nạp lại PhoBERT và head **mỗi lần dự
  đoán**, mất vài giây cho mỗi request.
- Docker image bake sẵn PhoBERT để khởi động không phụ thuộc mạng.

**Nối tiếp:** "Với bài toán bảo mật, degrade có an toàn không?" Câu hỏi hay. Với pipeline bắt buộc
che tên, nên cấu hình `strict=True` để fail closed (từ chối xử lý) thay vì degrade âm thầm. Hiện là
lựa chọn của người triển khai.

---

## Q14. Bạn phát hiện lỗi bảo mật gì trong demo v1?

**Muốn kiểm tra:** ý thức bảo mật, kể cả ở phần demo.

**Gợi ý trả lời:**
- v1 render văn bản người dùng vào HTML bằng `unsafe_allow_html` **không escape**, nên người dùng có
  thể chèn `<script>` hoặc HTML tuỳ ý (XSS, HTML injection) vào demo.
- v2: `render_html` (`vietner/render.py`) gọi `html.escape` cho **mọi đoạn văn bản người dùng** và cả
  nhãn. Có test riêng.
- Các vấn đề vận hành khác của v1: deploy lên Space ở mỗi lần push mà không có test, commit file
  `.pyc`, README mô tả file không tồn tại. v2: CI (ruff + 68 test) chặn deploy khi test fail.

---

## Q15. Hạn chế lớn nhất? Nếu có thêm thời gian bạn làm gì?

**Muốn kiểm tra:** tự nhận thức, hiểu khoảng cách giữa benchmark và thực tế.

**Gợi ý trả lời:**
1. **Lệch domain:** VLSP là văn bản báo chí, còn use case thật là hợp đồng, CV, chat log. Cần một
   **test set ngoài domain** tự gán nhãn.
2. **Địa chỉ:** nhãn LOC của VLSP không bao gồm số nhà, đường, phường, nên địa chỉ chi tiết bị lọt. Cần
   recognizer địa chỉ.
3. **Đồng tham chiếu:** "ông Nam" và "Nguyễn Văn Nam" hiện ra hai placeholder khác nhau. Cần gộp theo
   coreference.
4. **Công bố kết quả entity-level:** probe vs fine-tune vs LLM zero-shot (F1, latency, chi phí trên
   1000 câu).
5. **Serving trên CPU:** export ONNX và quantize.
6. **Score của model là hằng số 0.8:** dùng marginal của CRF để có confidence thực.

---

## Câu hỏi nhanh nên thuộc

| Câu hỏi | Trả lời ngắn |
|---|---|
| Bộ nhãn? | Legacy 7 tag (MISC → O, như v1); scheme `vlsp` 9 tag giữ MISC |
| Giải mã BIO? | Lenient kiểu conlleval: `I-X` không nối tiếp `X` thì mở thực thể mới, không bị bỏ (`bio_to_spans`) |
| Word segmenter? | pyvi (`ViTokenizer`), có fallback tách theo khoảng trắng |
| Validator CCCD? | 12 số, bắt đầu bằng 0, 3 số đầu là mã tỉnh hợp lệ |
| Validator thẻ? | Thuật toán Luhn, 13–19 chữ số |
| Định dạng checkpoint? | `vietner/v2`: config + head + encoder (chỉ khi finetune) + metadata; vẫn đọc được state dict thô v1 |
| Test không tải PhoBERT? | Tokenizer và encoder giả kích thước nhỏ trong `tests/conftest.py`, mô phỏng định dạng `@@` và `<unk>` |
