# Vietnamese Named Entity Recognition

## 🛠️ Set Up Your Environment With Conda

### Option 1: Using `requirements.txt`

```bash
conda create --name vnner python=3.10
conda activate vnner
pip install -r requirements.txt
```

### Option 2: Using `environment.yml`

```bash
conda env create -f environment.yml
conda activate vnner
```

---

## 📂 Project Structure

```
my_ai_project/
│
├── data/                  
│   ├── raw/               # Dữ liệu gốc
│   ├── processed/         # Dữ liệu sau khi tiền xử lý
│   └── external/          # Dữ liệu từ nguồn bên ngoài (nếu có)
│
├── notebooks/                      # Thử nghiệm và khám phá dữ liệu
│   ├── Duc_Notebook.ipynb          # CRF + RandomForest
│   ├── Softmax_PhoBERT.ipynb       # Softmax
│
├── src/                   # Mã nguồn chính của dự án
│   ├── __init__.py
│   ├── data_loader.py     # Nạp và xử lý dữ liệu
│   ├── preprocessing.py   # Hàm tiền xử lý dữ liệu
│   ├── model.py           # Định nghĩa kiến trúc mô hình
│   ├── train.py           # Huấn luyện mô hình
│   ├── evaluate.py        # Đánh giá mô hình
│   └── predict.py         # Dự đoán với mô hình đã huấn luyện
│
├── models/                # Mô hình đã lưu sau khi huấn luyện
│   └── best_model.pth     # File trọng số mô hình
│
├── outputs/               # Kết quả, biểu đồ, log, metrics
│   ├── logs/              # Nhật ký huấn luyện (tensorboard/logging)
│   └── figures/           # Biểu đồ trực quan hóa
│
├── configs/               # File cấu hình cho mô hình, huấn luyện
│   └── config.yaml
│
├── tests/                 # Unit test cho các hàm chính
│
├── requirements.txt       # Thư viện cần cài đặt
├── environment.yml        # Môi trường Conda
├── README.md              # Giới thiệu dự án
└── run.py                 # Script chính để chạy toàn bộ pipeline
```

---

## 📚 Additional Resources (Optional)

If you have any questions about the project structure, consider reading these helpful articles first:

* [Understanding `__init__.py`](https://zetcode.com/python/init-file/)
* [Markdown Basic Syntax](https://www.markdownguide.org/basic-syntax/#escaping-characters)
* [Difference Between `requirements.txt` and `environment.yml`](https://www.reddit.com/r/learnpython/comments/xvlpdz/why_do_people_provide_a_requirementstxt_or/)

These resources could be useful for you!
