# ML Math — Toán nền tảng cho Machine Learning

Lộ trình tự học gồm **4 chương, 21 bài**. Mỗi bài có:

- `notes.md`: lý thuyết, công thức, trực giác, ví dụ và câu hỏi tự kiểm tra.
- `practice.ipynb`: ví dụ chạy được, bài tập NumPy và phần mở rộng.

## Cấu trúc

```text
ml-math/
├── chapters/
│   ├── 01-linear-algebra/       # Bài 01–12
│   ├── 02-matrix-calculus/      # Bài 13–16
│   ├── 03-probability/          # Bài 17–20
│   └── 04-parameter-estimation/ # Bài 21
├── docs/                        # Ký hiệu, công thức, hướng dẫn học
├── src/ml_math/                 # Hàm dùng chung cho notebook
├── tests/                       # Kiểm thử tiện ích
├── environment.yml             # Môi trường Conda
├── requirements.txt            # Môi trường pip
└── Makefile                     # Lệnh cài đặt/kiểm tra
```

Mỗi chương có một `README.md` làm mục lục. Các bài nằm trong thư mục `lesson-XX-...`.

## Bắt đầu nhanh

Yêu cầu Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
jupyter lab
```

Hoặc với Conda:

```bash
conda env create -f environment.yml
conda activate ml-math
jupyter lab
```

Chạy kiểm tra:

```bash
make test
make notebooks
```

## Cách học đề xuất

1. Đọc mục tiêu và phần ký hiệu trong `notes.md`.
2. Tự làm ví dụ trên giấy trước khi xem lời giải.
3. Chạy `practice.ipynb` từ trên xuống dưới.
4. Hoàn thành các ô `TODO`; không chỉ đọc output có sẵn.
5. Trả lời câu hỏi tự kiểm tra và ghi lại lỗi sai.
6. Chỉ chuyển bài khi đạt tối thiểu 80% checklist cuối bài.

Xem [hướng dẫn học](docs/study-guide.md), [quy ước ký hiệu](docs/notation.md) và [bảng công thức](docs/formula-cheatsheet.md).

## Lộ trình

| Chương | Bài | Trọng tâm |
|---|---:|---|
| Đại số tuyến tính | 01–12 | Vector, ma trận, không gian vector, phân rã, norm |
| Giải tích ma trận | 13–16 | Gradient, Jacobian, chain rule, gradient check |
| Xác suất | 17–20 | Phân phối, Bayes, kỳ vọng, hiệp phương sai |
| Ước lượng tham số | 21 | MLE, MAP và regularization |

## Quy ước notebook

- Seed mặc định: `42` để kết quả có thể tái lập.
- Dùng `numpy.ndarray`; vector 1D có shape `(n,)`, vector cột cần ghi rõ `(n, 1)`.
- Mỗi notebook có các phần: mục tiêu, khởi tạo, ví dụ, bài tập, kiểm tra, tổng kết.
- `assert` là phản hồi tức thời, không thay thế việc giải thích bằng lời.

