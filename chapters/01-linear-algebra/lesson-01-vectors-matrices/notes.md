# Bài 01 — Ký hiệu cơ bản, Vector và Ma trận

## Mục tiêu

- Phân biệt scalar, vector, matrix, tensor và đọc đúng chỉ số.
- Xác định shape, chuyển vị, truy cập hàng/cột.
- Biểu diễn một mẫu dữ liệu và một tập dữ liệu bằng ma trận.

## Trọng tâm

Scalar $a\in\mathbb{R}$ là một số. Vector $\mathbf{x}\in\mathbb{R}^n$ là dãy có thứ tự. Ma trận $\mathbf{A}\in\mathbb{R}^{m\times n}$ có $m$ hàng và $n$ cột. $A_{ij}$ là phần tử hàng $i$, cột $j$; $\mathbf{A}^T\in\mathbb{R}^{n\times m}$ và $(A^T)_{ij}=A_{ji}$.

Trong ML, thường mỗi hàng là một mẫu và mỗi cột là một đặc trưng: $\mathbf{X}\in\mathbb{R}^{N\times d}$. Vector NumPy shape `(d,)` không tự động là vector cột; khi cần, dùng `x.reshape(-1, 1)`.

## Ví dụ

Với $X=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$, shape là $2\times3$, hàng đầu là một mẫu, cột thứ hai là đặc trưng thứ hai, và $X^T$ có shape $3\times2$.

## Lỗi thường gặp

- Nhầm số hàng và số cột; lẫn chỉ số toán học bắt đầu từ 1 với Python bắt đầu từ 0.
- Cho rằng `(n,)`, `(1,n)` và `(n,1)` luôn tương đương.

## Tự kiểm tra

1. Nếu $X$ có 100 mẫu, 8 đặc trưng thì shape nào hợp lý?
2. Shape của $X^T$ là gì? `X[:, 2]` lấy hàng hay cột?
3. Hãy mô tả một ảnh RGB như một tensor.

## Checklist

- [ ] Đọc được ký hiệu chỉ số và shape.
- [ ] Tạo, reshape, transpose và slice bằng NumPy.
- [ ] Giải thích được cách dữ liệu tabular được lưu trong ma trận.

