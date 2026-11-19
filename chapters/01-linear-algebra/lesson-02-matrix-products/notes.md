# Bài 02 — Nhân ma trận, Tích vô hướng và Tích Hadamard

## Mục tiêu

- Phân biệt dot product, matrix product và element-wise product.
- Kiểm tra điều kiện tương thích shape trước khi nhân.
- Liên hệ phép nhân ma trận với lớp tuyến tính trong ML.

## Trọng tâm

$\mathbf{x}^T\mathbf{y}=\sum_i x_i y_i$ trả về scalar. Nếu $A\in\mathbb{R}^{m\times n}$ và $B\in\mathbb{R}^{n\times p}$ thì $AB\in\mathbb{R}^{m\times p}$ với $(AB)_{ij}=\sum_k A_{ik}B_{kj}$. Tích Hadamard $A\odot B$ nhân từng phần tử và đòi hỏi shape tương thích.

Phép nhân ma trận kết hợp nhưng không giao hoán. Trong NumPy dùng `@` cho matrix product, `*` cho Hadamard. Linear layer cho một batch thường là `X @ W + b`.

## Ví dụ

Nếu `X.shape == (32, 10)` và `W.shape == (10, 4)`, output có shape `(32, 4)`; bias `(4,)` được broadcast theo từng mẫu.

## Tự kiểm tra

1. $AB$ tồn tại khi nào và shape kết quả là gì?
2. Vì sao $AB$ thường khác $BA$?
3. Cosine similarity liên hệ dot product ra sao?

## Checklist

- [ ] Tính tay dot product và matrix product.
- [ ] Phân biệt `@`, `*` và broadcasting.
- [ ] Suy luận được shape của một linear layer.

