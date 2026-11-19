# Bài 13 — Đạo hàm Hàm trả về Vô hướng và Vector

## Mục tiêu

- Phân biệt derivative, gradient và Jacobian.
- Theo dõi shape của đầu vào/đầu ra/đạo hàm.
- Tính đạo hàm từng phần cho hàm nhiều biến.

## Trọng tâm

Với $f:\mathbb{R}^n\to\mathbb{R}$, gradient $\nabla_x f=[\partial f/\partial x_1,\ldots,\partial f/\partial x_n]^T$ chỉ hướng tăng nhanh nhất và có shape như $x$. Với $f:\mathbb{R}^n\to\mathbb{R}^m$, Jacobian $J_f\in\mathbb{R}^{m\times n}$ có phần tử $(J_f)_{ij}=\partial f_i/\partial x_j$.

Ví dụ $f(x)=x^Tx$ có $\nabla f=2x$; $g(x)=Ax+b$ có $J_g=A$. Hãy luôn khai báo convention vì một số tài liệu dùng transpose của Jacobian.

## Tự kiểm tra

1. Shape Jacobian của $f:\mathbb{R}^5\to\mathbb{R}^3$ là gì?
2. Gradient biểu diễn local linear approximation ra sao?
3. Tính gradient của $x_1^2+3x_1x_2$.

## Checklist

- [ ] Tính partial derivative và gradient.
- [ ] Xây Jacobian cho hàm vector đơn giản.
- [ ] Kiểm tra shape trước khi kiểm tra giá trị.

