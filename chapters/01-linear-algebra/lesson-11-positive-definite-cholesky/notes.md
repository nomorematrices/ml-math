# Bài 11 — Ma trận Xác định dương và Cholesky

## Mục tiêu

- Phân biệt positive definite (PD) và semidefinite (PSD).
- Kiểm tra bằng quadratic form/eigenvalue.
- Dùng Cholesky để giải hệ và mô phỏng Gaussian.

## Trọng tâm

Ma trận đối xứng $A$ là PD nếu $x^TAx>0$ với mọi $x\ne0$, PSD nếu $x^TAx\ge0$. Với ma trận đối xứng, PD tương đương mọi eigenvalue dương. Covariance matrix luôn PSD; thêm regularization $\lambda I$ thường làm nó PD.

Nếu $A$ đối xứng PD, Cholesky cho $A=LL^T$ với $L$ triangular và đường chéo dương. Phân rã này hiệu quả để giải hệ, tính log-determinant và sinh Gaussian tương quan.

## Tự kiểm tra

1. Vì sao $X^TX$ luôn PSD?
2. Khi nào $X^TX+\lambda I$ là PD?
3. Cholesky thất bại cho loại ma trận nào?

## Checklist

- [ ] Kiểm tra symmetry và eigenvalue.
- [ ] Tái tạo $A$ từ $L$.
- [ ] Dùng triangular solve thay cho nghịch đảo.

