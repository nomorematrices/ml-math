# Bài 12 — Norm của Vector, Frobenius Norm và Trace

## Mục tiêu

- Tính và so sánh $L_1,L_2,L_p,L_\infty$.
- Dùng Frobenius norm và trace cho ma trận.
- Liên hệ norm với distance và regularization.

## Trọng tâm

$\|x\|_p=(\sum_i|x_i|^p)^{1/p}$ với $p\ge1$; $\|x\|_\infty=\max_i|x_i|$. Frobenius norm là $\|A\|_F=\sqrt{\sum_{ij}A_{ij}^2}=\sqrt{\mathrm{tr}(A^TA)}$. Trace là tổng đường chéo và có tính cyclic: $\mathrm{tr}(ABC)=\mathrm{tr}(BCA)$ khi shape hợp lệ.

$L_1$ khuyến khích sparsity, $L_2$ co nhỏ hệ số trơn hơn. Norm phải không âm, đồng nhất tuyệt đối và thỏa bất đẳng thức tam giác.

## Tự kiểm tra

1. Khi nào $L_1$ và $L_2$ bằng nhau?
2. Vì sao squared $L_2$ không phải norm?
3. Chứng minh $\|A\|_F^2=\mathrm{tr}(A^TA)$.

## Checklist

- [ ] Tính các norm bằng tay và NumPy.
- [ ] Kiểm tra triangle inequality bằng mô phỏng.
- [ ] Liên hệ $L_1/L_2$ với regularization.

