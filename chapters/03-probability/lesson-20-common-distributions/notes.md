# Bài 20 — Các Phân phối Thường gặp

## Mục tiêu

- Nhận biết Bernoulli, Categorical, Normal, Beta và Dirichlet.
- Chọn distribution theo support và kiểu dữ liệu.
- Mô phỏng, ước lượng moment và vẽ mật độ.

## Trọng tâm

- Bernoulli$(p)$: một biến nhị phân, $E[X]=p$, $Var(X)=p(1-p)$.
- Categorical$(\pi)$: một trong $K$ lớp, $\sum_k\pi_k=1$.
- Normal$(\mu,\sigma^2)$: biến liên tục trên $\mathbb{R}$.
- Beta$(\alpha,\beta)$: phân phối trên $[0,1]$, prior liên hợp cho Bernoulli probability.
- Dirichlet$(\alpha)$: phân phối trên probability simplex, mở rộng Beta cho Categorical.

Tham số hóa variance/scale dễ bị nhầm giữa thư viện. Luôn đọc API và kiểm tra bằng moment mô phỏng.

## Tự kiểm tra

1. Distribution nào phù hợp xác suất click chưa biết?
2. Dirichlet sample có tổng các thành phần bằng bao nhiêu?
3. Beta$(1,1)$ có hình dạng gì?

## Checklist

- [ ] Chọn đúng distribution theo support.
- [ ] Sinh mẫu reproducible.
- [ ] So sánh sample mean/variance với lý thuyết.

