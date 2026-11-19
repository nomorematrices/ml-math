# Bài 19 — Kỳ vọng, Phương sai và Covariance Matrix

## Mục tiêu

- Tính expectation, variance, covariance.
- Dùng linearity of expectation.
- Xây và diễn giải covariance/correlation matrix.

## Trọng tâm

$E[X]=\sum_x xp(x)$ hoặc $\int xf(x)dx$. $\mathrm{Var}(X)=E[(X-E[X])^2]=E[X^2]-E[X]^2$. $\mathrm{Cov}(X,Y)=E[(X-E[X])(Y-E[Y])]$; correlation chuẩn hóa về $[-1,1]$.

Covariance matrix $\Sigma=E[(X-\mu)(X-\mu)^T]$ đối xứng PSD; đường chéo là variance. Zero covariance không nhất thiết độc lập, trừ một số họ như jointly Gaussian. Với sample variance, `ddof=1` dùng mẫu số $n-1$.

## Tự kiểm tra

1. $E[aX+b]$ và $Var(aX+b)$ bằng gì?
2. Vì sao covariance phụ thuộc đơn vị còn correlation không?
3. Chứng minh covariance matrix PSD qua quadratic form.

## Checklist

- [ ] Tính đại lượng từ PMF và từ samples.
- [ ] So sánh covariance với correlation.
- [ ] Kiểm tra eigenvalue của covariance không âm (tới sai số số).

