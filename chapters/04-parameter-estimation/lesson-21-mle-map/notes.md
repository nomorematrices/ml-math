# Bài 21 — Maximum Likelihood (MLE) và Maximum A Posteriori (MAP)

## Mục tiêu

- Xây likelihood/log-likelihood từ mô hình xác suất.
- Suy ra MLE cho Bernoulli và Gaussian đơn giản.
- Hiểu MAP là MLE cộng thông tin prior/regularization.

## Trọng tâm

Với dữ liệu i.i.d. $D=\{x_i\}$, likelihood $L(\theta)=\prod_i p(x_i\mid\theta)$. MLE chọn $\hat\theta=\arg\max L(\theta)$; thực tế tối ưu log-likelihood $\ell(\theta)=\sum_i\log p(x_i\mid\theta)$ để tránh underflow.

MAP chọn mode của posterior: $\hat\theta_{MAP}=\arg\max[\log p(D\mid\theta)+\log p(\theta)]$. Gaussian prior trên weights tạo $L_2$ penalty; Laplace prior tạo $L_1$ penalty. Khi dữ liệu lớn, likelihood thường chi phối prior. MLE/MAP là point estimate, không biểu diễn toàn bộ uncertainty.

Ví dụ Bernoulli: MLE $\hat p=k/n$. Với Beta$(\alpha,\beta)$ prior, MAP nội miền là $(k+\alpha-1)/(n+\alpha+\beta-2)$ khi điều kiện tham số phù hợp.

## Tự kiểm tra

1. Vì sao log không đổi nghiệm argmax?
2. MLE khác MAP ở thành phần nào?
3. Prior mạnh ảnh hưởng thế nào khi dữ liệu ít/nhiều?
4. Negative log-likelihood liên hệ loss function ra sao?

## Checklist

- [ ] Viết likelihood và log-likelihood đúng support.
- [ ] Tìm MLE bằng giải tích và grid search.
- [ ] So sánh MLE/MAP khi thay đổi cỡ mẫu và prior.
- [ ] Liên hệ MAP với regularized empirical risk.

