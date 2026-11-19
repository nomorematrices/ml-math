# Bài 16 — Numerical Gradient Check với Python

## Mục tiêu

- Cài forward và central finite difference.
- Chọn step size hợp lý và đo relative error.
- Dùng gradient check để tìm lỗi đạo hàm.

## Trọng tâm

Central difference $[f(x+he_i)-f(x-he_i)]/(2h)$ có sai số truncation tốt hơn forward difference. Nếu $h$ quá lớn, xấp xỉ tuyến tính kém; nếu quá nhỏ, floating-point cancellation chi phối. Với `float64`, thử dải $10^{-2}$ đến $10^{-7}$.

Sai số tương đối $\|g_a-g_n\|/(\|g_a\|+\|g_n\|+\epsilon)$ hữu ích hơn absolute error. Gradient check nên chạy trên input nhỏ, deterministic, tránh điểm không khả vi và vô hiệu hóa randomness.

## Tự kiểm tra

1. Vì sao central difference thường chính xác hơn?
2. Điều gì xảy ra tại điểm gấp khúc của ReLU?
3. Gradient đúng có chắc code huấn luyện đúng hoàn toàn không?

## Checklist

- [ ] Cài gradient check không sửa input gốc.
- [ ] Quét nhiều step size và vẽ sai số.
- [ ] Cố ý tạo bug rồi phát hiện bằng relative error.

