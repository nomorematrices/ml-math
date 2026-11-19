# Bài 17 — Biến Ngẫu nhiên và Phân phối Xác suất

## Mục tiêu

- Phân biệt biến ngẫu nhiên rời rạc/liên tục, PMF/PDF/CDF.
- Tính joint, marginal và conditional probability.
- Kiểm tra chuẩn hóa và support.

## Trọng tâm

Biến ngẫu nhiên ánh xạ kết quả thí nghiệm sang số. PMF $p(x)=P(X=x)$ cho biến rời rạc và tổng bằng 1. PDF $f(x)$ cho biến liên tục; xác suất là diện tích tích phân, nên $P(X=x)=0$ tại một điểm. CDF $F(x)=P(X\le x)$ áp dụng cho cả hai.

Từ joint $p(x,y)$, marginal $p(x)=\sum_y p(x,y)$ (hoặc tích phân). Conditional $p(x\mid y)=p(x,y)/p(y)$ khi $p(y)>0$. Bảng joint phải không âm và tổng bằng 1.

## Tự kiểm tra

1. PDF có thể lớn hơn 1 không? Vì sao?
2. Lấy marginal từ joint table như thế nào?
3. Conditional distribution cần chuẩn hóa theo trục nào?

## Checklist

- [ ] Xây và kiểm tra joint table.
- [ ] Tính marginal/conditional bằng NumPy.
- [ ] Mô phỏng để so sánh tần suất với xác suất.

