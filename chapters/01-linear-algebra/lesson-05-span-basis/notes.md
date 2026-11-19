# Bài 05 — Tổ hợp tuyến tính, Span và Basis

## Mục tiêu

- Viết vector dưới dạng tổ hợp tuyến tính.
- Phân biệt span, độc lập tuyến tính và basis.
- Tìm tọa độ của vector theo một basis.

## Trọng tâm

$v=\sum_i c_i v_i$ là một tổ hợp tuyến tính. Span của tập vector là toàn bộ tổ hợp tuyến tính của chúng. Một basis của không gian vừa độc lập tuyến tính vừa sinh ra toàn bộ không gian; mọi vector có biểu diễn tọa độ duy nhất theo basis đó.

Các cột của $A$ sinh ra column space. Câu hỏi “$b$ có thuộc span các cột của $A$ không?” tương đương “$Ax=b$ có nghiệm không?”. Trong ML, basis chọn cách biểu diễn đặc trưng; basis tốt có thể làm cấu trúc dữ liệu rõ hơn.

## Tự kiểm tra

1. Hai vector cùng phương có thể là basis của $\mathbb{R}^2$ không?
2. Một basis của $\mathbb{R}^d$ có bao nhiêu vector?
3. Hãy liên hệ coefficients trong linear regression với tổ hợp tuyến tính.

## Checklist

- [ ] Kiểm tra độc lập tuyến tính bằng rank.
- [ ] Tìm coefficients bằng giải hệ.
- [ ] Phân biệt spanning set và basis.

