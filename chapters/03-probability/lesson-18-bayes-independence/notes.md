# Bài 18 — Quy tắc Bayes và Độc lập

## Mục tiêu

- Áp dụng Bayes từ prior, likelihood và evidence.
- Phân biệt independence với conditional independence.
- Diễn giải base-rate effect trong bài toán chẩn đoán/phân loại.

## Trọng tâm

$p(A\mid B)=p(B\mid A)p(A)/p(B)$, với $p(B)=\sum_i p(B\mid A_i)p(A_i)$. Prior mô tả niềm tin trước dữ liệu; likelihood đánh giá dữ liệu theo giả thuyết; posterior là niềm tin sau cập nhật.

$X$ và $Y$ độc lập nếu $p(x,y)=p(x)p(y)$; khi đó $p(x\mid y)=p(x)$. Conditional independence $X\perp Y\mid Z$ không suy ra độc lập vô điều kiện. Naive Bayes giả định các feature độc lập có điều kiện theo class.

## Tự kiểm tra

1. Vì sao test chính xác cao vẫn có posterior thấp khi bệnh hiếm?
2. Likelihood có phải phân phối theo tham số không?
3. Pairwise independence có luôn kéo theo mutual independence không?

## Checklist

- [ ] Tính posterior và kiểm tra tổng bằng 1.
- [ ] Mô phỏng base-rate example.
- [ ] Kiểm tra independence từ joint table.

