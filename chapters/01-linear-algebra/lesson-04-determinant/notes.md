# Bài 04 — Định thức và Tính chất

## Mục tiêu

- Tính determinant cho ma trận nhỏ và hiểu ý nghĩa hình học.
- Dùng determinant để nhận biết khả nghịch.
- Biết giới hạn của determinant trong tính toán số.

## Trọng tâm

$\det(A)$ là hệ số co giãn thể tích có hướng của phép biến đổi tuyến tính. $|\det(A)|$ là hệ số thể tích; dấu cho biết orientation có bị đảo hay không. $A$ khả nghịch khi và chỉ khi $\det(A)\ne0$.

Các tính chất: $\det(AB)=\det(A)\det(B)$, $\det(A^T)=\det(A)$, đổi chỗ hai hàng đổi dấu, và determinant của ma trận triangular là tích đường chéo. Với ma trận lớn, không dùng determinant để kiểm tra ổn định; dùng rank, singular values hoặc condition number.

## Tự kiểm tra

1. Phép biến đổi có determinant $-3$ thay đổi diện tích và hướng ra sao?
2. Điều gì xảy ra khi hai hàng phụ thuộc tuyến tính?
3. Vì sao `slogdet` hữu ích hơn `det` cho likelihood Gaussian nhiều chiều?

## Checklist

- [ ] Tính determinant $2\times2$ và $3\times3$.
- [ ] Giải thích ý nghĩa hình học.
- [ ] Kiểm chứng các tính chất bằng NumPy.

