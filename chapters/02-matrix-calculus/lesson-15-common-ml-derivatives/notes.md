# Bài 15 — Các Đạo hàm Thường gặp trong ML

## Mục tiêu

- Ghi nhớ và tự suy ra các identity quan trọng.
- Tính gradient least-squares và hàm trace.
- Kiểm tra công thức bằng differential và shape.

## Trọng tâm

Các công thức theo column-gradient convention:

- $\nabla_x(a^Tx)=a$.
- $J_{Ax}=A$.
- $\nabla_x\|Ax-b\|_2^2=2A^T(Ax-b)$.
- $\nabla_X\mathrm{tr}(AX)=A^T$.
- $\nabla_x(x^TAx)=(A+A^T)x$, bằng $2Ax$ nếu $A$ đối xứng.

Quy trình an toàn: ghi shape, dùng differential, đưa về $df=\mathrm{tr}(G^T dX)$ rồi đọc gradient $G$, cuối cùng kiểm tra số.

## Tự kiểm tra

1. Vì sao least-squares gradient có $A^T$?
2. Khi nào gradient quadratic form bằng $2Ax$?
3. Tìm nghiệm stationary của least squares khi $A^TA$ khả nghịch.

## Checklist

- [ ] Suy ra least-squares gradient.
- [ ] Kiểm tra identity trace bằng numerical gradient.
- [ ] Cài một bước gradient descent và thấy loss giảm.

