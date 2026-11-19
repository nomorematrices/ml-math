# Bài 07 — Hệ trực chuẩn và Ma trận trực giao

## Mục tiêu

- Kiểm tra trực giao, trực chuẩn và orthogonal matrix.
- Hiểu projection và quy trình Gram–Schmidt/QR.
- Giải thích vì sao phép biến đổi trực giao bảo toàn norm.

## Trọng tâm

$x\perp y$ khi $x^Ty=0$. Một hệ trực chuẩn gồm các vector vuông góc đôi một và có norm 1. $Q$ trực giao khi $Q^TQ=QQ^T=I$, do đó $Q^{-1}=Q^T$ và $\|Qx\|_2=\|x\|_2$.

Projection lên vector đơn vị $q$ là $(q^Tx)q$; lên không gian có basis trực chuẩn $Q$ là $QQ^Tx$. QR decomposition tạo basis trực chuẩn bền vững hơn Gram–Schmidt cổ điển.

## Tự kiểm tra

1. Hai vector có dot product 0 nhưng chưa chuẩn hóa gọi là gì?
2. Chứng minh ngắn gọn $\|Qx\|_2=\|x\|_2$.
3. Projection matrix $QQ^T$ có tính chất đối xứng/idempotent nào?

## Checklist

- [ ] Chuẩn hóa vector và kiểm tra $Q^TQ\approx I$.
- [ ] Tính projection và residual.
- [ ] Dùng QR để tạo orthonormal basis.

