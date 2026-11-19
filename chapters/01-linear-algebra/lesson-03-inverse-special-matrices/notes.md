# Bài 03 — Ma trận đơn vị, Nghịch đảo và Ma trận đặc biệt

## Mục tiêu

- Nhận biết identity, diagonal, triangular, symmetric và permutation matrix.
- Hiểu điều kiện tồn tại và ý nghĩa của nghịch đảo.
- Giải hệ tuyến tính ổn định bằng `solve`.

## Trọng tâm

$AI=IA=A$. Ma trận vuông $A$ khả nghịch nếu tồn tại $A^{-1}$ sao cho $AA^{-1}=I$. Khi đó nghiệm của $Ax=b$ là $x=A^{-1}b$, nhưng trong tính toán nên dùng phép giải hệ thay vì tạo nghịch đảo tường minh.

Ma trận chéo giúp biến đổi độc lập từng tọa độ; ma trận đối xứng thỏa $A=A^T$; triangular matrix cho phép thế tiến/lùi hiệu quả. Ma trận suy biến không có nghịch đảo và có thể có vô số hoặc không có nghiệm.

## Tự kiểm tra

1. Vì sao chỉ ma trận vuông mới có nghịch đảo hai phía thông thường?
2. Khi nào `solve(A, b)` tốt hơn `inv(A) @ b`?
3. Nêu shape và tác dụng của ma trận permutation.

## Checklist

- [ ] Phân loại được các ma trận đặc biệt.
- [ ] Kiểm tra $AA^{-1}\approx I$ bằng sai số số học.
- [ ] Giải và kiểm tra hệ $Ax=b$.

