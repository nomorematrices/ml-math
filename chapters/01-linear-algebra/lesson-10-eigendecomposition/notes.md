# Bài 10 — Chéo hóa Ma trận

## Mục tiêu

- Viết $A=V\Lambda V^{-1}$ khi đủ eigenvector độc lập.
- Dùng phân rã để tính lũy thừa ma trận.
- Nhận biết trường hợp không chéo hóa được.

## Trọng tâm

Nếu các cột của $V$ là $n$ eigenvector độc lập và $\Lambda$ chứa eigenvalue tương ứng trên đường chéo thì $AV=V\Lambda$ và $A=V\Lambda V^{-1}$. Khi đó $A^k=V\Lambda^kV^{-1}$, giúp hiểu động lực học lặp.

Không phải ma trận nào cũng có đủ eigenvector độc lập. Với symmetric matrix, spectral theorem đảm bảo $A=Q\Lambda Q^T$ với $Q$ trực giao, ổn định và dễ diễn giải hơn.

## Tự kiểm tra

1. Điều kiện nào để $V^{-1}$ tồn tại?
2. Vì sao tính $A^{100}$ qua diagonalization đơn giản?
3. Phân biệt eigendecomposition với SVD.

## Checklist

- [ ] Tái tạo $A$ từ eigenpairs.
- [ ] So sánh $A^k$ trực tiếp và qua $\Lambda^k$.
- [ ] Phát hiện ma trận defective đơn giản.

