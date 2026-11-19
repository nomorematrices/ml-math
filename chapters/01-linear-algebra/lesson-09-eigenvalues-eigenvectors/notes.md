# Bài 09 — Trị riêng và Vector riêng

## Mục tiêu

- Hiểu phương trình eigen và ý nghĩa hình học.
- Tính, sắp xếp và kiểm tra eigenpair.
- Liên hệ eigenvector với hướng bất biến và PCA.

## Trọng tâm

$Av=\lambda v$ với $v\ne0$: phép biến đổi giữ nguyên phương của $v$, chỉ co giãn/đảo hướng theo $\lambda$. Eigenvalue là nghiệm của $\det(A-\lambda I)=0$. Eigenvector chỉ xác định tới một hệ số khác 0.

Ma trận đối xứng thực có eigenvalue thực và một orthonormal eigenbasis. Đây là nền tảng của PCA trên covariance matrix. `np.linalg.eigh` nên dùng cho ma trận đối xứng thay vì `eig`.

## Tự kiểm tra

1. Eigenvalue 0 nói gì về khả nghịch?
2. Vì sao eigenvector không phải là duy nhất về độ dài/dấu?
3. Trace và determinant liên hệ các eigenvalue ra sao?

## Checklist

- [ ] Kiểm tra $Av\approx\lambda v$.
- [ ] Dùng `eigh` cho symmetric matrix.
- [ ] Diễn giải eigenvector chính của covariance matrix.

