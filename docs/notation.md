# Quy ước ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $a$ | vô hướng (scalar) |
| $\mathbf{x}$ | vector cột |
| $\mathbf{A}$ | ma trận |
| $x_i$ | phần tử thứ $i$ của vector |
| $A_{ij}$ | phần tử hàng $i$, cột $j$ |
| $\mathbf{A}^T$ | chuyển vị |
| $\mathbf{A}^{-1}$ | nghịch đảo, nếu tồn tại |
| $\mathbf{I}_n$ | ma trận đơn vị cấp $n$ |
| $\|\mathbf{x}\|_p$ | chuẩn $L_p$ |
| $\nabla_{\mathbf{x}} f$ | gradient theo $\mathbf{x}$ |
| $\mathbf{J}_f$ | Jacobian của hàm vector |
| $\mathbb{E}[X]$ | kỳ vọng |
| $\mathrm{Var}(X)$ | phương sai |
| $\mathrm{Cov}(X,Y)$ | hiệp phương sai |
| $p(x)$ | hàm khối/mật độ xác suất theo ngữ cảnh |

Trong code, luôn kiểm tra `shape`, phân biệt `*` (Hadamard) với `@` (nhân ma trận), và ưu tiên `np.linalg.solve(A, b)` thay vì tính `inv(A) @ b`.

