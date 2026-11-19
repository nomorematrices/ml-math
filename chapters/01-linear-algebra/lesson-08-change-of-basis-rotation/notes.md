# Bài 08 — Đổi Cơ sở và Phép xoay trục

## Mục tiêu

- Phân biệt vector hình học với tọa độ của nó.
- Chuyển tọa độ giữa hai basis.
- Xây dựng và áp dụng ma trận xoay 2D.

## Trọng tâm

Nếu cột của $B$ là các vector basis viết trong hệ chuẩn và $[x]_B$ là tọa độ, thì $x=B[x]_B$, do đó $[x]_B=B^{-1}x$. Từ basis $B$ sang $C$: $[x]_C=C^{-1}B[x]_B$.

Ma trận xoay chủ động ngược chiều kim đồng hồ góc $\theta$ là $R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}$. Cần phân biệt xoay vector (active) và xoay hệ trục (passive), thường khác nhau bởi chuyển vị/nghịch đảo.

## Tự kiểm tra

1. Vì sao tọa độ đổi nhưng vector hình học không đổi?
2. $R^{-1}$ liên hệ $R^T$ và $R(-\theta)$ ra sao?
3. Kiểm tra composition của hai phép xoay.

## Checklist

- [ ] Đổi tọa độ qua lại và tái tạo vector.
- [ ] Xoay một tập điểm 2D.
- [ ] Phân biệt active/passive transformation.

