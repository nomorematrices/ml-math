# Bảng công thức nhanh

## Đại số tuyến tính

- Dot product: $\mathbf{x}^T\mathbf{y}=\sum_i x_i y_i$.
- Matrix product: $(\mathbf{A}\mathbf{B})_{ij}=\sum_k A_{ik}B_{kj}$.
- Determinant: $\det(\mathbf{A}\mathbf{B})=\det(\mathbf{A})\det(\mathbf{B})$.
- Rank-nullity: $\mathrm{rank}(\mathbf{A})+\mathrm{nullity}(\mathbf{A})=n$ với $\mathbf{A}\in\mathbb{R}^{m\times n}$.
- Eigendecomposition: $\mathbf{A}=\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{-1}$ khi chéo hóa được.
- Cholesky: $\mathbf{A}=\mathbf{L}\mathbf{L}^T$ khi $\mathbf{A}$ đối xứng xác định dương.
- Norm: $\|\mathbf{x}\|_p=(\sum_i|x_i|^p)^{1/p}$; $\|\mathbf{A}\|_F=\sqrt{\sum_{ij}A_{ij}^2}$.

## Giải tích ma trận

- $\nabla_{\mathbf{x}}(\mathbf{a}^T\mathbf{x})=\mathbf{a}$.
- $\mathbf{J}_{\mathbf{A}\mathbf{x}}=\mathbf{A}$.
- $\nabla_{\mathbf{x}}\|\mathbf{A}\mathbf{x}-\mathbf{b}\|_2^2=2\mathbf{A}^T(\mathbf{A}\mathbf{x}-\mathbf{b})$.
- $\nabla_{\mathbf{X}}\mathrm{tr}(\mathbf{A}\mathbf{X})=\mathbf{A}^T$.
- Central difference: $f'(x)\approx [f(x+h)-f(x-h)]/(2h)$.

## Xác suất và ước lượng

- Bayes: $p(\theta\mid x)=p(x\mid\theta)p(\theta)/p(x)$.
- $\mathrm{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2$.
- MLE: $\hat\theta=\arg\max_\theta\sum_i\log p(x_i\mid\theta)$.
- MAP: $\hat\theta=\arg\max_\theta[\log p(D\mid\theta)+\log p(\theta)]$.

