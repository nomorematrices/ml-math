# Bài 06 — Range, Null Space và Rank

Tài liệu đi kèm gồm [thực hành NumPy](practice.ipynb) và
[liên hệ với Machine Learning trong thực tế](ml-in-practice.md).

## Mục tiêu

Sau bài này, bạn có thể làm được các việc sau.

- Đọc shape và xác định không gian chứa range, row space, null space và left null space.
- Tìm basis của các không gian con bằng khử hàng trên ma trận nhỏ.
- Áp dụng định lý rank–nullity và phân biệt full row rank với full column rank.
- Dựa vào range và null space để phân loại nghiệm của hệ $Ax=b$.
- Dùng NumPy và SVD để tính numerical rank, lấy basis và kiểm chứng bằng phần dư.
- Giải thích vì sao feature phụ thuộc làm nhiều bộ trọng số tạo ra cùng dự đoán.

## 1. Từ span đến bốn không gian con

Cho $A\in\mathbb R^{m\times n}$. Phép biến đổi $x\mapsto Ax$ nhận $x\in\mathbb R^n$
và trả về $Ax\in\mathbb R^m$. Trong NumPy, `(m, n) @ (n,)` cho `(m,)`.

| Không gian | Định nghĩa | Nằm trong | Số chiều nếu $\operatorname{rank}(A)=r$ |
|---|---|---|---:|
| Column space hay range | $\operatorname{Col}(A)=\{Ax:x\in\mathbb R^n\}$ | $\mathbb R^m$ | $r$ |
| Row space | $\operatorname{Row}(A)=\operatorname{Col}(A^T)$ | $\mathbb R^n$ | $r$ |
| Null space hay kernel | $\ker(A)=\{x\in\mathbb R^n:Ax=0\}$ | $\mathbb R^n$ | $n-r$ |
| Left null space | $\ker(A^T)=\{y\in\mathbb R^m:A^Ty=0\}$ | $\mathbb R^m$ | $m-r$ |

Theo Bài 05, $Ax$ là tổ hợp tuyến tính của các cột nên range là span của các cột.
Theo góc nhìn từng hàng ở Bài 02, mỗi thành phần của $Ax$ là dot product của $x$
với một hàng của $A$. Vì vậy $Ax=0$ khi $x$ vuông góc với từng hàng, hay
$\ker(A)=\operatorname{Row}(A)^\perp$. Áp dụng lập luận cho $A^T$ cho ta
$\ker(A^T)=\operatorname{Col}(A)^\perp$.

Null space mô tả các hướng đầu vào bị triệt tiêu. Nếu $z\in\ker(A)$ thì
$A(x+z)=Ax$. Left null space cho các ràng buộc trên đầu ra. Với mọi
$y\in\ker(A^T)$ và $b=Ax$, ta có $y^Tb=0$.

## 2. Tìm basis bằng tay từ một ví dụ xuyên suốt

Xét ma trận $3\times4$ sau.

$$
A=\begin{bmatrix}
1&0&1&2\\
0&1&2&-1\\
1&1&3&1
\end{bmatrix}
\quad\longrightarrow\quad
\begin{bmatrix}
1&0&1&2\\
0&1&2&-1\\
0&0&0&0
\end{bmatrix}.
$$

Trừ hai hàng đầu khỏi hàng thứ ba để được dạng bậc thang rút gọn. Hai pivot
nằm ở cột 1 và 2, theo chỉ số toán học bắt đầu từ 1.

- **Column space.** Lấy cột 1 và 2 của $A$ gốc, tức $(1,0,1)^T$ và $(0,1,1)^T$.
  Range là mặt phẳng $b_3=b_1+b_2$ trong $\mathbb R^3$ đã gặp ở Bài 05.
- **Row space.** Hai hàng khác zero của ma trận rút gọn tạo một basis. Mỗi hàng
  có bốn thành phần.
- **Null space.** Đặt $x_3=s$, $x_4=t$. Hai hàng đầu cho $x_1=-s-2t$ và
  $x_2=-2s+t$. Vì vậy mọi nghiệm của $Ax=0$ có dạng
  $s(-1,-2,1,0)^T+t(-2,1,0,1)^T$. Hai hệ số tự do cho hai vector basis.
- **Left null space.** Hàng 1 cộng hàng 2 bằng hàng 3, nên $(1,1,-1)^T$ thuộc
  $\ker(A^T)$. Không gian này có số chiều $3-2=1$, nên vector đó tạo một basis.

Phép khử hàng bảo toàn row space và tập nghiệm của $Ax=0$. Nó giữ quan hệ
phụ thuộc giữa các cột, vì vậy các chỉ số pivot xác định cột nào cần lấy từ
ma trận gốc. Bản thân các cột trong ma trận rút gọn có thể nằm ngoài range gốc.

```python
import numpy as np

A = np.array([[1., 0., 1., 2.], [0., 1., 2., -1.], [1., 1., 3., 1.]])
C = A[:, :2]                                       # (3, 2), basis của range
Z = np.array([[-1., -2.], [-2., 1.], [1., 0.], [0., 1.]])  # (4, 2)
z_left = np.array([1., 1., -1.])                   # (3,)

assert np.linalg.matrix_rank(C) == 2
assert np.linalg.matrix_rank(Z) == 2
np.testing.assert_allclose(A @ Z, 0., atol=1e-12)
np.testing.assert_allclose(A.T @ z_left, 0., atol=1e-12)
```

## 3. Rank, rank–nullity và full rank

Rank là số pivot, bằng cả số chiều column space lẫn số chiều row space.
Với $n$ biến, có $r$ biến pivot và $n-r$ biến tự do trong hệ thuần nhất. Mỗi biến
tự do cho một hướng độc lập trong null space. Đây là định lý **rank–nullity**.

$$
\operatorname{rank}(A)+\dim\ker(A)=n.
$$

Áp dụng cho $A^T$ cho $r+\dim\ker(A^T)=m$. Trong ví dụ trên, $2+2=4$ ở phía
đầu vào và $2+1=3$ ở phía đầu ra.

| Điều kiện | Ý nghĩa với $Ax=b$ |
|---|---|
| Full column rank, $r=n$ | $\ker(A)=\{0\}$, nên mỗi $b$ có tối đa một nghiệm chính xác. |
| Full row rank, $r=m$ | Range là $\mathbb R^m$, nên mọi $b$ đều có ít nhất một nghiệm. |
| Vuông và full rank, $r=m=n$ | Mọi $b$ có đúng một nghiệm, tương đương $\det(A)\ne0$ theo Bài 04. |

Luôn có $0\le r\le\min(m,n)$. Ma trận cao ($m>n$) có thể full column rank
và vẫn có đầu ra không đạt được. Ma trận rộng ($m<n$) có thể full row rank
và vẫn có biến tự do. Với ma trận zero, rank bằng 0 và mọi đầu vào thuộc kernel.

## 4. Range quyết định có nghiệm, null space quyết định nghiệm duy nhất

Hệ $Ax=b$ có nghiệm chính xác khi $b\in\operatorname{Col}(A)$. Nếu đã có một
nghiệm $x_0$, tập nghiệm là $x_0+\ker(A)$. Thật vậy, hiệu $x-x_0$ giữa hai
nghiệm thỏa $A(x-x_0)=0$, và cộng bất kỳ vector kernel nào vào $x_0$ vẫn cho $b$.
Trong ví dụ trên, $b=(2,-1,1)^T$ có nghiệm $x_0=(2,-1,0,0)^T$ cùng vô số
nghiệm sinh ra khi cộng các hướng của $Z$. Đích $(2,-1,0)^T$ nằm ngoài range.

Để kiểm tra bằng NumPy, dùng `lstsq` và tính phần dư trực tiếp như Bài 05.

```python
targets = np.array([[2., 2.], [-1., -1.], [1., 0.]])  # (3, 2), mỗi cột một đích
solutions, _, rank, _ = np.linalg.lstsq(A, targets, rcond=None)
fitted = A @ solutions                                # (3, 2)
errors = np.linalg.norm(fitted - targets, axis=0)      # (2,)
limits = 1e-10 + 1e-8 * np.linalg.norm(targets, axis=0)
np.testing.assert_array_equal(errors <= limits, [True, False])
```

Khi $b$ nằm ngoài range, least squares tìm $\hat b=A\hat x$ gần $b$ nhất theo
chuẩn Euclid. $\hat b$ là duy nhất. Các hệ số $\hat x$ là duy nhất khi $A$ có
full column rank. Khi có kernel khác $\{0\}$, `lstsq` chọn nghiệm có chuẩn
Euclid nhỏ nhất trong số các nghiệm tối ưu. Phần dư $b-\hat b$ vuông góc với
range nên $A^T(b-\hat b)=0$, liên hệ với normal equation ở Bài 03.

## 5. Tìm basis bằng SVD với NumPy

SVD viết $A=U\Sigma V^T$, trong đó các cột của $U$ và $V$ tạo thành các basis
trực chuẩn ở hai phía. $\Sigma$ có các singular value $\sigma_1\ge\sigma_2\ge\cdots\ge0$
trên đường chéo. Với mỗi cặp tương ứng, $Av_i=\sigma_i u_i$. Các hướng $v_i$
có singular value bằng 0 bị triệt tiêu. Khi $n>m$, các hướng $v_i$ ngoài
$\min(m,n)$ singular value được liệt kê cũng bị triệt tiêu.

```python
U, s, Vt = np.linalg.svd(A, full_matrices=True)
r = np.linalg.matrix_rank(A)
column_basis = U[:, :r]            # (m, r)
row_basis = Vt[:r, :].T             # (n, r), mỗi cột là một vector basis
null_basis = Vt[r:, :].T            # (n, n - r)
left_null_basis = U[:, r:]          # (m, m - r)

assert r + null_basis.shape[1] == A.shape[1]
np.testing.assert_allclose(A @ null_basis, 0., atol=1e-12)
np.testing.assert_allclose(A.T @ left_null_basis, 0., atol=1e-12)
np.testing.assert_allclose(row_basis.T @ null_basis, 0., atol=1e-12)
```

Với SVD đầy đủ, `U` có shape `(m, m)`, `Vt` có shape `(n, n)`, còn `s` có
`min(m, n)` phần tử. `full_matrices=False` phù hợp khi chỉ cần các hướng giữ lại
để tái tạo hạng thấp. Với ma trận rộng, dạng rút gọn thiếu $n-m$ hướng của
kernel. Tương tự, với ma trận cao, nó thiếu $m-n$ hướng của left null space.

Basis từ SVD có thể khác basis tìm tay về dấu và hướng. Kiểm tra bằng số chiều,
phần dư và tính độc lập. Với basis trực chuẩn, `Z.T @ Z` gần ma trận đơn vị.

## 6. Numerical rank phụ thuộc vào tolerance

Trong số học chính xác, rank đếm singular value dương. Với số thực dấu phẩy động,
NumPy ước lượng rank bằng cách đếm `s > tol`. Ngưỡng mặc định dựa trên
$s_{\max}\max(m,n)\varepsilon$, với $\varepsilon$ theo kiểu dữ liệu của singular value.
Ngưỡng lớn hơn có thể phù hợp khi độ nhiễu của dữ liệu lớn hơn sai số làm tròn.

```python
near = np.diag([1.0, 1e-9])
print(np.linalg.svd(near, compute_uv=False))  # [1.e+00 1.e-09]
assert np.linalg.matrix_rank(near, tol=1e-8) == 1
assert np.linalg.matrix_rank(near, tol=1e-10) == 2
```

Ma trận trên có rank đại số 2. Với `tol=1e-8`, ta xem hướng $e_2$ là gần kernel,
vì `near @ e2` có chuẩn $10^{-9}$. Ngưỡng `tol` là tuyệt đối trên scale của
singular value. Khi đổi đơn vị hoặc nhân cả ma trận với một hằng số, xem lại
ngưỡng hoặc dùng quy tắc theo $s_{\max}$. Trong ví dụ học tập, hãy in cả `s`
và `tol` để giải thích vì sao hai phép đo rank cho kết quả khác nhau.

## 7. Hai ứng dụng trong ML

**Hồi quy có feature phụ thuộc.** Cho $X\in\mathbb R^{N\times d}$ và giả sử
cột thứ hai bằng hai lần cột thứ nhất. Khi ấy $(2,-1,0,\ldots,0)^T\in\ker(X)$.
Cộng bội bất kỳ của hướng này vào trọng số giữ nguyên dự đoán $Xw$ trên bảng đang
xét. Nếu quan hệ giữa hai feature chỉ đúng trên train, hai bộ trọng số có thể
cho dự đoán khác nhau trên một mẫu mới. Khi phân tích mô hình có intercept,
rank được tính trên design matrix đã thêm cột toàn 1.

**Biểu diễn hạng thấp.** Nếu ma trận điểm $R$ được xấp xỉ bằng $UV$, với $U$
shape `(users, k)` và $V$ shape `(k, movies)`, mỗi hàng điểm nằm trong span của
$k$ hàng của $V$. Rank của $UV$ vì vậy tối đa là $k$. Phần ML in Practice dùng
ý tưởng này để tái tạo điểm và chọn phim chưa xem.

## Lỗi thường gặp

- **Dùng $m-r$ cho nullity của $A$.** Đếm biến đầu vào, tức số cột $n$.
- **Lấy cột pivot từ ma trận đã khử.** Dùng chỉ số pivot để lấy cột từ ma trận gốc.
- **Dùng `Vt[:, r:]` làm null basis.** `Vt` lưu hướng theo hàng, nên lấy
  `Vt[r:, :].T` nếu muốn xếp các vector basis thành cột.
- **Dùng SVD rút gọn để lấy toàn bộ kernel của ma trận rộng.** Kiểm tra số cột
  của basis so với $n-r$.
- **Đọc `residuals` rỗng từ `lstsq` thành phần dư bằng 0.** Tính trực tiếp
  `A @ x - b` rồi so chuẩn với ngưỡng phù hợp.
- **Dùng `det` để đo rank của ma trận chữ nhật.** Dùng `matrix_rank` hoặc đếm pivot.

## Quy trình kiểm tra bằng NumPy

1. Ghi `A.shape == (m, n)` và xác định không gian của vector đang xét.
2. Dự đoán rank từ quan hệ giữa cột hoặc hàng, rồi kiểm tra bằng `matrix_rank`.
3. Xem singular value nếu dữ liệu gần phụ thuộc. Ghi rõ tolerance đang dùng.
4. Lấy basis từ SVD đầy đủ khi cần kernel, rồi kiểm tra shape và phần dư.
5. Với $Ax=b$, dùng `lstsq` nếu cần xấp xỉ. Kiểm tra phần dư để biết $b$ có
   thuộc range trong sai số cho phép và dùng nullity để đọc tính duy nhất.

## Tự kiểm tra

1. Ma trận `(3, 7)` có rank 2. Range, row space, null space và left null space
   nằm ở đâu và có số chiều bao nhiêu?
2. Nếu hai cột bằng nhau, hãy viết một hướng kernel liên quan đến hai cột đó.
3. Tại sao các cột pivot phải lấy từ $A$ gốc?
4. Một ma trận `(5, 3)` có rank 3. Mọi $b\in\mathbb R^5$ có nghiệm chính xác không?
   Least squares có bao nhiêu nghiệm?
5. Một ma trận `(3, 5)` có rank 3. Có bao nhiêu biến tự do nếu $Ax=b$?
6. Tại sao `full_matrices=False` có thể làm thiếu hướng kernel khi $n>m$?
7. Với `s = [10, 0.01, 1e-7]`, rank là bao nhiêu ở `tol=1e-6` và `tol=1e-8`?
8. Nếu một hướng kernel của design matrix train không còn là kernel trên test,
   điều đó ảnh hưởng thế nào đến dự đoán khi thay đổi trọng số theo hướng đó?

<details>
<summary>Đáp án ngắn</summary>

1. Range và left null space nằm trong $\mathbb R^3$, có số chiều 2 và 1.
   Row space và null space nằm trong $\mathbb R^7$, có số chiều 2 và 5.
2. $e_i-e_j$ cho hai cột bằng nhau ở vị trí $i$ và $j$.
3. Khử hàng giữ quan hệ phụ thuộc giữa cột nhưng có thể đổi không gian cột.
4. Chỉ các $b$ trong range mới có nghiệm chính xác. Least squares có nghiệm duy nhất cho mọi $b$.
5. Hai biến tự do. Mọi $b\in\mathbb R^3$ đều đạt được vì full row rank.
6. `Vt` rút gọn chỉ có $m$ hàng, trong khi basis đầy đủ phía đầu vào cần $n$ hướng.
7. Lần lượt là 2 và 3.
8. Dự đoán train giữ nguyên. Trên test, thành phần thay đổi là $X_{\text{test}}\Delta w$.

</details>

## Checklist

- [ ] Tôi xác định được shape và số chiều của bốn không gian con.
- [ ] Tôi tìm được basis bằng khử hàng và kiểm chứng bằng NumPy.
- [ ] Tôi phân biệt được full row rank và full column rank.
- [ ] Tôi dùng range và nullity để phân loại nghiệm của $Ax=b$.
- [ ] Tôi lấy đủ hướng kernel từ SVD và giải thích được tác dụng của tolerance.
- [ ] Tôi liên hệ được kernel với feature dư và rank thấp với hệ gợi ý.
- [ ] Tôi hoàn thành notebook, gồm bài tổng hợp dự đoán điểm cho người dùng mới.
