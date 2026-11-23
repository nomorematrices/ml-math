# Bài 03 — Ma trận đơn vị, Nghịch đảo và Ma trận đặc biệt

Tài liệu đi kèm: [thực hành NumPy](practice.ipynb) và
[liên hệ với Machine Learning trong thực tế](ml-in-practice.md).

## Mục tiêu

Sau bài này, bạn có thể:

- Nhận biết và tạo identity, diagonal, triangular, symmetric và permutation matrix.
- Giải thích nghịch đảo là phép biến đổi “hoàn tác” và nêu đúng điều kiện tồn tại.
- Phân biệt ma trận khả nghịch, suy biến và gần suy biến.
- Giải một hay nhiều hệ tuyến tính bằng `np.linalg.solve` và kiểm tra phần dư.
- Giải thích vì sao không nên tính `inv(A) @ b` chỉ để giải $A\mathbf{x}=\mathbf{b}$.
- Dùng `np.linalg.cond`, `np.linalg.lstsq` và `np.linalg.pinv` đúng ngữ cảnh cơ bản.
- Liên hệ ma trận đặc biệt với hồi quy tuyến tính, regularization và xáo trộn dữ liệu.

## 1. Ma trận đơn vị: phần tử trung hòa của phép nhân

Ma trận đơn vị cấp $n$ là ma trận vuông có số 1 trên đường chéo chính và 0 ở nơi khác:

$$
I_n=
\begin{bmatrix}
1&0&\cdots&0\\
0&1&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&1
\end{bmatrix}.
$$

Với $A\in\mathbb{R}^{m\times n}$:

$$
I_mA=A,
\qquad
AI_n=A.
$$

Kích thước của hai ma trận đơn vị không nhất thiết giống nhau. Ở bên trái cần $I_m$ để giữ
nguyên $m$ hàng; ở bên phải cần $I_n$ để giữ nguyên $n$ cột.

```python
A = np.arange(6).reshape(2, 3)
left_identity = np.eye(2)
right_identity = np.eye(3)

np.testing.assert_allclose(left_identity @ A, A)
np.testing.assert_allclose(A @ right_identity, A)
```

Trong ML, $I$ xuất hiện trực tiếp trong Ridge regression, phép cộng “jitter” để ổn định số
học và nhiều công thức regularization.

## 2. Nghịch đảo: phép biến đổi hoàn tác

Với ma trận vuông $A\in\mathbb{R}^{n\times n}$, nếu tồn tại $A^{-1}$ sao cho

$$
AA^{-1}=A^{-1}A=I_n,
$$

thì $A$ được gọi là **khả nghịch**. Có thể hiểu $A$ thực hiện một phép biến đổi tuyến tính,
còn $A^{-1}$ hoàn tác phép biến đổi đó:

$$
\mathbf{y}=A\mathbf{x}
\quad\Longrightarrow\quad
\mathbf{x}=A^{-1}\mathbf{y}.
$$

Không phải ma trận vuông nào cũng khả nghịch. Ma trận không có nghịch đảo gọi là **suy
biến** (singular). Một ma trận không vuông không có nghịch đảo hai phía theo định nghĩa
thông thường, dù có thể có giả nghịch đảo ở Phần 11.

### 2.1. Nghịch đảo của ma trận $2\times2$

Với

$$
A=\begin{bmatrix}a&b\\c&d\end{bmatrix},
$$

nếu $ad-bc\ne0$ thì

$$
A^{-1}=\frac{1}{ad-bc}
\begin{bmatrix}
d&-b\\-c&a
\end{bmatrix}.
$$

Đại lượng $ad-bc$ là định thức, sẽ được học kỹ ở Bài 04. Ở đây chỉ cần ghi nhớ: nếu nó
bằng 0, công thức phải chia cho 0 và nghịch đảo không tồn tại.

Ví dụ:

$$
A=\begin{bmatrix}2&1\\1&1\end{bmatrix},
\qquad
A^{-1}=\begin{bmatrix}1&-1\\-1&2\end{bmatrix}.
$$

Khi kiểm tra bằng số thực dấu phẩy động, dùng so sánh gần đúng:

```python
A_inv = np.linalg.inv(A)
np.testing.assert_allclose(A @ A_inv, np.eye(A.shape[0]))
```

Không nên kiểm tra bằng `(A @ A_inv == I).all()` vì sai số làm tròn có thể khiến các giá trị
đáng lẽ bằng 0 lại thành một số rất nhỏ.

## 3. Các tính chất quan trọng của nghịch đảo

Với các ma trận vuông khả nghịch, có cùng kích thước:

$$
(A^{-1})^{-1}=A,
\qquad
(A^T)^{-1}=(A^{-1})^T,
$$

và

$$
(AB)^{-1}=B^{-1}A^{-1}.
$$

Thứ tự bị đảo vì muốn hoàn tác $AB$ phải hoàn tác phép biến đổi tác động sau cùng trước:

$$
(AB)(B^{-1}A^{-1})
=A(BB^{-1})A^{-1}
=I.
$$

Nói chung:

$$
(A+B)^{-1}\ne A^{-1}+B^{-1}.
$$

Không được “phân phối” phép nghịch đảo qua phép cộng.

## 4. Giải hệ tuyến tính: dùng `solve`, không tạo nghịch đảo

Xét hệ:

$$
A\mathbf{x}=\mathbf{b},
$$

với $A\in\mathbb{R}^{n\times n}$ và $\mathbf{b}\in\mathbb{R}^n$. Nếu $A$ khả nghịch,
nghiệm duy nhất về mặt toán học là

$$
\mathbf{x}=A^{-1}\mathbf{b}.
$$

Trong code, nên viết:

```python
x = np.linalg.solve(A, b)
```

thay vì:

```python
x = np.linalg.inv(A) @ b
```

`solve` thể hiện đúng ý định, thường nhanh hơn và tránh tạo toàn bộ nghịch đảo không cần
thiết. Nghịch đảo tường minh còn thêm sai số làm tròn trước bước nhân với $b$.

Luồng shape:

```text
A: (n, n)
b: (n,)
x: (n,)
```

Nếu có $k$ vế phải, xếp chúng thành các cột của $B\in\mathbb{R}^{n\times k}$:

```text
A: (n, n)
B: (n, k)
X = solve(A, B): (n, k)
```

Mỗi cột của $X$ là nghiệm ứng với cột cùng vị trí trong $B$.

### Kiểm tra nghiệm bằng phần dư

Phần dư của nghiệm $\hat{\mathbf{x}}$ là:

$$
\mathbf{r}=A\hat{\mathbf{x}}-\mathbf{b}.
$$

Với một hệ được giải chính xác về mặt số học, $\|\mathbf{r}\|_2$ phải rất nhỏ:

```python
residual = A @ x - b
assert np.linalg.norm(residual) < 1e-10
```

Phần dư nhỏ xác nhận nghiệm thỏa hệ đã cho; nó không tự động bảo đảm bài toán có độ điều
kiện tốt.

## 5. Ma trận chéo

Ma trận chéo chỉ có thể khác 0 trên đường chéo chính:

$$
D=\operatorname{diag}(d_1,\ldots,d_n).
$$

Nhân $D\mathbf{x}$ scale từng tọa độ độc lập:

$$
D\mathbf{x}=
\begin{bmatrix}
d_1x_1\\ \vdots\\ d_nx_n
\end{bmatrix}.
$$

Nếu mọi $d_i\ne0$, nghịch đảo rất đơn giản:

$$
D^{-1}=\operatorname{diag}
\left(\frac1{d_1},\ldots,\frac1{d_n}\right).
$$

Trong NumPy, `np.diag(v)` tạo ma trận chéo từ vector 1D, còn `np.diag(D)` trích đường chéo
từ ma trận 2D:

```python
scales = np.array([2.0, 0.5, -1.0])
D = np.diag(scales)
recovered_scales = np.diag(D)
```

Không cần tạo `D` nếu chỉ muốn scale từng feature: với `X.shape == (N, d)`, `X * scales`
thường rõ nghĩa và tiết kiệm hơn `X @ D`.

## 6. Ma trận tam giác

Ma trận tam giác trên có mọi phần tử dưới đường chéo bằng 0:

$$
U=
\begin{bmatrix}
u_{11}&u_{12}&\cdots&u_{1n}\\
0&u_{22}&\cdots&u_{2n}\\
\vdots&\ddots&\ddots&\vdots\\
0&\cdots&0&u_{nn}
\end{bmatrix}.
$$

Ma trận tam giác dưới được định nghĩa tương tự với các phần tử phía trên đường chéo bằng 0.
Hệ tam giác có thể giải hiệu quả bằng thế lùi hoặc thế tiến. Các phân rã ma trận thường
biến một hệ tổng quát thành các hệ tam giác dễ giải hơn.

```python
upper = np.triu(A)
lower = np.tril(A)
```

Một ma trận tam giác khả nghịch khi mọi phần tử trên đường chéo khác 0. Nội dung này sẽ xuất
hiện lại trong phân rã Cholesky ở Bài 11.

## 7. Ma trận đối xứng

Ma trận vuông $S$ đối xứng nếu:

$$
S=S^T.
$$

Trong tính toán số, nên kiểm tra bằng:

```python
np.allclose(S, S.T)
```

Các ma trận quan trọng trong ML thường đối xứng:

- Gram matrix $X^TX$;
- covariance matrix;
- Hessian của một hàm đủ trơn;
- $A^TA$ với bất kỳ ma trận thực $A$.

Ta có thể tạo phần đối xứng của một ma trận vuông bằng:

$$
S=\frac{A+A^T}{2}.
$$

Đối xứng không đồng nghĩa với khả nghịch. Chẳng hạn ma trận toàn số 1 là đối xứng nhưng có
các hàng trùng nhau nên suy biến.

## 8. Ma trận hoán vị

Ma trận hoán vị $P$ được tạo bằng cách hoán vị các hàng của ma trận đơn vị. Mỗi hàng và mỗi
cột có đúng một số 1. Nhân bên trái đổi thứ tự hàng; nhân bên phải đổi thứ tự cột:

$$
PA \quad\text{hoán vị hàng của }A,
\qquad
AP \quad\text{hoán vị cột của }A.
$$

Ma trận hoán vị thỏa:

$$
P^{-1}=P^T.
$$

Trong code thực tế, indexing thường đơn giản và rẻ hơn việc tạo ma trận $P$ dày đặc:

```python
order = np.array([2, 0, 1])
rows_reordered = A[order]

P = np.eye(3)[order]
np.testing.assert_allclose(P @ A, rows_reordered)
```

Khi shuffle dataset, phải áp dụng cùng một `order` cho `X` và `y` để không phá vỡ cặp
feature–label.

## 9. Suy biến và hệ không có nghiệm duy nhất

Nếu các hàng hoặc cột của $A$ phụ thuộc tuyến tính, $A$ suy biến. Ví dụ:

$$
A=
\begin{bmatrix}
1&2\\2&4
\end{bmatrix}.
$$

Hàng thứ hai bằng hai lần hàng thứ nhất, nên hai phương trình không cung cấp hai ràng buộc
độc lập.

- Với $\mathbf{b}=[3,6]^T$, hệ có vô số nghiệm.
- Với $\mathbf{b}=[3,7]^T$, hệ vô nghiệm vì hai phương trình mâu thuẫn.

Trong cả hai trường hợp, `np.linalg.solve(A, b)` thường báo `LinAlgError: Singular matrix`.
Thông báo này không cho biết hệ thuộc trường hợp vô số nghiệm hay vô nghiệm; muốn phân tích
sâu hơn cần dùng rank hoặc least squares.

## 10. Ma trận gần suy biến và độ điều kiện

Một ma trận có thể khả nghịch về mặt toán học nhưng rất nhạy với sai số dữ liệu. Số điều
kiện đo mức khuếch đại sai số tương đối trong trường hợp xấu nhất:

```python
kappa = np.linalg.cond(A)
```

- $\kappa(A)\ge1$ với norm 2;
- gần 1 thường là điều kiện tốt;
- rất lớn nghĩa là bài toán nhạy;
- `inf` thường báo hiệu suy biến theo độ chính xác đang dùng.

Ví dụ:

$$
A_\varepsilon=
\begin{bmatrix}
1&1\\1&1+\varepsilon
\end{bmatrix}.
$$

Khi $\varepsilon$ rất nhỏ, hai hàng gần trùng nhau. Một thay đổi rất nhỏ trong $b$ có thể
làm nghiệm thay đổi lớn dù phần dư vẫn nhỏ. Vì vậy:

> “Giải được” không đồng nghĩa với “nghiệm ổn định”.

Không có một ngưỡng condition number đúng cho mọi bài toán; phải xét `dtype`, độ nhiễu và
độ chính xác ứng dụng cần.

## 11. Least squares và giả nghịch đảo

Với $A\in\mathbb{R}^{m\times n}$ không vuông hoặc hệ không có nghiệm chính xác, ta thường
tìm nghiệm least squares:

$$
\hat{\mathbf{x}}
=\arg\min_{\mathbf{x}}\|A\mathbf{x}-\mathbf{b}\|_2^2.
$$

NumPy cung cấp:

```python
x_lstsq, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
```

Giả nghịch đảo Moore–Penrose $A^+$ mở rộng ý tưởng nghịch đảo cho ma trận chữ nhật hoặc suy
biến:

```python
x_pinv = np.linalg.pinv(A) @ b
```

Với một bài toán least squares thông thường, nên ưu tiên `lstsq(A, b)` vì nó nói rõ mục đích
và không cần tạo toàn bộ giả nghịch đảo. `pinv` hữu ích khi thật sự cần toán tử giả nghịch
đảo hoặc cần áp dụng nó cho nhiều bước sau.

## 12. Bảng tổng hợp ma trận đặc biệt

| Loại ma trận | Dấu hiệu nhận biết | Tính chất/ứng dụng chính | NumPy |
|---|---|---|---|
| Đơn vị | đường chéo bằng 1, còn lại 0 | phần tử trung hòa; regularization | `np.eye(n)` |
| Chéo | ngoài đường chéo bằng 0 | scale từng tọa độ độc lập | `np.diag(v)` |
| Tam giác trên | dưới đường chéo bằng 0 | giải bằng thế lùi | `np.triu(A)` |
| Tam giác dưới | trên đường chéo bằng 0 | giải bằng thế tiến | `np.tril(A)` |
| Đối xứng | $S=S^T$ | Gram, covariance, Hessian | `np.allclose(S, S.T)` |
| Hoán vị | mỗi hàng/cột có một số 1 | đổi thứ tự hàng/cột | `np.eye(n)[order]` |

## 13. Lỗi thường gặp

### Lỗi 1 — Tính nghịch đảo chỉ để giải hệ

Dùng `np.linalg.solve(A, b)` thay cho `np.linalg.inv(A) @ b`.

### Lỗi 2 — Kiểm tra số thực bằng `==`

Dùng `np.allclose` hoặc `np.testing.assert_allclose` khi kiểm tra $AA^{-1}\approx I$ và
$A\mathbf{x}\approx\mathbf{b}$.

### Lỗi 3 — Nghĩ rằng ma trận vuông luôn khả nghịch

Vuông là điều kiện cần, không phải điều kiện đủ. Các hàng/cột phụ thuộc làm ma trận suy biến.

### Lỗi 4 — Dựa duy nhất vào `det(A) == 0`

Tính định thức rồi so sánh đúng bằng 0 không phải cách đáng tin cậy để quyết định ổn định số.
Condition number và cách giải/phân rã phù hợp cung cấp nhiều thông tin hơn.

### Lỗi 5 — Nhầm phần dư nhỏ với nghiệm đáng tin cậy

Ma trận gần suy biến vẫn có thể cho phần dư rất nhỏ nhưng nghiệm cực kỳ nhạy với nhiễu.

### Lỗi 6 — Shuffle `X` mà không shuffle `y`

Dùng cùng một permutation index cho feature và label.

### Lỗi 7 — Tạo ma trận chéo/hoán vị lớn không cần thiết

Ưu tiên broadcasting để scale và indexing để hoán vị khi không thật sự cần biểu diễn ma
trận tường minh.

## 14. Quy trình giải và kiểm tra một hệ tuyến tính

1. Ghi shape và ý nghĩa của $A$, $b$ và nghiệm cần tìm.
2. Nếu $A$ vuông và kỳ vọng khả nghịch, dùng `np.linalg.solve`.
3. Nếu $A$ chữ nhật hoặc cần nghiệm xấp xỉ, dùng `np.linalg.lstsq`.
4. Kiểm tra `np.isfinite` trên dữ liệu đầu vào.
5. Tính phần dư $A\hat{x}-b$ và norm của nó.
6. Nếu nghiệm bất thường, xem `np.linalg.cond(A)` và kiểm tra feature phụ thuộc/gần phụ thuộc.
7. Với bài toán ML, cân nhắc chuẩn hóa đặc trưng hoặc regularization thay vì cố tính nghịch
   đảo.

## Tự kiểm tra

1. Với $A\in\mathbb{R}^{4\times7}$, cần ma trận đơn vị kích thước nào ở bên trái và bên phải?
2. Ma trận vuông có hai hàng giống nhau có thể khả nghịch không? Vì sao?
3. Vì sao `solve(A, b)` thường tốt hơn `inv(A) @ b`?
4. Shape của nghiệm khi `A.shape == (5, 5)` và `B.shape == (5, 3)` là gì?
5. Tại sao $(AB)^{-1}=B^{-1}A^{-1}$ phải đảo thứ tự?
6. Nhân ma trận chéo bên phải của `X(N, d)` có tác dụng gì lên các cột?
7. `P @ A` và `A @ P` lần lượt hoán vị gì?
8. Ma trận đối xứng có nhất thiết khả nghịch không? Cho một phản ví dụ.
9. Phần dư rất nhỏ có đủ để kết luận nghiệm ổn định không?
10. Khi nào nên chuyển từ `solve` sang `lstsq`?
11. `np.linalg.pinv(A) @ b` khác `np.linalg.inv(A) @ b` về phạm vi áp dụng như thế nào?
12. Vì sao phải dùng cùng một permutation cho `X` và `y`?

## Checklist

- [ ] Tôi tạo đúng identity và chọn đúng kích thước khi nhân bên trái/bên phải.
- [ ] Tôi giải thích được nghịch đảo như một phép biến đổi hoàn tác.
- [ ] Tôi kiểm tra $AA^{-1}\approx I$ bằng so sánh gần đúng.
- [ ] Tôi dùng `solve` thay cho nghịch đảo tường minh khi giải hệ vuông.
- [ ] Tôi giải được nhiều vế phải trong một lần gọi `solve`.
- [ ] Tôi nhận biết diagonal, triangular, symmetric và permutation matrix.
- [ ] Tôi phân biệt được suy biến với gần suy biến.
- [ ] Tôi tính và diễn giải được residual norm và condition number.
- [ ] Tôi biết khi nào dùng `lstsq` hoặc `pinv`.
- [ ] Tôi hoàn thành notebook, gồm bài tổng hợp hồi quy tuyến tính có regularization.

