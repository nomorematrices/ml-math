# Bài 02 — Tích vô hướng, nhân ma trận và tích Hadamard

Tài liệu đi kèm: [thực hành NumPy](practice.ipynb) và
[liên hệ với Machine Learning trong thực tế](ml-in-practice.md).

## Mục tiêu

Sau bài này, bạn có thể:

- phân biệt dot product, matrix-vector product, matrix product và Hadamard product;
- suy luận điều kiện tồn tại và shape của kết quả trước khi tính;
- giải thích phép nhân ma trận bằng hai góc nhìn: hàng nhân cột và tổ hợp tuyến tính;
- dùng đúng `@`, `np.matmul`, `np.dot` và `*` trong NumPy;
- xử lý đúng vector 1D `(n,)`, vector hàng `(1, n)` và vector cột `(n, 1)`;
- cài đặt một linear layer cho một mẫu và cho cả batch;
- nhận ra các lỗi shape, broadcasting và thứ tự nhân thường gặp trong code ML.

## 1. Bốn phép nhân dễ bị nhầm

### 1.1. Nhân vô hướng

Với scalar $c$ và ma trận $\mathbf{A}$, phép nhân $c\mathbf{A}$ nhân mọi phần tử của $\mathbf{A}$ với $c$. Shape không đổi.

$$
2\begin{bmatrix}1&-1\\3&0\end{bmatrix}
=
\begin{bmatrix}2&-2\\6&0\end{bmatrix}.
$$

Trong NumPy: `2 * A`.

### 1.2. Tích vô hướng — dot product

Với $\mathbf{x},\mathbf{y}\in\mathbb{R}^n$:

$$
\mathbf{x}^T\mathbf{y}=\sum_{k=1}^{n}x_k y_k.
$$

Kết quả là một scalar. Ví dụ:

$$
\begin{bmatrix}1&2&3\end{bmatrix}
\begin{bmatrix}4\\-1\\2\end{bmatrix}
=1\cdot4+2\cdot(-1)+3\cdot2=8.
$$

Trong NumPy, với hai mảng 1D cùng độ dài, `x @ y` và `np.dot(x, y)` đều trả về scalar.

Dot product còn đo mức độ cùng hướng:

$$
\mathbf{x}^T\mathbf{y}=\|\mathbf{x}\|_2\|\mathbf{y}\|_2\cos\theta.
$$

Do đó cosine similarity là:

$$
\cos\theta=
\frac{\mathbf{x}^T\mathbf{y}}
{\|\mathbf{x}\|_2\|\mathbf{y}\|_2},
$$

với điều kiện cả hai vector khác vector không.

### 1.3. Nhân ma trận — matrix product

Nếu $\mathbf{A}\in\mathbb{R}^{m\times n}$ và $\mathbf{B}\in\mathbb{R}^{n\times p}$ thì:

$$
\mathbf{C}=\mathbf{A}\mathbf{B}\in\mathbb{R}^{m\times p},
\qquad
C_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}.
$$

Quy tắc shape:

$$
(m,n)@(n,p)\longrightarrow(m,p).
$$

Hai kích thước ở giữa phải bằng nhau; hai kích thước ở ngoài tạo nên shape kết quả. Trong NumPy nên dùng `A @ B`.

Ví dụ:

$$
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix}
\begin{bmatrix}
1&2\\
0&1\\
-1&3
\end{bmatrix}
=
\begin{bmatrix}
-2&13\\
-2&31
\end{bmatrix}.
$$

Phần tử ở hàng 1, cột 2 là dot product giữa hàng 1 của $\mathbf{A}$ và cột 2 của $\mathbf{B}$:

$$
C_{12}=[1,2,3]\cdot[2,1,3]=13.
$$

### 1.4. Tích Hadamard — element-wise product

Với hai ma trận cùng shape:

$$
(\mathbf{A}\odot\mathbf{B})_{ij}=A_{ij}B_{ij}.
$$

Trong NumPy: `A * B`. Đây không phải nhân ma trận. NumPy còn cho phép `*` với các shape broadcast được, vì vậy code có thể chạy nhưng mang ý nghĩa khác với điều bạn muốn.

| Phép toán | Công thức/NumPy | Điều kiện chính | Kết quả |
|---|---|---|---|
| Nhân scalar | `c * A` | `c` là scalar | cùng shape với `A` |
| Dot product | `x @ y` | hai vector cùng độ dài | scalar |
| Matrix product | `A @ B` | `A.shape[-1] == B.shape[-2]` | các trục ngoài |
| Hadamard | `A * B` | cùng shape hoặc broadcast được | shape sau broadcasting |

## 2. Ba cách hiểu phép nhân ma trận

### 2.1. Mỗi phần tử là “hàng nhân cột”

Để tính $C_{ij}$, lấy hàng $i$ của $\mathbf{A}$ dot với cột $j$ của $\mathbf{B}$. Đây là cách phù hợp khi tính tay một phần tử.

### 2.2. Mỗi cột kết quả là một tổ hợp tuyến tính

Viết $\mathbf{B}$ theo từng cột $\mathbf{b}_j$:

$$
\mathbf{A}\mathbf{B}
=
\begin{bmatrix}
\mathbf{A}\mathbf{b}_1 & \cdots & \mathbf{A}\mathbf{b}_p
\end{bmatrix}.
$$

Nếu $\mathbf{b}_j=(b_{1j},\ldots,b_{nj})^T$ thì:

$$
\mathbf{A}\mathbf{b}_j
=b_{1j}\mathbf{a}_1+\cdots+b_{nj}\mathbf{a}_n,
$$

trong đó $\mathbf{a}_k$ là cột thứ $k$ của $\mathbf{A}$. Vì vậy, mỗi cột của $\mathbf{A}\mathbf{B}$ nằm trong span của các cột của $\mathbf{A}$.

### 2.3. Hợp thành các phép biến đổi

Nếu $\mathbf{B}$ biến $\mathbf{x}$ thành $\mathbf{B}\mathbf{x}$ và sau đó $\mathbf{A}$ biến kết quả thành $\mathbf{A}(\mathbf{B}\mathbf{x})$, phép biến đổi tổng hợp là $\mathbf{A}\mathbf{B}$:

$$
\mathbf{A}(\mathbf{B}\mathbf{x})=(\mathbf{A}\mathbf{B})\mathbf{x}.
$$

Thứ tự đọc từ phải sang trái: $\mathbf{B}$ tác động trước, $\mathbf{A}$ tác động sau.

## 3. Tính chất quan trọng

### 3.1. Kết hợp

Khi các shape tương thích:

$$
(\mathbf{A}\mathbf{B})\mathbf{C}
=\mathbf{A}(\mathbf{B}\mathbf{C}).
$$

Hai vế bằng nhau về toán học, nhưng trong tính toán chúng có thể khác đáng kể về chi phí. Ví dụ:

- $\mathbf{A}$ có shape `(1000, 10)`;
- $\mathbf{B}$ có shape `(10, 1000)`;
- $\mathbf{x}$ có shape `(1000,)`.

Tính `(A @ B) @ x` tạo ma trận trung gian `(1000, 1000)`. Tính `A @ (B @ x)` chỉ tạo vector trung gian `(10,)` và rẻ hơn rất nhiều.

### 3.2. Không giao hoán

Nói chung:

$$
\mathbf{A}\mathbf{B}\ne\mathbf{B}\mathbf{A}.
$$

Một trong hai tích thậm chí có thể không tồn tại. Nếu cả hai tồn tại, chúng có thể khác shape hoặc khác giá trị. Không được đổi thứ tự nhân chỉ để “sửa” lỗi shape.

### 3.3. Phân phối

$$
\mathbf{A}(\mathbf{B}+\mathbf{C})
=\mathbf{A}\mathbf{B}+\mathbf{A}\mathbf{C}.
$$

### 3.4. Chuyển vị của một tích

$$
(\mathbf{A}\mathbf{B})^T
=\mathbf{B}^T\mathbf{A}^T.
$$

Khi chuyển vị một tích, thứ tự các thừa số bị đảo ngược.

### 3.5. Ma trận đơn vị

$$
\mathbf{I}_m\mathbf{A}=\mathbf{A},
\qquad
\mathbf{A}\mathbf{I}_n=\mathbf{A}
$$

với $\mathbf{A}\in\mathbb{R}^{m\times n}$.

## 4. Matrix-vector product và các góc nhìn tương đương

Với $\mathbf{A}\in\mathbb{R}^{m\times n}$ và $\mathbf{x}\in\mathbb{R}^n$:

$$
\mathbf{y}=\mathbf{A}\mathbf{x}\in\mathbb{R}^m.
$$

- Góc nhìn theo hàng: $y_i$ là dot product giữa hàng $i$ của $\mathbf{A}$ và $\mathbf{x}$.
- Góc nhìn theo cột: $\mathbf{y}=x_1\mathbf{a}_1+\cdots+x_n\mathbf{a}_n$.
- Góc nhìn ánh xạ: $\mathbf{A}$ biến một vector trong $\mathbb{R}^n$ thành một vector trong $\mathbb{R}^m$.

Ba góc nhìn mô tả cùng một phép toán, nhưng mỗi góc giúp giải một loại câu hỏi khác nhau.

## 5. Shape trong NumPy: điểm khác với ký hiệu toán học

Trong toán học, $\mathbf{x}$ thường ngầm hiểu là vector cột. Trong NumPy, mảng 1D `x.shape == (n,)` không có hướng hàng hay cột.

Với `A.shape == (m, n)` và `x.shape == (n,)`:

- `A @ x` có shape `(m,)`;
- `x @ A.T` cũng có shape `(m,)`;
- `x.reshape(n, 1)` là vector cột, và `A @ x[:, None]` có shape `(m, 1)`;
- `x.reshape(1, n)` là vector hàng, và `x[None, :] @ A.T` có shape `(1, m)`.

Hai biểu thức có thể chứa cùng giá trị nhưng khác shape. Điều đó ảnh hưởng tới concatenation và broadcasting ở bước sau.

Một trường hợp đặc biệt:

- `(n,) @ (n,) -> scalar`;
- `(n, 1) @ (1, n) -> (n, n)`, đây là outer product;
- `(1, n) @ (n, 1) -> (1, 1)`, đây là ma trận chứa dot product.

Trong NumPy có thể dùng `np.outer(x, y)` để tạo outer product một cách rõ nghĩa.

## 6. `@`, `np.matmul`, `np.dot` và `*`

- `A @ B` là cú pháp nên ưu tiên cho matrix product; ý định rõ ràng nhất.
- `np.matmul(A, B)` có cùng quy tắc cốt lõi với `@`.
- `np.dot(x, y)` dễ hiểu với hai vector 1D nhưng có quy tắc khó nhớ hơn cho tensor nhiều chiều.
- `A * B` là element-wise multiplication có broadcasting.

Trong bài này, dùng `@` cho các phép nhân tuyến tính và `*` cho phép nhân từng phần tử.

Với tensor batch, `@`/`np.matmul` nhân trên hai trục cuối và broadcast các trục đứng trước. Ví dụ:

```python
A.shape == (32, 4, 3)
B.shape == (32, 3, 2)
(A @ B).shape == (32, 4, 2)
```

Tức là thực hiện 32 phép nhân `(4, 3) @ (3, 2)`.

## 7. Linear layer trong Machine Learning

### 7.1. Một mẫu

Với vector đặc trưng $\mathbf{x}\in\mathbb{R}^{d}$, ma trận trọng số $\mathbf{W}\in\mathbb{R}^{d\times h}$ và bias $\mathbf{b}\in\mathbb{R}^{h}$:

$$
\mathbf{z}=\mathbf{x}^T\mathbf{W}+\mathbf{b}.
$$

Theo quy ước NumPy dùng vector 1D:

```python
z = x @ W + b
```

Shape: `(d,) @ (d, h) + (h,) -> (h,)`.

### 7.2. Cả batch

Xếp $N$ mẫu thành các hàng của $\mathbf{X}\in\mathbb{R}^{N\times d}$:

$$
\mathbf{Z}=\mathbf{X}\mathbf{W}+\mathbf{b}.
$$

Shape trong code:

```text
X: (N, d)
W: (d, h)
b: (h,)
Z: (N, h)
```

`X @ W` cho `(N, h)`. Bias `(h,)` được broadcast qua trục batch: cùng một bias được cộng vào mỗi hàng. Đây là broadcasting có chủ đích.

Nếu thư viện lưu trọng số theo `(out_features, in_features)`, công thức có thể là `X @ W.T + b`. Luôn đọc quy ước shape thay vì ghi nhớ máy móc một biểu thức.

## 8. Ví dụ suy luận shape trước khi chạy

Cho:

```text
X: (64, 128)   # 64 mẫu, 128 đặc trưng
W1: (128, 32)
b1: (32,)
W2: (32, 10)
b2: (10,)
```

Khi đó:

```text
H = X @ W1 + b1   -> (64, 32)
Y = H @ W2 + b2   -> (64, 10)
```

Trục batch `64` được giữ nguyên; mỗi layer thay đổi số đặc trưng ở trục cuối.

## 9. Các lỗi thường gặp

### Lỗi 1 — Dùng `*` thay cho `@`

`X * W` không mô tả linear layer. Nó có thể báo lỗi shape hoặc tệ hơn, chạy nhờ broadcasting nhưng cho phép toán sai.

### Lỗi 2 — Chỉ nhìn tổng số phần tử

Hai ma trận có cùng `size` chưa chắc nhân được. Matrix product kiểm tra các trục cụ thể, không kiểm tra tổng số phần tử.

### Lỗi 3 — Nhầm `(n,)` với `(n, 1)`

`.T` không làm thay đổi vector 1D. Muốn vector cột, dùng `x[:, None]` hoặc `x.reshape(-1, 1)`.

### Lỗi 4 — Bias có shape sai

Với output `(N, h)`, bias `(h,)` hoặc `(1, h)` là phù hợp. Bias `(N,)` thường không broadcast được, hoặc vô tình gắn một bias cho mỗi mẫu thay vì mỗi output feature.

### Lỗi 5 — Đổi thứ tự để hết lỗi

`A @ B` và `B @ A` biểu diễn các phép biến đổi khác nhau. Việc `B @ A` chạy không có nghĩa đó là cách sửa đúng.

### Lỗi 6 — Tạo ma trận trung gian quá lớn

Nhờ tính kết hợp, hãy chọn thứ tự ngoặc giúp giảm kích thước trung gian. Điều này không đổi kết quả toán học nhưng có thể giảm mạnh thời gian và bộ nhớ.

## 10. Quy trình debug shape

Khi gặp lỗi matrix multiplication:

1. ghi shape và ý nghĩa từng trục của mọi biến;
2. viết phép toán shape, ví dụ `(N, d) @ (d, h) -> (N, h)`;
3. kiểm tra hai kích thước ở giữa;
4. kiểm tra có vô tình mất trục do indexing hay không;
5. kiểm tra bias theo quy tắc broadcasting từ phải sang trái;
6. chỉ `reshape` hoặc chuyển vị khi điều đó đúng với ý nghĩa dữ liệu.

Không nên dùng `reshape` mù quáng chỉ để code chạy: reshape sai có thể trộn mẫu và đặc trưng mà không báo lỗi.

## Tự kiểm tra

1. Nếu $\mathbf{A}$ có shape `(5, 3)` và $\mathbf{B}$ có shape `(3, 7)`, shape của $\mathbf{A}\mathbf{B}$ là gì?
2. Vì sao `(5, 3) @ (5, 3)` không hợp lệ nhưng `(5, 3) * (5, 3)` hợp lệ?
3. Với `x.shape == (3,)`, shape của `x @ x`, `x[:, None] @ x[None, :]` và `x[None, :] @ x[:, None]` là gì?
4. Vì sao $\mathbf{A}\mathbf{B}$ thường khác $\mathbf{B}\mathbf{A}$?
5. Trong $(\mathbf{A}\mathbf{B})\mathbf{x}=\mathbf{A}(\mathbf{B}\mathbf{x})$, phép biến đổi nào tác động lên $\mathbf{x}$ trước?
6. Với `X.shape == (32, 10)`, `W.shape == (10, 4)`, `b.shape == (4,)`, shape của `X @ W + b` là gì và bias được broadcast theo trục nào?
7. Cosine similarity bằng 0 nói gì về hai vector khác zero?
8. Khi nào `A * B` có thể chạy dù `A.shape != B.shape`?
9. Tại sao `x.T` vẫn có shape `(n,)` nếu `x` là mảng NumPy 1D?
10. Với `A(1000, 10)`, `B(10, 1000)`, `x(1000,)`, thứ tự ngoặc nào tránh tạo ma trận `(1000, 1000)`?

## Checklist

- [ ] Tôi tính tay được dot product và một phần tử của matrix product.
- [ ] Tôi viết được quy tắc `(m, n) @ (n, p) -> (m, p)` và giải thích từng kích thước.
- [ ] Tôi phân biệt được `@`, `*`, `np.dot` và broadcasting.
- [ ] Tôi giải thích được matrix-vector product theo hàng và theo cột.
- [ ] Tôi nhớ rằng matrix multiplication kết hợp nhưng không giao hoán.
- [ ] Tôi phân biệt được `(n,)`, `(1, n)` và `(n, 1)` trong NumPy.
- [ ] Tôi cài đặt được `X @ W + b` cho một batch và kiểm tra shape.
- [ ] Tôi tính được cosine similarity và xử lý trường hợp vector zero.
- [ ] Tôi chọn được thứ tự ngoặc giúp tránh ma trận trung gian lớn.
- [ ] Tôi hoàn thành notebook và tự giải thích ít nhất một lỗi shape đã sửa.
