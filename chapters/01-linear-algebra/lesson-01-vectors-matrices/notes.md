# Bài 01 — Ký hiệu cơ bản, Vector và Ma trận

Tài liệu đi kèm: [thực hành NumPy](practice.ipynb) và
[liên hệ với Machine Learning trong thực tế](ml-in-practice.md).

## Mục tiêu

Sau bài này, bạn có thể:

- Phân biệt scalar, vector, matrix, tensor và đọc đúng chỉ số.
- Giải thích `shape`, `ndim`, `size`, `dtype` của một mảng NumPy.
- Truy cập hàng, cột, vùng con; `reshape` và chuyển vị mà không làm sai ý nghĩa dữ liệu.
- Phân biệt rõ `(n,)`, `(1, n)` và `(n, 1)`.
- Mô tả dữ liệu tabular, ảnh và một mini-batch bằng tensor.
- Dự đoán shape trước khi chạy code và phát hiện các lỗi shape thường gặp.

## 1. Từ số đến tensor

### Scalar

Scalar là một số đơn lẻ, chẳng hạn nhiệt độ, learning rate hoặc một giá trị loss:

$$a \in \mathbb{R}.$$

Trong NumPy, `np.array(3.0)` là mảng 0 chiều, có shape `()`.

### Vector

Vector là một dãy số có thứ tự:

$$
\mathbf{x}=
\begin{bmatrix}
x_1\\x_2\\\vdots\\x_n
\end{bmatrix}
\in\mathbb{R}^n.
$$

Vector có thể biểu diễn một điểm trong không gian, một mẫu có $n$ đặc trưng, hoặc một tập
tham số. Trong toán học, ta thường vẽ vector thành cột. Trong NumPy, `np.array([1, 2, 3])`
lại có shape `(3,)`: nó chỉ có một trục, chưa phải hàng hay cột.

### Matrix

Ma trận là bảng chữ nhật gồm $m$ hàng và $n$ cột:

$$
\mathbf{A}=
\begin{bmatrix}
A_{11} & \cdots & A_{1n}\\
\vdots & \ddots & \vdots\\
A_{m1} & \cdots & A_{mn}
\end{bmatrix}
\in\mathbb{R}^{m\times n}.
$$

$A_{ij}$ là phần tử ở hàng $i$, cột $j$. Chỉ số toán học thường bắt đầu từ 1, còn chỉ số
Python bắt đầu từ 0: $A_{23}$ tương ứng với `A[1, 2]`.

### Tensor

Tensor trong ngữ cảnh NumPy/ML là mảng có số trục bất kỳ. Scalar, vector và matrix lần lượt
là tensor 0, 1 và 2 chiều. Ví dụ:

- ảnh xám: `(height, width)`;
- ảnh RGB: `(height, width, channels)`;
- batch ảnh RGB: `(batch, height, width, channels)` theo quy ước channels-last.

Từ “chiều” dễ gây nhầm: vector thuộc $\mathbb{R}^{100}$ có 100 thành phần nhưng mảng NumPy
của nó chỉ có **một trục**, tức `ndim == 1`.

## 2. Shape là hợp đồng của dữ liệu

Với một mảng `A`:

- `A.shape`: độ dài trên từng trục;
- `A.ndim`: số trục;
- `A.size`: tổng số phần tử, bằng tích các kích thước trong `shape`;
- `A.dtype`: kiểu dữ liệu của mỗi phần tử.

Ví dụ, `A.shape == (2, 3)` nghĩa là `A` có 2 hàng và 3 cột. Khi thiết kế code ML, hãy
coi shape là một “hợp đồng”: mỗi trục phải có ý nghĩa rõ ràng và được giữ nhất quán.

| Đối tượng | Ký hiệu | Shape NumPy thường dùng |
|---|---|---|
| Scalar | $a$ | `()` |
| Vector 1D | $\mathbf{x}\in\mathbb{R}^d$ | `(d,)` |
| Một mẫu dạng hàng | $\mathbf{x}^T$ | `(1, d)` |
| Một vector cột | $\mathbf{x}$ | `(d, 1)` |
| Dataset tabular | $\mathbf{X}\in\mathbb{R}^{N\times d}$ | `(N, d)` |

Trong quy ước phổ biến cho dữ liệu tabular:

$$
\mathbf{X}=
\begin{bmatrix}
--- (\mathbf{x}^{(1)})^T ---\\
--- (\mathbf{x}^{(2)})^T ---\\
\vdots\\
--- (\mathbf{x}^{(N)})^T ---
\end{bmatrix}
\in\mathbb{R}^{N\times d},
$$

trong đó mỗi hàng là một mẫu, mỗi cột là một đặc trưng. Nhãn thường có shape `(N,)` hoặc
`(N, 1)`; phải chọn một quy ước và dùng nhất quán.

## 3. Trục (`axis`) và phép gộp

Với `X.shape == (N, d)`:

- `axis=0` gộp theo chiều các hàng, nên kết quả còn một giá trị cho mỗi **cột/đặc trưng**;
- `axis=1` gộp theo chiều các cột, nên kết quả còn một giá trị cho mỗi **hàng/mẫu**.

Ví dụ:

```python
feature_means = X.mean(axis=0)  # shape (d,)
sample_means = X.mean(axis=1)   # shape (N,)
```

Cách nhớ tốt hơn câu “ngang/dọc” là: trục được nêu trong `axis` sẽ biến mất, trừ khi dùng
`keepdims=True`. Khi đó kích thước trục được giữ lại bằng 1:

```python
feature_means = X.mean(axis=0, keepdims=True)  # shape (1, d)
```

`keepdims=True` thường giúp phép broadcasting sau đó dễ đọc và ít lỗi hơn.

## 4. Indexing và slicing

Giả sử `X.shape == (N, d)`:

```python
X[0]       # hàng đầu tiên, shape (d,)
X[0, :]    # tương đương X[0]
X[:, 1]    # cột thứ hai, shape (N,)
X[:, 1:2]  # vẫn là ma trận hai chiều, shape (N, 1)
X[1:4, :2] # hàng 1..3 và hai cột đầu, shape (3, 2)
X[-1]      # hàng cuối
```

Điểm quan trọng: dùng một chỉ số nguyên thường làm mất trục tương ứng; dùng một slice giữ
trục đó. Vì vậy `X[:, 1]` và `X[:, 1:2]` chứa cùng các giá trị nhưng có shape khác nhau.

Boolean indexing lọc phần tử hoặc hàng theo điều kiện:

```python
mask = X[:, 0] > 0
positive_rows = X[mask]
```

## 5. `(n,)`, `(1, n)` và `(n, 1)` không giống nhau

```python
x = np.array([1, 2, 3])  # (3,)
row = x.reshape(1, -1)    # (1, 3)
col = x.reshape(-1, 1)    # (3, 1)
```

Với mảng 1D, `x.T` vẫn có shape `(3,)`; chuyển vị không thể tạo ra một trục mới. Muốn có
vector cột, dùng `x[:, None]` hoặc `x.reshape(-1, 1)`. Muốn có vector hàng, dùng
`x[None, :]` hoặc `x.reshape(1, -1)`.

`-1` trong `reshape` yêu cầu NumPy tự suy ra đúng một kích thước còn thiếu. Tổng số phần tử
không được thay đổi:

```python
np.arange(12).reshape(3, -1)  # NumPy suy ra shape (3, 4)
```

## 6. Chuyển vị và hoán đổi trục

Với $A\in\mathbb{R}^{m\times n}$, chuyển vị $A^T\in\mathbb{R}^{n\times m}$ thỏa:

$$
(A^T)_{ij}=A_{ji}.
$$

Trong NumPy, `A.T` đảo thứ tự tất cả các trục. Với tensor nhiều chiều, nên viết rõ phép
hoán đổi trục bằng `transpose` hoặc `moveaxis`:

```python
images_hwc = np.zeros((32, 28, 28, 3))
images_chw = images_hwc.transpose(0, 3, 1, 2)  # (32, 3, 28, 28)
```

Đây không phải biến đổi nội dung ảnh; chỉ thay đổi ý nghĩa và thứ tự của các trục.

## 7. Broadcasting cơ bản

Broadcasting cho phép NumPy tính toán giữa các mảng khác shape khi các kích thước tương
thích. So sánh shape từ phải sang trái; hai kích thước tương thích nếu bằng nhau hoặc một
trong hai bằng 1.

Với `X.shape == (N, d)` và `mu.shape == (d,)`, phép `X - mu` trừ cùng vector trung bình khỏi
mỗi hàng. Có thể hình dung `mu` được lặp thành `(N, d)`, nhưng NumPy không cần tạo bản sao
đầy đủ trong bộ nhớ.

```python
mu = X.mean(axis=0)            # (d,)
sigma = X.std(axis=0)          # (d,)
X_standardized = (X - mu) / sigma
```

Broadcasting mạnh nhưng cũng có thể che giấu lỗi. Trước mỗi phép toán, hãy ghi shape cạnh
biến và tự kiểm tra các trục nào đang được mở rộng.

## 8. `dtype` và kiểu dữ liệu

NumPy lưu các phần tử của một mảng với cùng `dtype`:

```python
labels = np.array([0, 1, 1])            # thường là integer
features = np.array([1, 2, 3], dtype=float)
mask = labels == 1                       # boolean
```

Một số phép toán có thể đổi dtype. Chuẩn hóa dữ liệu hoặc chia số thường cần floating point.
Không nên dựa vào ép kiểu ngầm; hãy dùng `astype` khi ý định cần được thể hiện rõ.

## 9. View, copy và thay đổi tại chỗ

Nhiều phép slicing cơ bản trả về một **view** dùng chung dữ liệu với mảng gốc:

```python
first_two_rows = X[:2]
first_two_rows[0, 0] = -999  # X cũng có thể bị thay đổi
```

Khi cần một mảng độc lập, dùng `.copy()`. Đây là lỗi khó thấy trong pipeline tiền xử lý, vì
một thay đổi tưởng chỉ áp dụng cho tập con có thể làm đổi dữ liệu gốc.

## 10. Ví dụ xuyên suốt: dữ liệu học sinh

Giả sử ba đặc trưng là số giờ học, số giờ ngủ và điểm kiểm tra trước:

$$
X=
\begin{bmatrix}
2 & 7 & 60\\
4 & 6 & 72\\
6 & 8 & 85\\
8 & 7 & 92
\end{bmatrix}.
$$

Khi đó `X.shape == (4, 3)`: 4 mẫu và 3 đặc trưng. `X[1]` là mẫu thứ hai, `X[:, 0]` là toàn
bộ giá trị của đặc trưng “giờ học”, còn `X.mean(axis=0)` trả về trung bình của từng đặc trưng.

Nếu chuẩn hóa từng đặc trưng:

$$
Z_{ij}=\frac{X_{ij}-\mu_j}{\sigma_j},
$$

thì $Z$ vẫn có shape `(4, 3)`, trung bình mỗi cột xấp xỉ 0 và độ lệch chuẩn mỗi cột xấp xỉ
1. Đây là một ví dụ trực tiếp về `axis=0` kết hợp broadcasting.

## Lỗi thường gặp

- Nhầm số hàng và số cột; lẫn chỉ số toán học bắt đầu từ 1 với Python bắt đầu từ 0.
- Cho rằng `(n,)`, `(1, n)` và `(n, 1)` luôn tương đương.
- Dùng `x.T` với vector 1D rồi nghĩ rằng đã tạo vector cột.
- Dùng `X[:, j]` khi phần code sau yêu cầu shape `(N, 1)`.
- Gộp sai `axis`, dẫn đến chuẩn hóa theo từng mẫu thay vì từng đặc trưng.
- Broadcasting chạy được nhưng trên sai trục.
- Sửa một slice và vô tình sửa luôn mảng gốc vì slice là view.
- Dùng `reshape` đúng số phần tử nhưng làm mất ý nghĩa của các trục.

## Quy trình chống lỗi shape

1. Viết ý nghĩa từng trục: ví dụ `(samples, features)`.
2. Dự đoán shape đầu ra trước khi chạy.
3. In hoặc assert shape tại ranh giới quan trọng.
4. Với phép gộp, nói thành lời trục nào biến mất.
5. Với broadcasting, căn các shape từ phải sang trái.
6. Chỉ dùng `reshape` khi có thể giải thích ý nghĩa shape mới.

## Tự kiểm tra

1. Nếu `X` có 100 mẫu, 8 đặc trưng thì shape hợp lý là gì?
2. Shape của `X.T` là gì? `X[:, 2]` lấy hàng hay cột, và có shape nào?
3. `X[:, 2:3]` khác `X[:, 2]` ở điểm nào?
4. Vì sao `np.array([1, 2, 3]).T` không trở thành vector cột?
5. Với `X.shape == (100, 8)`, `X.mean(axis=0)` và `X.mean(axis=1)` có shape gì?
6. `X - X.mean(axis=0)` dùng broadcasting như thế nào?
7. Hãy mô tả shape của một batch 32 ảnh RGB kích thước $224\times224$.
8. Khi nào cần `.copy()` sau một phép slicing?

<details>
<summary>Đáp án ngắn</summary>

1. `(100, 8)`.
2. `(8, 100)`; lấy cột thứ ba; shape `(100,)`.
3. Giá trị giống nhau, nhưng shape lần lượt là `(100, 1)` và `(100,)`.
4. Mảng chỉ có một trục nên không có hai trục để hoán đổi.
5. `(8,)` và `(100,)`.
6. Vector `(8,)` được khớp với trục đặc trưng của ma trận `(100, 8)` và trừ khỏi mỗi hàng.
7. Thường là `(32, 224, 224, 3)` hoặc `(32, 3, 224, 224)`, tùy quy ước.
8. Khi muốn sửa tập con mà không làm thay đổi mảng gốc.

</details>

## Checklist

- [ ] Đọc được ký hiệu chỉ số và giải thích ý nghĩa mỗi trục.
- [ ] Phân biệt `shape`, `ndim`, `size`, `dtype`.
- [ ] Tạo, reshape, transpose, index và slice bằng NumPy.
- [ ] Phân biệt `(n,)`, `(1, n)` và `(n, 1)`.
- [ ] Chọn đúng `axis` cho phép gộp.
- [ ] Giải thích được một phép broadcasting cơ bản.
- [ ] Nhận biết khi slice có thể làm thay đổi mảng gốc.
- [ ] Biểu diễn được dữ liệu tabular và dữ liệu ảnh bằng tensor.
