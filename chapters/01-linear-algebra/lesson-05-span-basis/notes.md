# Bài 05 — Tổ hợp tuyến tính, Span và Basis

Tài liệu đi kèm gồm [thực hành NumPy](practice.ipynb) và
[liên hệ với Machine Learning trong thực tế](ml-in-practice.md).

## Mục tiêu

Sau bài này, bạn có thể làm được các việc sau.

- Viết một tổ hợp tuyến tính dưới dạng phép nhân ma trận–vector.
- Suy ra span là đường thẳng, mặt phẳng hay không gian lớn hơn từ các hướng sinh.
- Kiểm tra độc lập tuyến tính và rút một basis từ tập vector sinh có phần tử dư.
- Tìm tọa độ trong basis chuẩn và basis không chuẩn, rồi tái tạo vector gốc.
- Phân biệt hệ có nghiệm chính xác với nghiệm xấp xỉ do least squares trả về.
- Giải thích span xuất hiện ở đâu trong hồi quy tuyến tính và biểu diễn embedding.

## 1. Tổ hợp tuyến tính và shape

Cho $k$ vector $v_1,\ldots,v_k\in\mathbb{R}^d$. Một **tổ hợp tuyến tính** là vector

$$
v=c_1v_1+\cdots+c_kv_k,\qquad c_i\in\mathbb R.
$$

Mỗi hệ số $c_i$ quyết định mức đóng góp của $v_i$. Hệ số âm đảo chiều vector, còn hệ số
bằng 0 bỏ qua hướng tương ứng. Ví dụ, với $v_1=(1,2)^T$ và $v_2=(2,1)^T$,
ta có $3v_1-v_2=(1,5)^T$.

Xếp các vector sinh thành **cột** của $A=[v_1\ \cdots\ v_k]$. Khi đó $v=Ac$, với
quy tắc shape $(d,k)@(k,)\longrightarrow(d,)$ trong NumPy.

```python
import numpy as np

v1 = np.array([1.0, 2.0])
v2 = np.array([2.0, 1.0])
A = np.column_stack([v1, v2])  # (d, k)
c = np.array([3.0, -1.0])      # (k,)
v = A @ c                     # (d,)

np.testing.assert_allclose(v, [1.0, 5.0])
```

## 2. Span là tập tất cả kết quả có thể tạo ra

$$
\operatorname{span}(v_1,\ldots,v_k)
=\{c_1v_1+\cdots+c_kv_k:c_1,\ldots,c_k\in\mathbb R\}.
$$

Một tổ hợp tuyến tính là **một điểm** kết quả. Span là **tập mọi điểm** khi thay đổi
các hệ số. Span luôn chứa vector zero và đóng dưới phép cộng và nhân scalar.

| Vector sinh trong $\mathbb{R}^3$ | Span | Số chiều |
|---|---|---:|
| $(0,0,0)^T$ | chỉ gồm vector zero | 0 |
| $(1,0,1)^T$ và $(2,0,2)^T$ | đường thẳng qua gốc | 1 |
| $(1,0,1)^T$ và $(0,1,1)^T$ | mặt phẳng $z=x+y$ | 2 |
| Hai vector ở dòng trên và $(0,0,1)^T$ | toàn bộ $\mathbb R^3$ | 3 |

Ví dụ, mọi tổ hợp của $(1,0,1)^T$ và $(0,1,1)^T$ có dạng $(a,b,a+b)^T$. Vector
$(2,-1,1)^T$ nằm trong mặt phẳng đó với hệ số $(2,-1)^T$. Vector $(2,-1,0)^T$ nằm
ngoài vì tọa độ thứ ba phải bằng $2+(-1)=1$.

## 3. Độc lập tuyến tính và vector dư

Các vector $v_1,\ldots,v_k$ **độc lập tuyến tính** nếu phương trình

$$
c_1v_1+\cdots+c_kv_k=0
$$

chỉ có nghiệm $c_1=\cdots=c_k=0$. Nếu có một nghiệm với ít nhất một hệ số khác 0,
các vector **phụ thuộc tuyến tính**. Khi đó, chọn một hệ số khác 0 và chuyển vế để
biểu diễn vector tương ứng bằng các vector còn lại.

Chẳng hạn, nếu $v_3=v_1+2v_2$ thì $v_1+2v_2-v_3=0$. Bộ hệ số $(1,2,-1)^T$
chứng minh phụ thuộc. Có thể bỏ $v_3$ mà vẫn giữ nguyên span.

Với các vector đặt theo cột, số hướng độc lập là rank của $A$. Kiểm tra nhanh:

```python
generators = np.column_stack([v1, v2, v1 + 2.0 * v2])
rank = np.linalg.matrix_rank(generators)
assert rank == 2
assert rank < generators.shape[1]
```

Tập $k>d$ vector trong $\mathbb R^d$ luôn phụ thuộc. Tập độc lập có thể sinh một
không gian nhỏ hơn $\mathbb R^d$, chẳng hạn hai hướng độc lập trong $\mathbb R^3$.

## 4. Basis và số chiều

Một **basis** (cơ sở) của không gian $S$ là một tập vector độc lập tuyến tính có
span đúng bằng $S$. Tính sinh bảo đảm mỗi vector trong $S$ biểu diễn được. Tính độc lập
bảo đảm hệ số duy nhất: nếu $Bc=Bc'$, thì $B(c-c')=0$, suy ra $c=c'$.

| Tập trong $\mathbb R^2$ | Độc lập | Sinh $\mathbb R^2$ | Basis của $\mathbb R^2$ |
|---|---|---|---|
| $(1,0)^T$, $(0,1)^T$ | có | có | có |
| $(1,0)^T$ | có | không | không |
| $(1,0)^T$, $(0,1)^T$, $(1,1)^T$ | không | có | không |
| $(1,0)^T$, $(2,0)^T$ | không | không | không |

Số chiều của $S$ là số vector trong một basis bất kỳ của $S$. Một đường trong
$\mathbb R^3$ có basis gồm một vector. Một mặt phẳng qua gốc có basis gồm hai vector.
Thứ tự các vector basis quyết định thứ tự hệ số tọa độ.

## 5. Tọa độ theo basis

Basis chuẩn của $\mathbb R^2$ gồm $e_1=(1,0)^T$ và $e_2=(0,1)^T$. Nếu chọn
$b_1=(1,1)^T$, $b_2=(1,-1)^T$, cùng vector $v=(4,2)^T$ có tọa độ mới
$[v]_B=(3,1)^T$, vì $3b_1+b_2=v$.

```python
B = np.array([[1.0, 1.0], [1.0, -1.0]])  # b_i là các cột
v = np.array([4.0, 2.0])
coordinates = np.linalg.solve(B, v)

np.testing.assert_allclose(coordinates, [3.0, 1.0])
np.testing.assert_allclose(B @ coordinates, v)
```

$v$ là vector viết theo basis chuẩn. $[v]_B$ là danh sách hệ số theo basis $B$.
Mã hóa giải $Bc=v$, còn tái tạo tính $Bc$. Hai biến mô tả cùng vector qua hai
hệ tọa độ. Ví dụ, đổi thứ tự các cột của $B$ sẽ đổi thứ tự hệ số.

Với $N$ vector xếp thành các hàng của $X$, ta giải nhiều vế phải như Bài 03.
Nếu $B$ vuông khả nghịch, `C = np.linalg.solve(B, X.T).T` có một bộ tọa độ
trên mỗi hàng. Tái tạo bằng `C @ B.T`.

## 6. Kiểm tra một vector có thuộc span hay không

Kiểm tra $v\in\operatorname{span}(A)$ tương đương hỏi hệ $Ac=v$ có ít nhất một
nghiệm. Với một ma trận chữ nhật hoặc phụ thuộc, dùng `lstsq` để tìm $c$
rồi kiểm tra **phần dư** $Ac-v$.

```python
A_plane = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
v_in = np.array([2.0, -1.0, 1.0])
v_out = np.array([2.0, -1.0, 0.0])

for target in (v_in, v_out):
    coefficients, _, _, _ = np.linalg.lstsq(A_plane, target, rcond=None)
    error = np.linalg.norm(A_plane @ coefficients - target)
    limit = 1e-10 + 1e-8 * np.linalg.norm(target)
    print(error, error <= limit)
```

Với hai vector mẫu, phép kiểm tra lần lượt cho `True` và `False`.
Khi target nằm ngoài span, $Ac$ vẫn nằm trong span và là vector gần target nhất
theo chuẩn Euclid. Ngưỡng trong ví dụ phù hợp với dữ liệu nhỏ dạng `float64`.
Với dữ liệu đo, chọn ngưỡng tuyệt đối và tương đối theo scale và độ nhiễu.

Mảng `residuals` do `lstsq` trả về có
thể rỗng khi hệ thiếu rank hoặc có số hàng không lớn hơn số cột. Hãy tính phần dư
trực tiếp bằng `A @ coefficients - target` để kiểm tra khả năng tái tạo.

## 7. Rút basis từ một tập sinh

Với $A$ có shape `(d, k)`, duyệt các **cột gốc** từ trái sang phải. Giữ một cột khi việc thêm
nó làm rank của tập đang giữ tăng. Cột bị bỏ nằm trong span hiện có. Sau khi duyệt,
các cột giữ lại vừa độc lập vừa sinh cùng không gian trong số học chính xác.

```python
chosen = []
for j in range(generators.shape[1]):
    candidate = generators[:, chosen + [j]]
    if np.linalg.matrix_rank(candidate, tol=1e-10) > len(chosen):
        chosen.append(j)

basis = generators[:, chosen]
assert chosen == [0, 1]
```

Ví dụ dùng một ngưỡng cố định vì các vector có scale gần nhau. Với các hướng gần
phụ thuộc, rank ước lượng phụ thuộc ngưỡng. Thay đổi thứ tự cột cũng có thể chọn
được basis khác. Đây là cách thực hành cho ma trận nhỏ, còn Bài 06 đi sâu vào rank
và các không gian liên quan.

## 8. Hai ứng dụng trong ML

Trong hồi quy tuyến tính, với design matrix $X\in\mathbb R^{N\times d}$, vector
dự đoán $Xw\in\mathbb R^N$ là tổ hợp tuyến tính của $d$ **cột feature**. Đích $y$
nằm ngoài span thì mô hình chỉ có thể xấp xỉ. Nếu đã có intercept, hãy tính cột
toàn 1 vào design matrix khi xét span.

Trong embedding, các hàng của bảng có thể đại diện cho các từ, mỗi hàng là một
vector trong $\mathbb R^d$. Phép cộng và trừ embedding tạo một vector truy vấn
trong cùng không gian. Việc đổi basis cũng cho phép biểu diễn những vector đó
bằng hệ tọa độ khác. Phần ứng dụng minh họa cách dùng cosine để tìm hàng gần
vector truy vấn và cách kiểm tra tái tạo trước khi bỏ chiều biểu diễn.

## Lỗi thường gặp

- **Đặt vector sinh theo hàng khi lập hệ.** Dùng `np.column_stack` để
  `A @ c` kết hợp theo cột. Nếu đầu vào là bảng embedding theo hàng,
  lấy chuyển vị khi các hàng đó đóng vai trò vector sinh.
- **Đếm vector để kết luận số chiều.** Số chiều của span bằng số hướng độc lập.
  Ba vector có thể cùng nằm trên một đường.
- **Dùng `solve` cho basis chữ nhật.** Dùng `lstsq` và kiểm tra
  phần dư khi basis sinh một không gian con trong không gian lớn hơn.
- **Lấy `B.T @ v` làm tọa độ cho basis bất kỳ.** Công thức này cần các cột
  trực chuẩn. Với basis vuông thông thường, hãy giải $Bc=v$.
- **Đọc nhầm hệ số của tập sinh phụ thuộc là tọa độ duy nhất.** Tọa độ duy nhất gắn
  với một basis đã chọn và thứ tự cột của nó.

## Quy trình kiểm tra bằng NumPy

1. Ghi $d$ là số thành phần của mỗi vector và $k$ là số vector sinh.
2. Xếp cột để có $A$ shape `(d, k)`.
3. So sánh rank với $k$ để kiểm tra độc lập và với $d$ để kiểm tra khả năng sinh $\mathbb R^d$.
4. Nếu cần basis của span, giữ các cột làm tăng rank.
5. Chọn `solve` cho basis vuông khả nghịch hoặc `lstsq` cho
   bài toán tổng quát. Tái tạo vector và kiểm tra phần dư.

## Tự kiểm tra

1. Nếu mỗi vector có 4 thành phần và có 3 vector sinh, shape của $A$, $c$ và $Ac$ là gì?
2. Hai vector khác zero cùng phương sinh không gian bao nhiêu chiều?
3. Vì sao một tập chứa vector zero luôn phụ thuộc tuyến tính?
4. Tìm một bộ hệ số khác zero để chứng minh $(1,0)^T$, $(0,1)^T$, $(1,1)^T$ phụ thuộc.
5. Nếu thêm một vector đã thuộc span hiện tại, số chiều đổi thế nào?
6. Vì sao basis tạo ra biểu diễn tọa độ duy nhất?
7. Tìm tọa độ của $(2,0)^T$ theo basis $((1,1)^T,(1,-1)^T)$.
8. Khi `lstsq` trả hệ số cho target ngoài span, vector tái tạo biểu diễn điều gì?
9. Nếu hai feature ở hồi quy giống hệt nhau, điều gì xảy ra khi cộng $\delta$ vào trọng
   số thứ nhất và trừ $\delta$ khỏi trọng số thứ hai?

## Checklist

- [ ] Tôi viết được tổ hợp tuyến tính bằng tổng có trọng số và phép nhân ma trận.
- [ ] Tôi xác định được span và số chiều của các ví dụ nhỏ.
- [ ] Tôi phân biệt được tập sinh, tập độc lập và basis.
- [ ] Tôi tìm được một quan hệ phụ thuộc và bỏ được vector dư.
- [ ] Tôi giải được tọa độ theo basis và tái tạo vector gốc.
- [ ] Tôi dùng phần dư để kiểm tra một vector có thuộc span hay không.
- [ ] Tôi giải thích được vai trò của span trong hồi quy và embedding.

