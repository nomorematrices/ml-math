# Bài 08 — Đổi cơ sở và phép xoay trục

Một mũi tên hướng sang phải, có tọa độ $(1,0)$ trong hệ trục trên một tờ
giấy. Nếu quay tờ giấy một phần tư vòng ngược chiều kim đồng hồ trong khi giữ
mũi tên cố định, người đọc theo trục trên giấy sẽ ghi tọa độ nào? Nếu giữ
tờ giấy và quay mũi tên thì sao? Bài 05 cho ta cách dựng vector từ các hệ số,
còn Bài 07 cho ta cách đọc hệ số bằng tích vô hướng khi các trục trực chuẩn.

Tài liệu đi kèm gồm [thực hành NumPy](practice.ipynb) và
[ứng dụng đổi cơ sở trong PCA](ml-in-practice.md).

## Mục tiêu

- Tìm tọa độ theo một cơ sở bằng cách giải hệ và dựng lại vector ban đầu.
- Lập ma trận chuyển giữa hai cơ sở, kiểm tra chiều chuyển bằng một ví dụ.
- Phân biệt xoay vector và xoay hệ trục, xác định dấu của góc trong mỗi trường hợp.
- Áp dụng công thức cho một mẫu và một batch có các mẫu nằm theo hàng.
- Biểu diễn cùng một ánh xạ tuyến tính trong hai cơ sở.
- Phân biệt đổi cơ sở đầy đủ với phép chiếu giảm chiều trong PCA.

## 1. Một vector, hai cách ghi tọa độ

Gọi $E$ là cơ sở chuẩn của $\mathbb R^d$, $x=[x]_E$ là tọa độ chuẩn và
$B=[b_1\;\cdots\;b_d]$ là ma trận có các **cột** là vector của cơ sở mới,
viết theo $E$. Các cột phải độc lập tuyến tính. Khi đó mỗi vector có đúng một
bộ hệ số $c=[x]_B$ thỏa mãn

$$
x=c_1b_1+\cdots+c_db_d=Bc.
$$

Chọn $b_1=(1,0)^T$, $b_2=(1,1)^T$ và $x=(3,2)^T$. Muốn đi tới độ cao 2
cần bước hai lần theo $b_2$, tới $(2,2)^T$. Đi thêm một bước theo $b_1$ thì
chạm đầu mũi tên. Vậy $[x]_B=(1,2)^T$. Hai số này đếm số bước dọc theo hai
hướng đã chọn. Thứ tự cột quyết định hệ số nào đi với hướng nào.

```python
import numpy as np

B = np.array([[1., 1.], [0., 1.]])  # (2, 2), các cột là b1, b2
x = np.array([3., 2.])               # (2,), tọa độ chuẩn
c = np.linalg.solve(B, x)            # giải B @ c = x
np.testing.assert_allclose(c, [1., 2.])
np.testing.assert_allclose(B @ c, x)
```

Trong công thức ta viết $c=B^{-1}x$. Khi tính bằng NumPy, dùng
`np.linalg.solve(B, x)` như ở Bài 03. Nếu thay $b_2$ bằng $2b_1$, cả hai
hướng đều nằm ngang. Khi đó điểm $(3,2)^T$ nằm ngoài span, còn điểm $(3,0)^T$
có vô số cách biểu diễn. Hai cột ấy không tạo được cơ sở của mặt phẳng.

## 2. Ma trận chuyển từ cơ sở $B$ sang cơ sở $C$

Giả sử đã biết $c_B=[x]_B$. Đi qua hệ chuẩn để lập phương trình
$C c_C=x=B c_B$. Gọi ma trận chuyển là $T_{C\leftarrow B}$, đọc chỉ số từ
phải sang trái theo chiều đi của tọa độ. Khi đó

$$
T_{C\leftarrow B}=C^{-1}B,\qquad c_C=T_{C\leftarrow B}c_B.
$$

Giữ $B,x$ ở trên và chọn $C=[(1,1)^T\;(-1,1)^T]$. Để tìm cột đầu của
$T$, biểu diễn $b_1=(1,0)^T$ theo $C$. Lấy nửa vector thứ nhất trừ nửa
vector thứ hai của $C$ cho đúng $b_1$. Cột sau dễ hơn vì $b_2$ chính là
cột đầu của $C$. Ghép lại ta có

$$
T_{C\leftarrow B}=\begin{bmatrix}1/2&1\\-1/2&0\end{bmatrix},
\qquad [x]_C=T_{C\leftarrow B}\begin{bmatrix}1\\2\end{bmatrix}
=\begin{bmatrix}5/2\\-1/2\end{bmatrix}.
$$

Kiểm tra tay bằng cách ghép $\tfrac52(1,1)^T-\tfrac12(-1,1)^T=(3,2)^T$.
Mỗi cột thứ $j$ của $T$ là $[b_j]_C$, nên đẳng thức $CT=B$ kiểm tra được
cả ma trận chuyển. Khi đi qua cơ sở thứ ba $D$, phép chuyển đầu tiên nằm
bên phải trong tích $T_{D\leftarrow C}T_{C\leftarrow B}=T_{D\leftarrow B}$.

```python
C = np.array([[1., -1.], [1., 1.]])
T = np.linalg.solve(C, B)              # giải đồng thời cho hai cột của B
c_C = T @ c
np.testing.assert_allclose(C @ T, B)
np.testing.assert_allclose(c_C, [2.5, -0.5])
np.testing.assert_allclose(np.linalg.solve(B, C) @ c_C, c)
```

## 3. Đọc tọa độ bằng tích vô hướng

Với một cơ sở trực chuẩn $Q$, nhân $x=Qc$ với $Q^T$ cho $c=Q^Tx$.
Số $q_j^Tx$ là độ dài có dấu của hình chiếu lên trục đơn vị $q_j$. Với
$C$ ở mục 2, hai cột vuông góc và mỗi cột dài $\sqrt2$. Khi ấy $C^TC=2I$,
nên $C^Tx=(5,-1)^T$ bằng **hai lần** bộ tọa độ cần tìm.

Độ dài hình học cũng đọc được từ tọa độ trong một cơ sở xiên. Thay $x=Bc$
vào tích vô hướng ta được

$$
\|x\|_2^2=c^T(B^TB)c.
$$

Ở mục 1, $\|x\|^2=13$ trong khi $1^2+2^2=5$. Ma trận Gram
$B^TB=\begin{bmatrix}1&1\\1&2\end{bmatrix}$ tính đến cả chiều dài và góc
giữa hai trục, cho lại kết quả 13. Khi $B$ trực chuẩn, công thức rút gọn
thành $\|x\|_2^2=c^Tc$ như ở Bài 07.

## 4. Xoay mũi tên và xoay hệ trục

Quy ước góc dương ngược chiều kim đồng hồ, trục ngang hướng sang phải và
trục đứng hướng lên. Sau phép xoay góc $\theta$, $e_1$ đi tới
$(\cos\theta,\sin\theta)^T$ còn $e_2$ đi tới $(-\sin\theta,\cos\theta)^T$.
Hai ảnh này tạo thành hai cột của

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta\end{bmatrix}.
$$

| Thao tác | Phương trình với vector cột | Ví dụ $x=(1,0)^T$, $\theta=+90^\circ$ |
|---|---|---|
| Giữ trục, xoay vector (active) | $y=R(\theta)x$ | Đầu mũi tên đi đến $(0,1)^T$ |
| Giữ vector, xoay trục (passive) | $[x]_R=R(\theta)^Tx$ | Mũi tên hướng ngược trục mới thứ hai, tọa độ $(0,-1)^T$ |

Ở hàng thứ hai, hai trục mới là hai cột của $R$. Kiểm tra bằng $R[x]_R=x$.
Vì $R^TR=I$, phép xoay bảo toàn độ dài và góc. Dấu $\det R=+1$ phân biệt
phép xoay với phép phản xạ trong các ma trận trực giao hai chiều. Đảo phép
xoay tương đương xoay góc đối, nên $R(\theta)^{-1}=R(\theta)^T=R(-\theta)$.

Nếu xoay $30^\circ$ rồi $60^\circ$, vector cột cuối là
$R(60^\circ)R(30^\circ)x=R(90^\circ)x$. Trong mặt phẳng, hai góc xoay quanh
cùng gốc cộng được với nhau. Nếu phép thứ hai là kéo dãn theo một trục, thứ
tự có thể đổi kết quả. Khi tính bằng NumPy, đổi độ sang radian bằng
`np.deg2rad` trước khi gọi `np.sin` và `np.cos`.

## 5. Từ một mẫu sang cả batch

Cho $X$ có shape `(N, d)`, mỗi hàng là một mẫu trong hệ chuẩn. Lấy chuyển vị
công thức cho từng vector cột rồi xếp các kết quả thành hàng.

| Mục đích | Một mẫu `x` | Nhiều mẫu `X` theo hàng |
|---|---|---|
| Đọc tọa độ trong cơ sở bất kỳ $B$ | `np.linalg.solve(B, x)` | `np.linalg.solve(B, X.T).T` |
| Dựng lại mẫu từ tọa độ trong $B$ | `B @ c` | `coords @ B.T` |
| Xoay chủ động tất cả các điểm | `R @ x` | `X @ R.T` |
| Đọc tọa độ theo các trục đã xoay | `R.T @ x` | `X @ R` |

Hai tích cuối đều trả về shape `(N, d)` và giữ nguyên norm của mỗi hàng.
Dùng điểm kiểm tra $(1,0)$ với góc $+90^\circ$ để phát hiện lỗi dấu bằng
hướng mũi tên. Với vector NumPy 1D, `x.T` vẫn có shape `(d,)`.

## 6. Biểu diễn cùng một ánh xạ trong cơ sở mới

Cho $A$ là ma trận trong hệ chuẩn của một phép biến đổi tuyến tính. Nếu cả
đầu vào lẫn đầu ra đều viết theo $B$, ta lần lượt dựng vector $Bc$, áp dụng
$A$ rồi đọc tọa độ của $A Bc$ theo $B$. Vì thế ma trận của ánh xạ theo
$B$ là $A_B=B^{-1}AB$. Đẳng thức $BA_B=AB$ cho ta hai cách kiểm tra cùng một kết quả.

Chọn $A=\operatorname{diag}(2,1)$, phép kéo dãn tọa độ ngang hai lần. Với
$B$ ở mục 1, $Ab_1=(2,0)^T$ có tọa độ $(2,0)^T$ theo $B$, còn
$Ab_2=(2,1)^T$ có tọa độ $(1,1)^T$. Vậy
$A_B=\begin{bmatrix}2&1\\0&1\end{bmatrix}$. Với $[x]_B=(1,2)^T$, ta thu
được $[Ax]_B=(4,2)^T$, dựng về hệ chuẩn được $(6,2)^T$. Cách đọc từng
cột này sẽ xuất hiện lại ở Bài 09 khi ta chọn các trục mà $A$ chỉ co giãn.

## 7. Đổi cơ sở của một bảng dữ liệu trong PCA

Với hai cảm biến cùng đo nhiệt độ, một trục có thể đo mức nóng chung, trục
vuông góc còn lại đo chênh lệch giữa hai cảm biến. PCA tìm các hướng trực
chuẩn theo thứ tự phương sai giảm dần trên dữ liệu đã trừ trung bình. Gọi
$\mu$ là hàng trung bình của tập train và $Q$ là ma trận có các hướng ấy làm cột.
Dịch gốc bằng cách trừ $\mu$ là bước riêng, rồi mới đổi trục.

- Giữ đủ $d$ hướng cho $Z=(X-\mu)Q$ và $X=ZQ^T+\mu$.
- Giữ $k<d$ hướng đầu cho $Z_k=(X-\mu)Q_k$ và
  $\hat X=Z_kQ_k^T+\mu$. Mỗi hàng của $X-\hat X$ vuông góc với các hướng giữ lại.

Trong [ví dụ hai cảm biến](ml-in-practice.md), hai lần đo có cùng mức nóng
chung vẫn có thể có chênh lệch rất khác nhau. Dùng phần dư để xem chi tiết
nào biến mất khi chỉ lưu một tọa độ.

## Lỗi thường gặp

| Dấu hiệu | Cách kiểm tra |
|---|---|
| Hai cách đổi cơ sở cho kết quả khác nhau | Gắn nhãn cơ sở đầu và cuối, rồi kiểm tra $CT_{C\leftarrow B}=B$. |
| Dùng tích vô hướng cho cơ sở xiên | Tính $B^TB$. Để lấy tọa độ tổng quát, giải $Bc=x$. |
| Norm đúng nhưng góc quay ngược | Thử phép quay trên $e_1$ và đọc dấu của thành phần thứ hai. |
| Batch đúng shape nhưng một mẫu sai | So hàng đầu của batch với công thức vector cột tương ứng. |
| Hệ số rất nhạy khi chỉnh nhẹ đầu vào | Kiểm tra tính độc lập và độ điều kiện của $B$ như ở Bài 03. |
| PCA tái tạo ra giá trị quanh 0 dù cảm biến đo khoảng 18 °C | Khi giải mã, cộng lại đúng trung bình dùng lúc mã hóa. |

## Quy trình kiểm tra bằng NumPy

1. Ghi rõ gốc, đơn vị, hướng từng cột của cơ sở và trục lưu mẫu trong batch.
2. Tính tay một mẫu. Kiểm tra cả tọa độ dự kiến lẫn vector dựng lại bằng `assert_allclose`.
3. Với phép xoay, kiểm tra $R^TR$, định thức và ảnh của $e_1$. Với hai cơ sở bất kỳ, kiểm tra chiều đi và chiều về.
4. Khi giảm chiều, đo sai số tái tạo trên mẫu mới và kiểm tra phần dư vuông góc với các hướng giữ lại.

## Tự kiểm tra

1. Ở mục 1, nếu đổi chỗ $b_1,b_2$, tọa độ mới của $x$ là gì?
2. Cột thứ hai của $T_{C\leftarrow B}$ biểu diễn vector nào, theo cơ sở nào?
3. Hai cột của $C$ ở mục 2 vuông góc. Vì sao cần chia $C^Tx$ cho 2 để có tọa độ?
4. Với $x=(0,2)^T$, hãy tính riêng kết quả xoay vector và xoay trục góc $+90^\circ$.
5. Viết phép đổi tọa độ cho batch shape `(5, 2)` sang cơ sở $B$ ở mục 1. Đối số thứ hai của `solve` có shape nào?
6. Dùng $BA_B=AB$ để giải thích vì sao cột thứ hai của $A_B$ ở mục 6 là $(1,1)^T$.
7. Khi PCA chỉ giữ một trục, điều kiện nào để tái tạo chính xác một mẫu mới?

<details>
<summary>Đáp án ngắn</summary>

1. $(2,1)^T$. Số bước đi với từng hướng được hoán đổi theo thứ tự cột.
2. Đó là $[b_2]_C$. Trong ví dụ $b_2$ trùng cột đầu của $C$, nên cột này bằng $(1,0)^T$.
3. Vì $x=Cc$ và $C^TC=2I$, ta có $C^Tx=2c$.
4. Xoay vector cho $(-2,0)^T$. Xoay trục cho tọa độ $(2,0)^T$.
5. `np.linalg.solve(B, X.T).T`. Vế phải có shape `(2, 5)`, mỗi cột là một mẫu.
6. Cột thứ hai của $AB$ là $(2,1)^T=b_1+b_2$, nên cần một bước theo mỗi trục của $B$.
7. Sau khi trừ trung bình train, mẫu phải nằm trên đường thẳng sinh bởi trục được giữ.

</details>

## Checklist

- [ ] Tôi tìm được tọa độ trong một cơ sở xiên và dựng lại vector bằng tay.
- [ ] Tôi lập được ma trận chuyển giữa hai cơ sở và đọc được ý nghĩa từng cột.
- [ ] Tôi dự đoán được dấu của tọa độ khi xoay vector hoặc xoay hệ trục.
- [ ] Tôi xoay được cả batch và kiểm tra một hàng bằng kết quả tính tay.
- [ ] Tôi viết được ma trận của cùng một ánh xạ trong cơ sở mới.
- [ ] Tôi dùng phần dư tái tạo để giải thích thông tin mất đi trong bài tổng hợp PCA.
