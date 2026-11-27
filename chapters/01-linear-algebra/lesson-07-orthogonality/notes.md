# Bài 07 — Hệ trực chuẩn và ma trận trực giao

Ở Bài 06, ta biết phần dư của nghiệm least squares vuông góc với column space.
Bài này dựng hình chiếu bằng tay để thấy phần dư ấy xuất hiện ở đâu. Cùng một
phép tính tích vô hướng còn giúp kiểm tra xem việc biến đổi embedding có giữ
nguyên khoảng cách giữa các mẫu hay không.

Tài liệu đi kèm gồm [thực hành NumPy](practice.ipynb) và
[liên hệ với Machine Learning trong thực tế](ml-in-practice.md).

## Mục tiêu

- Đọc ma trận các tích vô hướng để nhận diện một hệ trực giao hoặc trực chuẩn.
- Chứng minh một ma trận trực giao bảo toàn độ dài, khoảng cách và góc.
- Tính tọa độ, hình chiếu và phần dư của một vector trên một không gian con.
- Dựng hai cột trực chuẩn bằng Gram–Schmidt và kiểm tra phân rã QR bằng NumPy.
- Dùng QR để giải least squares và giải thích điều kiện áp dụng.
- Dự đoán tác động của phép quay và phép chiếu lên kết quả tìm láng giềng gần nhất.

## 1. Từ tích vô hướng đến hệ trực chuẩn

Với vector thực, $x^Ty=\sum_i x_i y_i$. Khi cả hai vector đều khác
zero, $x^Ty=\|x\|_2\|y\|_2\cos\theta$. Hai vector **trực giao** nếu $x^Ty=0$.
Một hệ **trực chuẩn** gồm các vector trực giao đôi một và mỗi vector có chuẩn
Euclid bằng 1. Chẳng hạn $u=(1,1)^T$ và $v=(2,-2)^T$ có tích vô hướng bằng
zero, nhưng chuẩn lần lượt là $\sqrt{2}$ và $2\sqrt{2}$. Chia từng vector cho
chuẩn của nó để được hai hướng trực chuẩn. Vector zero có tích vô hướng bằng
zero với mọi vector, nhưng không thể chuẩn hóa vì mẫu số bằng 0.

Xếp các vector thành cột của $B$. Phần tử $(i,j)$ trong **ma trận Gram**
$G=B^TB$ bằng $b_i^Tb_j$. Đường chéo cho biết bình phương độ dài từng cột.
Phần ngoài đường chéo cho biết quan hệ giữa các cặp cột.

| Kết quả đọc được từ $G$ | Kết luận |
|---|---|
| Ngoài đường chéo bằng 0, đường chéo dương | Các cột khác zero và trực giao đôi một |
| Đường chéo bằng 1 | Mỗi cột có độ dài 1 |
| $G=I$ | Các cột tạo thành hệ trực chuẩn |

Một hệ trực chuẩn luôn độc lập tuyến tính. Thật vậy, nếu $Qc=0$ và
$Q^TQ=I$, nhân hai vế với $Q^T$ cho $c=0$. Vì vậy các cột của $Q$ là một
basis của không gian chúng sinh ra (Bài 05).

## 2. Ma trận trực giao giữ lại những đại lượng nào?

Với $U\in\mathbb R^{d\times d}$, điều kiện $U^TU=I_d$ cho biết các cột trực
chuẩn. Vì $U$ vuông và có full rank, $U^{-1}=U^T$ và $UU^T=I_d$. Ma trận
vuông như vậy gọi là **ma trận trực giao**. Với hai vector $x,y$ bất kỳ,

$$
(Ux)^T(Uy)=x^T U^T U y=x^Ty.
$$

Cho $y=x$ ta được $\|Ux\|_2^2=\|x\|_2^2$. Áp dụng cho $x-y$ ta được
$\|Ux-Uy\|_2=\|x-y\|_2$. Với hai vector khác zero, cả tử và mẫu trong công
thức cosine đều được bảo toàn, do đó góc giữa chúng cũng được bảo toàn.

Thử tính với $x=(3,4)^T$. Ma trận $U=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$
quay $x$ thành $(-4,3)^T$. Cả hai điểm cách gốc 5 đơn vị. Mặt khác
$F=\operatorname{diag}(-1,1)$ biến $x$ thành $(-3,4)^T$ bằng phép phản xạ
qua trục đứng. Cả $U$ và $F$ đều trực giao. Từ Bài 04,
$1=\det(U^TU)=\det(U)^2$, nên định thức chỉ nhận hai giá trị $+1$ và $-1$.
Trong mặt phẳng, hai dấu này phân biệt phép quay và phép phản xạ. Riêng việc
biết $\det(U)=1$ chưa đủ để suy ra trực giao. Ví dụ
$\operatorname{diag}(2,1/2)$ có định thức 1 nhưng kéo dài hướng trục đầu tiên.

## 3. Hình chiếu lên một đường thẳng

Cho đường thẳng qua gốc có hướng $u\ne0$. Ta muốn chọn điểm $p=\alpha u$
gần $x$ nhất. Tại chân đường vuông góc, phần dư $r=x-p$ thỏa $u^Tr=0$.
Giải phương trình $u^Tx-\alpha u^Tu=0$ cho ta

$$
\alpha=\frac{u^Tx}{u^Tu},\qquad p=\alpha u,\qquad r=x-p.
$$

Lấy $x=(3,1)^T$, $u=(1,1)^T$. Khi ấy $\alpha=4/2=2$, $p=(2,2)^T$ và
$r=(1,-1)^T$. Kiểm tra nhanh được $u^Tr=0$ và $10=8+2$ theo định lý
Pythagore. Nếu đổi hướng sinh thành $2u$, hệ số giảm một nửa còn điểm $p$
giữ nguyên. Với hướng đơn vị $q=u/\|u\|_2$, công thức rút gọn thành
$p=q(q^Tx)$. Số $q^Tx$ là tọa độ có dấu dọc theo hướng $q$.

## 4. Hệ trực chuẩn trong một không gian con

Cho $Q=[q_1,\ldots,q_k]\in\mathbb R^{m\times k}$ có cột trực chuẩn và
$S=\operatorname{Col}(Q)$. Lấy tích vô hướng với từng cột, sau đó cộng các
hướng theo những hệ số vừa tìm.

| Đại lượng | Công thức | Shape NumPy | Ý nghĩa |
|---|---|---|---|
| Tọa độ | $c=Q^Tx$ | `(k,)` | Thành phần có dấu dọc mỗi hướng |
| Hình chiếu | $p=Qc$ | `(m,)` | Điểm trong $S$ gần $x$ nhất |
| Phần dư | $r=x-p$ | `(m,)` | Thành phần vuông góc với $S$ |
| Ma trận chiếu | $P=QQ^T$ | `(m, m)` | Ánh xạ $x$ thành $p$ |

Ta có $Q^Tr=Q^Tx-Q^TQc=0$. Với mọi $z\in S$, hai vector $r$ và $p-z$
vuông góc nên $\|x-z\|_2^2=\|r\|_2^2+\|p-z\|_2^2$. Khoảng cách nhỏ nhất
đạt được duy nhất tại $z=p$. Đây cũng là cách chứng minh vì sao chân đường
vuông góc ở phần 3 là điểm gần nhất.

Ma trận chiếu đối xứng vì $P^T=P$, và chiếu hai lần cho cùng kết quả vì
$P^2=Q(Q^TQ)Q^T=P$. Nếu $k<m$, $Q^TQ=I_k$ còn $QQ^T$ có rank $k$.
$P$ giữ nguyên mọi vector trong $S$ và triệt tiêu mọi vector trong $S^\perp$.
Do đó $Q$ cao bảo toàn độ dài khi đưa tọa độ $c$ vào $S$ bằng $Qc$. Khi đi
chiều ngược lại, $Q^Tx$ chỉ giữ đủ độ dài của $x$ nếu $x\in S$.

## 5. Dựng một basis bằng Gram–Schmidt

Xét hai cột độc lập $a_1=(1,1,0)^T$ và $a_2=(1,0,1)^T$. Chúng sinh ra mặt
phẳng $S$ gồm các điểm có thành phần thứ nhất bằng tổng hai thành phần sau.
Ta muốn giữ nguyên mặt phẳng này khi thay hai hướng xiên bằng hai hướng vuông góc.

1. Chuẩn hóa $a_1$ để có $q_1=(1,1,0)^T/\sqrt{2}$.
2. Đo phần của $a_2$ dọc theo $q_1$, được $q_1^Ta_2=1/\sqrt{2}$.
3. Trừ hình chiếu vừa đo. Vector còn lại là
   $v_2=a_2-q_1(q_1^Ta_2)=(1/2,-1/2,1)^T$.
4. Chia $v_2$ cho $\|v_2\|_2=\sqrt{3/2}$ để có $q_2=(1,-1,2)^T/\sqrt{6}$.

Mỗi bước chỉ trừ một tổ hợp của các cột trước, nên span của hai cột vẫn là $S$.
Nếu $a_2=2a_1$, bước 3 cho vector zero. Khi làm với số thực dấu phẩy động,
hãy so độ dài còn lại với một ngưỡng phù hợp với độ lớn dữ liệu trước khi chia.

Bây giờ chiếu $x=(3,-2,2)^T$ lên $S$. Hai tọa độ là $1/\sqrt{2}$ và
$9/\sqrt{6}$. Ghép lại, ta được $p=(2,-1,3)^T$ và $r=(1,-1,-1)^T$.
Thử kiểm tra $a_1^Tr=a_2^Tr=0$ trên giấy. Nếu dùng chính $p$ làm đầu vào,
lần chiếu tiếp theo cho phần dư bằng zero.

## 6. Đọc phân rã QR và giải least squares

Gom các hệ số ở phần 5 vào ma trận tam giác trên $R$. Với $A=[a_1,a_2]$ ta có

$$
A=QR,\qquad
R=\begin{bmatrix}\sqrt{2}&1/\sqrt{2}\\0&\sqrt{3/2}\end{bmatrix}.
$$

Cột thứ hai của $R$ ghi lại lượng $q_1$ và $q_2$ cần để dựng lại $a_2$.
Tổng quát, với $A$ cao hoặc vuông, shape `(m, k)` và full column rank, QR
rút gọn cho $Q$ shape `(m, k)`, $R$ shape `(k, k)`. Gram–Schmidt giúp
tính tay từng hướng. Trong tính toán số, `np.linalg.qr` dùng các phép phản
xạ Householder để giữ tính trực chuẩn tốt hơn khi các cột gần phụ thuộc.

```python
import numpy as np

A = np.array([[1., 1.], [1., 0.], [0., 1.]])
x = np.array([3., -2., 2.])
Q, R = np.linalg.qr(A, mode="reduced")
coords = Q.T @ x                  # (2,), tọa độ theo Q của hình chiếu
p = Q @ coords                    # (3,), cùng không gian với x
w = np.linalg.solve(R, coords)    # (2,), hệ số theo các cột gốc của A

np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-12)
np.testing.assert_allclose(Q @ R, A, atol=1e-12)
np.testing.assert_allclose(p, [2., -1., 3.], atol=1e-12)
np.testing.assert_allclose(w, [-1., 3.], atol=1e-12)
np.testing.assert_allclose(A.T @ (x - A @ w), 0., atol=1e-12)
```

Tại nghiệm least squares, dự đoán $Aw$ chính là $p$. Vì $Aw=QRw$ và
$p=Qc$, nhân hai vế với $Q^T$ cho hệ $Rw=c$. Các hệ số $w$ và $c$ cùng
mô tả $p$ theo hai basis khác nhau. Nhờ full column rank, $R$ khả nghịch và
`solve` trả về nghiệm duy nhất. Nếu các cột phụ thuộc, QR rút gọn vẫn có thể
trả về đủ $k$ cột trực chuẩn, kể cả các hướng ngoài range của $A$. Khi ấy dùng
`np.linalg.lstsq(A, x, rcond=None)` để fit hoặc lấy basis của range từ các
hướng SVD ứng với numerical rank như Bài 06.

## Lỗi thường gặp

- **Chuẩn hóa riêng từng cột rồi cho rằng cả hệ trực chuẩn.** Xem cả phần ngoài đường chéo của ma trận Gram.
- **Chiếu bằng $AA^Tx$ khi các cột của $A$ còn xiên.** Dùng basis trực chuẩn của range, hoặc tính $A\hat w$ từ least squares.
- **Đọc $Q^Tx$ như điểm trong không gian ban đầu.** Ghi shape của tọa độ rồi nhân $Q$ để dựng điểm.
- **So từng cột của hai kết quả QR.** Đổi dấu một cột $Q$ và hàng tương ứng của $R$ vẫn cho cùng $A$. Hãy kiểm tra $QR$, $Q^TQ$ và hình chiếu.
- **Tạo $P$ kích thước $m\times m$ chỉ để chiếu một vector.** Biểu thức `Q @ (Q.T @ x)` chỉ cần vector trung gian dài $k$.

## Quy trình kiểm tra bằng NumPy

1. Xác định vector basis nằm theo hàng hay theo cột. Với $k$ cột, kiểm tra `Q.T @ Q` với `np.eye(k)`.
2. Với phép chiếu, kiểm tra `p + r` khôi phục đầu vào và `Q.T @ r` gần zero. Chọn sai số phù hợp với scale của ví dụ.
3. Với QR, kiểm tra thêm `Q @ R`, tam giác trên của $R$ và rank của ma trận gốc trước khi giải hệ.
4. Với phép quay một batch, nếu mỗi mẫu là một hàng và quy tắc cho vector cột là $Ux$, dùng `X @ U.T`. Đối chiếu một hàng với `U @ X[0]`.

## Tự kiểm tra

1. Hai vector $(1,0)^T$ và $(1,1)^T/\sqrt{2}$ đều dài 1. Ma trận Gram của chúng có điểm nào khác $I_2$?
2. Nếu $Q$ có shape `(5, 2)` với hai cột trực chuẩn, hai tích $Q^TQ$ và $QQ^T$ có shape và rank bao nhiêu?
3. Dùng một dòng biến đổi để giải thích vì sao một phép trực giao giữ khoảng cách giữa hai mẫu.
4. Tính hình chiếu và phần dư của $(3,1)^T$ trên đường thẳng sinh bởi $(2,2)^T$.
5. Vì sao có thể kiểm tra $A^Tr=0$ dù đã dùng $Q$ để tính hình chiếu lên range của $A$?
6. Nếu thay $x$ ở phần 5 bằng $x+2r$, hình chiếu và phần dư mới là gì?
7. Khi hai cột của $A$ trùng nhau, bước nào của Gram–Schmidt gặp vấn đề? Bạn sẽ tính dự đoán least squares bằng NumPy như thế nào?

<details>
<summary>Đáp án ngắn</summary>

1. Hai phần tử ngoài đường chéo là $1/\sqrt{2}$.
2. Tích thứ nhất có shape `(2, 2)` và rank 2. Tích thứ hai có shape `(5, 5)` và cũng có rank 2.
3. $\|Ux-Uy\|_2^2=(x-y)^TU^TU(x-y)=\|x-y\|_2^2$.
4. Hệ số bằng 1, hình chiếu là $(2,2)^T$ và phần dư là $(1,-1)^T$.
5. Mỗi cột của $A$ là tổ hợp các cột của $Q$, nên cũng vuông góc với $r$.
6. Vì $x=p+r$, đầu vào mới là $p+3r$. Hình chiếu vẫn là $p$, phần dư mới là $3r$.
7. Sau khi trừ hình chiếu, cột thứ hai có độ dài zero. Lấy hệ số từ `lstsq` rồi nhân $A$ với hệ số để có dự đoán.

</details>

## Checklist

- [ ] Tôi đọc được độ dài và quan hệ vuông góc giữa các cột từ ma trận Gram.
- [ ] Tôi chứng minh được phép trực giao bảo toàn norm và dot product.
- [ ] Tôi chiếu được một vector lên đường thẳng và lên mặt phẳng, rồi kiểm tra phần dư.
- [ ] Tôi dựng được hai cột trực chuẩn bằng tay và nhận ra lúc xuất hiện cột phụ thuộc.
- [ ] Tôi giải least squares bằng QR khi có full column rank và kiểm tra với `lstsq`.
- [ ] Tôi hoàn thành bài tổng hợp và giải thích được khi nào láng giềng gần nhất có thể đổi.
