# Bài 04 — Định thức và Tính chất

Tài liệu đi kèm: [thực hành NumPy](practice.ipynb) và
[liên hệ với Machine Learning trong thực tế](ml-in-practice.md).

## Mục tiêu

Sau bài này, bạn có thể:

- Tính định thức của ma trận $2\times2$ và $3\times3$ bằng tay.
- Giải thích $|\det(A)|$ là hệ số co giãn diện tích/thể tích và dấu mô tả orientation.
- Suy luận định thức sau các phép biến đổi hàng mà không tính lại từ đầu.
- Dùng các tính chất của định thức cho tích, chuyển vị, nghịch đảo và ma trận tam giác.
- Giải thích vì sao $\det(A)=0$ tương đương phép biến đổi làm mất một chiều thông tin.
- Dùng `np.linalg.det` và `np.linalg.slogdet` đúng mục đích.
- Tránh dùng `det(A) == 0` để quyết định một hệ có ổn định số hay không.
- Liên hệ log-determinant với Gaussian likelihood, covariance và phép đổi biến.

## 1. Định thức là gì?

Định thức là một scalar gắn với **ma trận vuông**:

$$
A\in\mathbb{R}^{n\times n}
\quad\Longrightarrow\quad
\det(A)\in\mathbb{R}.
$$

Nó đồng thời mang ba lớp ý nghĩa:

1. **Đại số:** cho biết ma trận có khả nghịch hay không.
2. **Hình học:** đo hệ số co giãn thể tích có hướng của phép biến đổi $x\mapsto Ax$.
3. **Xác suất/ML:** đo thể tích của covariance và xuất hiện trong hệ số chuẩn hóa mật độ.

Định thức không được định nghĩa theo cách thông thường cho ma trận chữ nhật. Nếu
`A.shape == (m, n)` với `m != n`, `np.linalg.det(A)` sẽ báo lỗi vì không có một thể tích
$n$ chiều được ánh xạ trở lại chính không gian $n$ chiều.

Ký hiệu thường dùng:

$$
\det(A)
\qquad\text{hoặc}\qquad
|A|.
$$

Trong tài liệu này ưu tiên $\det(A)$ để không nhầm với trị tuyệt đối.

## 2. Định thức của ma trận $2\times2$

Với

$$
A=
\begin{bmatrix}
a&b\\c&d
\end{bmatrix},
$$

định thức là:

$$
\det(A)=ad-bc.
$$

Ví dụ:

$$
A=
\begin{bmatrix}
2&1\\0&3
\end{bmatrix}
\quad\Longrightarrow\quad
\det(A)=2\cdot3-1\cdot0=6.
$$

Trong NumPy:

```python
A = np.array([[2.0, 1.0], [0.0, 3.0]])
det_A = np.linalg.det(A)
np.testing.assert_allclose(det_A, 6.0)
```

`np.linalg.det` trả về số thực dấu phẩy động. Ngay cả khi kết quả toán học là số nguyên,
giá trị tính được có thể là `5.999999999999998`; nên dùng `allclose` thay cho `==`.

## 3. Ý nghĩa hình học trong hai chiều

Hai cột của $A\in\mathbb{R}^{2\times2}$ là ảnh của hai vector cơ sở:

$$
A\mathbf{e}_1=\mathbf{a}_1,
\qquad
A\mathbf{e}_2=\mathbf{a}_2.
$$

Hình vuông đơn vị có diện tích 1 được biến thành hình bình hành sinh bởi $\mathbf{a}_1$ và
$\mathbf{a}_2$. Diện tích mới là:

$$
\text{area}=|\det(A)|.
$$

Tổng quát, nếu một miền ban đầu có diện tích $S$, sau phép biến đổi tuyến tính $A$ nó có
diện tích:

$$
S_{\text{new}}=|\det(A)|S.
$$

### Dấu của định thức

- $\det(A)>0$: orientation được giữ nguyên.
- $\det(A)<0$: orientation bị đảo, giống như có một phép phản xạ.
- $\det(A)=0$: diện tích bị ép về 0; mặt phẳng bị co xuống một đường hoặc một điểm.

Ví dụ:

$$
R=
\begin{bmatrix}
-1&0\\0&1
\end{bmatrix}
$$

phản xạ qua trục tung. $\det(R)=-1$: diện tích không đổi vì $|-1|=1$, nhưng orientation bị
đảo.

## 4. Ý nghĩa trong $n$ chiều

Trong ba chiều, $|\det(A)|$ là hệ số co giãn thể tích. Trong $n$ chiều, nó là hệ số co giãn
**hypervolume** của hình hộp đơn vị sinh bởi các vector cơ sở.

- $|\det(A)|>1$: thể tích tăng.
- $0<|\det(A)|<1$: thể tích giảm nhưng chưa mất chiều.
- $|\det(A)|=1$: thể tích được bảo toàn.
- $\det(A)=0$: hypervolume bằng 0, ít nhất một chiều bị mất.

Đây là lý do định thức liên hệ trực tiếp với khả nghịch: nếu một phép biến đổi ép nhiều điểm
khác nhau vào cùng một không gian thấp chiều hơn, ta không thể khôi phục duy nhất đầu vào.

## 5. Định thức của ma trận $3\times3$

Với

$$
A=
\begin{bmatrix}
a&b&c\\
d&e&f\\
g&h&i
\end{bmatrix},
$$

khai triển theo hàng đầu tiên:

$$
\det(A)
=a(ei-fh)-b(di-fg)+c(dh-eg).
$$

Ví dụ:

$$
A=
\begin{bmatrix}
1&2&0\\
3&1&1\\
2&0&1
\end{bmatrix}.
$$

Khi đó:

$$
\det(A)
=1(1\cdot1-1\cdot0)
-2(3\cdot1-1\cdot2)
+0
=-1.
$$

Khai triển cofactor hữu ích để hiểu cấu trúc và tính tay ma trận nhỏ. Với ma trận lớn, không
nên cài đặt đệ quy theo cofactor: số phép tính tăng rất nhanh. Thư viện số dùng các phân rã
ma trận hiệu quả hơn.

## 6. Các phép biến đổi hàng và định thức

Ba phép biến đổi hàng cơ bản ảnh hưởng tới định thức theo cách khác nhau.

### 6.1. Đổi chỗ hai hàng

Nếu $B$ nhận được bằng cách đổi chỗ hai hàng của $A$:

$$
\det(B)=-\det(A).
$$

Đổi chỗ hai lần khôi phục dấu ban đầu.

### 6.2. Nhân một hàng với scalar

Nếu nhân đúng một hàng với $c$:

$$
\det(B)=c\det(A).
$$

Nếu nhân toàn bộ ma trận $n\times n$ với $c$, cả $n$ hàng đều được scale:

$$
\det(cA)=c^n\det(A).
$$

Đây là bẫy thường gặp: nói chung không phải $c\det(A)$.

### 6.3. Cộng bội của một hàng vào hàng khác

Phép biến đổi

$$
R_i\leftarrow R_i+cR_j,
\qquad i\ne j
$$

không đổi định thức. Về hình học, đây là phép shear: hình bị nghiêng nhưng thể tích không
đổi.

Ba quy tắc này cho phép đưa ma trận về dạng tam giác rồi tính định thức bằng tích đường chéo,
đồng thời theo dõi các lần đổi hàng và scale hàng.

## 7. Các tính chất quan trọng

### 7.1. Ma trận đơn vị

$$
\det(I_n)=1.
$$

### 7.2. Chuyển vị

$$
\det(A^T)=\det(A).
$$

Vì hàng của $A^T$ là cột của $A$, định thức đối xử đối xứng với hàng và cột.

### 7.3. Tích ma trận

$$
\det(AB)=\det(A)\det(B).
$$

Điều này khớp với hình học: $B$ scale thể tích trước, sau đó $A$ scale thêm lần nữa.

### 7.4. Nghịch đảo

Nếu $A$ khả nghịch:

$$
\det(A^{-1})=\frac{1}{\det(A)}.
$$

Vì $AA^{-1}=I$ nên:

$$
\det(A)\det(A^{-1})=1.
$$

### 7.5. Lũy thừa

Với số nguyên dương $k$:

$$
\det(A^k)=\det(A)^k.
$$

### 7.6. Tổng ma trận

Nói chung:

$$
\det(A+B)\ne\det(A)+\det(B).
$$

Định thức nhân được qua tích ma trận, nhưng không phân phối qua phép cộng.

## 8. Ma trận tam giác và ma trận chéo

Với ma trận tam giác trên hoặc dưới, định thức là tích các phần tử trên đường chéo:

$$
\det(U)=\prod_{i=1}^{n}U_{ii}.
$$

Ví dụ:

$$
U=
\begin{bmatrix}
2&4&-1\\
0&3&5\\
0&0&-2
\end{bmatrix}
\quad\Longrightarrow\quad
\det(U)=2\cdot3\cdot(-2)=-12.
$$

Ma trận chéo là trường hợp đặc biệt của ma trận tam giác, nên:

$$
\det(\operatorname{diag}(d_1,\ldots,d_n))=d_1\cdots d_n.
$$

Nếu một phần tử đường chéo của ma trận tam giác bằng 0, định thức bằng 0 và ma trận suy biến.

## 9. Định thức, khả nghịch và mất thông tin

Với ma trận vuông $A$, các phát biểu sau tương đương:

- $A$ khả nghịch;
- $\det(A)\ne0$;
- các cột của $A$ độc lập tuyến tính;
- phép biến đổi $x\mapsto Ax$ không ép không gian xuống số chiều thấp hơn;
- hệ $Ax=b$ có nghiệm duy nhất với mọi $b$.

Ví dụ:

$$
A=
\begin{bmatrix}
2&4\\1&2
\end{bmatrix}
$$

có cột thứ hai bằng hai lần cột thứ nhất:

$$
\det(A)=2\cdot2-4\cdot1=0.
$$

Hình vuông đơn vị bị ép thành một đoạn thẳng. Mọi điểm khác nhau theo hướng bị mất sẽ có thể
cho cùng đầu ra, nên không tồn tại phép nghịch đảo duy nhất.

## 10. Vì sao không dùng `det(A) == 0` để kiểm tra hệ số?

Đẳng thức $\det(A)=0$ là tiêu chuẩn toán học chính xác. Nhưng trong tính toán số, dùng
`np.linalg.det(A) == 0` để quyết định có giải được hệ hay không là không đáng tin cậy:

1. sai số dấu phẩy động có thể làm số 0 thành một số rất nhỏ;
2. định thức phụ thuộc mạnh vào kích thước và scale của ma trận;
3. định thức rất nhỏ không nhất thiết nghĩa là điều kiện xấu nếu toàn bộ ma trận cũng có scale
   nhỏ;
4. định thức khác 0 không bảo đảm nghiệm ổn định.

Ví dụ, $A=10^{-6}I_{10}$ có:

$$
\det(A)=10^{-60},
$$

rất nhỏ, nhưng $\kappa(A)=1$: hệ có độ điều kiện tốt vì mọi hướng được scale như nhau.

Ngược lại, một ma trận có determinant khác 0 vẫn có thể gần suy biến và condition number rất
lớn. Trong code:

- giải hệ bằng `np.linalg.solve` và xử lý `LinAlgError`;
- dùng `np.linalg.matrix_rank`, singular values hoặc `np.linalg.cond` để chẩn đoán;
- dùng `np.allclose` khi kiểm tra một đẳng thức số học.

## 11. `det` và `slogdet`

`np.linalg.det(A)` trả trực tiếp định thức. Trong nhiều bài toán xác suất, ta cần
$\log|\det(A)|$. Tính determinant trước rồi lấy log có thể overflow hoặc underflow:

```python
det_A = np.linalg.det(A)
logabsdet = np.log(abs(det_A))  # có thể không ổn định
```

NumPy cung cấp:

```python
sign, logabsdet = np.linalg.slogdet(A)
```

Kết quả thỏa:

$$
\det(A)=\text{sign}\times\exp(\text{logabsdet}).
$$

- `sign` là `-1`, `0` hoặc `1` cho ma trận thực;
- `logabsdet` là $\log|\det(A)|$;
- nếu ma trận suy biến, `sign == 0` và `logabsdet == -inf`.

Với covariance matrix hợp lệ, determinant phải dương, nên `sign` kỳ vọng bằng 1.

## 12. Determinant của một batch ma trận

NumPy áp dụng `det` và `slogdet` trên hai trục cuối:

```python
matrices = np.stack([A1, A2, A3])  # shape (3, n, n)
dets = np.linalg.det(matrices)      # shape (3,)
```

Các trục đứng trước được coi là batch. Điều này hữu ích khi xử lý nhiều covariance matrix,
nhiều Jacobian hoặc nhiều phép biến đổi cùng lúc.

## 13. Liên hệ với covariance và Gaussian likelihood

Với Gaussian $d$ chiều có mean $\mu$ và covariance $\Sigma$, log-density là:

$$
\log p(x)
=-\frac12\left[
d\log(2\pi)
+\log\det(\Sigma)
+(x-\mu)^T\Sigma^{-1}(x-\mu)
\right].
$$

$\det(\Sigma)$ đo “thể tích phân tán” của phân phối:

- determinant lớn: phân phối trải rộng trên nhiều hướng;
- determinant nhỏ: phân phối tập trung trong một thể tích hẹp;
- determinant 0: covariance suy biến, không có density đầy đủ trong $d$ chiều.

Khi tính code:

- dùng `slogdet(Sigma)` cho log-determinant;
- dùng `solve(Sigma, delta)` cho Mahalanobis term;
- không tạo `inv(Sigma)` chỉ để nhân với vector.

## 14. Liên hệ với Jacobian và phép đổi biến

Nếu $y=Ax$ với $A$ khả nghịch, một miền thể tích nhỏ quanh $x$ được scale bởi
$|\det(A)|$. Để tổng xác suất vẫn bằng 1, mật độ phải đổi theo hệ số nghịch đảo:

$$
p_Y(y)=p_X(A^{-1}y)\frac{1}{|\det(A)|}.
$$

Với phép biến đổi phi tuyến $y=f(x)$, ma trận đạo hàm cục bộ $J_f(x)$ đóng vai trò của $A$:

$$
\log p_Y(y)
=\log p_X(x)-\log|\det J_f(x)|.
$$

Đây là nền tảng của normalizing flows. Bài này chỉ cần nắm trực giác thể tích; Jacobian sẽ
được học kỹ hơn trong chương Matrix Calculus.

## 15. Lỗi thường gặp

### Lỗi 1 — Quên lấy trị tuyệt đối khi nói về diện tích

Diện tích/thể tích là $|\det(A)|$, còn dấu mô tả orientation.

### Lỗi 2 — Nghĩ rằng $\det(cA)=c\det(A)$

Với $A$ là $n\times n$, công thức đúng là $\det(cA)=c^n\det(A)$.

### Lỗi 3 — Phân phối determinant qua phép cộng

Nói chung $\det(A+B)\ne\det(A)+\det(B)$.

### Lỗi 4 — Dùng `==` với kết quả `det`

Dùng `np.isclose`/`np.allclose` cho kiểm chứng minh họa; dùng rank/condition number hoặc thuật
toán giải phù hợp cho chẩn đoán số.

### Lỗi 5 — Tính `log(det(A))` trực tiếp

Nếu cần log absolute determinant, dùng `slogdet` để tránh overflow/underflow và giữ thông tin
dấu.

### Lỗi 6 — Dùng determinant để so sánh ma trận khác số chiều

Determinant có đơn vị và scale phụ thuộc số chiều; không thể so sánh máy móc covariance 2D
với covariance 100D.

### Lỗi 7 — Cài cofactor đệ quy cho ma trận lớn

Cofactor phù hợp để học và tính tay ma trận nhỏ, không phải thuật toán production.

## 16. Quy trình làm việc với determinant

1. Kiểm tra ma trận có vuông không và ghi rõ ý nghĩa các trục.
2. Với ma trận $2\times2$ hoặc $3\times3$, tính tay để xây trực giác rồi đối chiếu NumPy.
3. Nếu ma trận tam giác/chéo, dùng tích đường chéo.
4. Khi theo dõi biến đổi hàng, ghi lại đổi hàng, scale hàng và row addition.
5. Nếu cần log-determinant, dùng `slogdet` và kiểm tra `sign`.
6. Nếu mục tiêu là giải hệ, gọi `solve` thay vì kiểm tra determinant trước.
7. Nếu mục tiêu là chẩn đoán ổn định, xem condition number, rank hoặc singular values.
8. Luôn phân biệt trị tuyệt đối (volume) với dấu (orientation).

## Tự kiểm tra

1. Phép biến đổi có determinant $-3$ thay đổi diện tích và orientation ra sao?
2. Vì sao hai cột phụ thuộc tuyến tính làm determinant bằng 0?
3. Đổi chỗ hai hàng hai lần ảnh hưởng determinant thế nào?
4. Nếu nhân một hàng của ma trận $4\times4$ với 5 thì determinant đổi ra sao? Nếu nhân toàn
   bộ ma trận với 5 thì sao?
5. Phép cộng bội một hàng vào hàng khác tương ứng với biến đổi hình học nào và ảnh hưởng thể
   tích ra sao?
6. Vì sao $\det(AB)=\det(A)\det(B)$ phù hợp với trực giác hợp thành phép biến đổi?
7. Định thức của ma trận tam giác có đường chéo `[2, -1, 4]` là bao nhiêu?
8. Một determinant rất nhỏ có đủ để kết luận ma trận gần suy biến không?
9. `slogdet` trả về hai đại lượng gì và vì sao hữu ích?
10. Vì sao log-density Gaussian chứa $\log\det(\Sigma)$?
11. Shape kết quả của `np.linalg.det(A)` khi `A.shape == (32, 5, 5)` là gì?
12. Trong phép đổi biến xác suất, vì sao mật độ phải chia cho $|\det(A)|$?

## Checklist

- [ ] Tôi tính được determinant $2\times2$ và $3\times3$ bằng tay.
- [ ] Tôi giải thích được $|\det(A)|$ bằng diện tích/thể tích.
- [ ] Tôi phân biệt được magnitude và dấu của determinant.
- [ ] Tôi suy luận được ảnh hưởng của ba phép biến đổi hàng.
- [ ] Tôi dùng được các tính chất của tích, chuyển vị, nghịch đảo và ma trận tam giác.
- [ ] Tôi giải thích được vì sao determinant 0 đồng nghĩa mất chiều thông tin.
- [ ] Tôi không dùng `det(A) == 0` để chẩn đoán ổn định số.
- [ ] Tôi dùng được `slogdet` và hiểu kết quả `sign`, `logabsdet`.
- [ ] Tôi liên hệ được log-determinant với covariance và Gaussian likelihood.
- [ ] Tôi hoàn thành notebook và bài tổng hợp phân loại Gaussian.

