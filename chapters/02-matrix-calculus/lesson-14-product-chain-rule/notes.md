# Bài 14 — Product Rule và Chain Rule

## Mục tiêu

- Áp dụng product rule cho scalar, dot product và matrix product.
- Áp dụng chain rule qua nhiều lớp hàm.
- Viết computational graph và backpropagate gradient.

## Trọng tâm

Scalar product rule: $(uv)'=u'v+uv'$. Với tích ma trận phụ thuộc $t$: $d(AB)=dA\,B+A\,dB$ và thứ tự không được đổi. Chain rule scalar: $d(f\circ g)/dx=f'(g(x))g'(x)$.

Với hàm vector theo convention Jacobian, $J_{f\circ g}(x)=J_f(g(x))J_g(x)$. Backpropagation chính là chain rule theo chiều ngược computational graph, tích lũy gradient khi một biến đi tới nhiều nhánh.

## Tự kiểm tra

1. Vì sao thứ tự nhân trong matrix product rule quan trọng?
2. Tính gradient của $(a^Tx)^2$.
3. Chain rule giải thích vanishing/exploding gradients như thế nào?

## Checklist

- [ ] Vẽ graph cho hàm hợp.
- [ ] Tính local derivatives và ghép theo chain rule.
- [ ] So sánh analytic gradient với finite difference.

