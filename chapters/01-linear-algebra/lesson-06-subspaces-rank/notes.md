# Bài 06 — Range, Null Space và Rank

## Mục tiêu

- Hiểu column space/range, row space và null space.
- Tính rank và áp dụng định lý rank–nullity.
- Liên hệ rank với thông tin thừa và nghiệm của hệ tuyến tính.

## Trọng tâm

Range của $A$ là $\{Ax:x\in\mathbb{R}^n\}$, tức span các cột. Null space là $\{x:Ax=0\}$. Rank là số chiều của column space và bằng số chiều row space. Với $A\in\mathbb{R}^{m\times n}$: $\mathrm{rank}(A)+\mathrm{nullity}(A)=n$.

Full column rank cho nghiệm least-squares duy nhất; rank deficiency thể hiện đặc trưng phụ thuộc hoặc tham số không nhận dạng được. Trong số học, rank phụ thuộc tolerance vì singular value hiếm khi bằng đúng 0.

## Tự kiểm tra

1. Null space chứa ít nhất vector nào?
2. Rank tối đa của ma trận $m\times n$ là bao nhiêu?
3. Nếu hai feature giống hệt nhau, rank của design matrix thay đổi ra sao?

## Checklist

- [ ] Tìm basis của null space bằng SVD/`scipy.linalg.null_space`.
- [ ] Kiểm tra rank–nullity.
- [ ] Giải thích rank deficiency trong linear regression.

