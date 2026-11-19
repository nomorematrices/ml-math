# Đóng góp nội dung

## Quy ước bài học

Mỗi thư mục bài học phải có:

- `notes.md`: mục tiêu, trọng tâm, tự kiểm tra và checklist.
- `practice.ipynb`: mục tiêu, code tái lập với seed 42, bài tập/TODO, assertion và tổng kết.

Mọi công thức cần nhất quán với `docs/notation.md`. Code phải ghi rõ shape ở nơi dễ nhầm và tránh nghịch đảo tường minh khi có thể dùng linear solve.

## Kiểm tra trước khi commit

```bash
make test
make notebooks
make lint
```

Notebook không commit dữ liệu lớn, secret, checkpoint hoặc output không cần thiết. Nếu thêm dependency, cập nhật cả `requirements.txt` và `environment.yml`.

