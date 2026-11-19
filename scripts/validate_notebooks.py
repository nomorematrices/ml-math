"""Kiểm tra tất cả notebook là JSON hợp lệ và có các phần bắt buộc."""

import json
from pathlib import Path


def main() -> None:
    notebooks = sorted(Path("chapters").glob("**/practice.ipynb"))
    if len(notebooks) != 21:
        raise SystemExit(f"Expected 21 notebooks, found {len(notebooks)}")
    required = ("Mục tiêu", "Bài tập", "Tổng kết")
    for path in notebooks:
        data = json.loads(path.read_text(encoding="utf-8"))
        text = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])
        missing = [heading for heading in required if heading not in text]
        if missing:
            raise SystemExit(f"{path}: missing {missing}")
    print(f"Validated {len(notebooks)} notebooks.")


if __name__ == "__main__":
    main()
