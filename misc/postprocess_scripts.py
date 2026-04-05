"""
nbconvert で生成した .py ファイルを uv inline script (PEP 723) 形式に整えるスクリプト。

処理内容:
  - !pip install → PEP 723 ヘッダに変換
  - Google Colab 固有コードを除去 (from google.colab, drive.mount, Drive へのコピー)
  - 残った !shell コマンドをコメントアウト
  - # In[ ]: マーカー / shebang / coding 宣言を除去
  - 連続空行を1行に圧縮
"""

import re
import sys
from pathlib import Path


# 対象ファイル (各章ディレクトリの .py)
TARGET_DIRS = [
    "chapter01",
    "chapter02",
    "chapter03",
    "chapter05",
    "chapter07",
    "chapter08",
]


def parse_pip_install(line: str) -> list[str]:
    """!pip install 行からパッケージ名リストを抽出する。"""
    line = re.sub(r"^!pip install\s+", "", line.strip())
    packages = []
    for part in re.findall(r"'[^']*'|\"[^\"]*\"|\S+", line):
        pkg = part.strip("'\"")
        if pkg and not pkg.startswith("-"):
            packages.append(pkg)
    return packages


def format_uv_header(packages: list[str], python_version: str = ">=3.10") -> str:
    """PEP 723 uv inline script メタデータを生成する。"""
    lines = [
        "# /// script",
        f'# requires-python = "{python_version}"',
        "# dependencies = [",
    ]
    for pkg in packages:
        lines.append(f'#   "{pkg}",')
    lines.append("# ]")
    lines.append("# ///")
    return "\n".join(lines)


# Colab 固有行を検出するパターン
COLAB_PATTERNS = [
    r"^from google\.colab\b",
    r"^import google\.colab\b",
    r"drive\.mount\(",
    r"^!mkdir.*[Dd]rive",
    r"^!cp.*[Dd]rive",
]


def is_colab_line(line: str) -> bool:
    s = line.strip()
    return any(re.search(p, s) for p in COLAB_PATTERNS)


def process_file(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    packages: list[str] = []
    output_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # shebang / coding 宣言
        if re.match(r"^#!/usr/bin/env python", stripped):
            continue
        if re.match(r"^# -\*- coding", stripped) or re.match(r"^# coding:", stripped):
            continue

        # IPython セルマーカー "# In[ ]:"
        if re.match(r"^# In\[.*?\]:$", stripped):
            continue

        # !pip install → パッケージ収集のみ、行は削除
        if re.match(r"^!pip install\b", stripped):
            packages.extend(parse_pip_install(stripped))
            continue

        # Colab 固有コード → 削除
        if is_colab_line(line):
            continue

        # その他の !shell コマンド → コメントアウト
        if stripped.startswith("!"):
            output_lines.append(f"# {stripped}")
            continue

        output_lines.append(line)

    # 先頭の空行を除去
    while output_lines and not output_lines[0].strip():
        output_lines.pop(0)

    # PEP 723 ヘッダを先頭に付与
    final_lines: list[str] = []
    if packages:
        final_lines.append(format_uv_header(packages))
        final_lines.append("")
    final_lines.extend(output_lines)

    # 連続空行を最大1行に圧縮
    result: list[str] = []
    prev_blank = False
    for line in final_lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        result.append(line)
        prev_blank = is_blank

    path.write_text("\n".join(result) + "\n", encoding="utf-8")
    print(f"  processed: {path}")


def main() -> None:
    root = Path(__file__).parent.parent
    for dir_name in TARGET_DIRS:
        chapter_dir = root / dir_name
        py_files = sorted(chapter_dir.glob("*.py"))
        if not py_files:
            continue
        print(f"\n[{dir_name}]")
        for py_file in py_files:
            process_file(py_file)
    print("\nDone.")


if __name__ == "__main__":
    main()
