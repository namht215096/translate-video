"""Draw functional blocks and their data flow with Matplotlib.

Run: .venv/bin/python scripts/draw_architecture.py
Outputs: docs/figures/architecture.png and architecture.svg
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#18263b"
MUTED = "#56667b"


def lane(ax, bottom, height, title):
    ax.add_patch(FancyBboxPatch(
        (0.5, bottom), 16.7, height,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor="#f5f7fb", edgecolor="#dce3ec", linewidth=1.2,
    ))
    ax.text(0.9, bottom + height - 0.3, title,
            fontsize=14, weight="bold", color=INK)


def block(ax, x, y, width, title, lines, color=MUTED, tint="white", height=1.55):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.13",
        facecolor=tint, edgecolor=color, linewidth=1.7,
    ))
    ax.text(x + 0.2, y + height - 0.35, title,
            fontsize=17, weight="bold", color=color)
    ax.text(x + 0.2, y + (height - 0.5) / 2, "\n".join(lines),
            fontsize=13.5, color=INK, va="center", linespacing=1.55)


def arrow(ax, start, end, label=None, label_xy=None, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=17,
        linewidth=1.6, color=MUTED,
        linestyle=(0, (4, 3)) if dashed else "-", shrinkA=3, shrinkB=3,
    ))
    if label:
        xy = label_xy or ((start[0] + end[0]) / 2, start[1] + 0.27)
        ax.text(*xy, label, fontsize=11.5, color=MUTED, ha="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})


def route(ax, points, dashed=False):
    ax.plot([point[0] for point in points[:-1]],
            [point[1] for point in points[:-1]], color=MUTED,
            linewidth=1.5, linestyle=(0, (4, 3)) if dashed else "-")
    arrow(ax, points[-2], points[-1], dashed=dashed)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(18, 10.8))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98)
    ax.set(xlim=(0, 17.7), ylim=(0, 10.8))
    ax.axis("off")

    ax.text(0.7, 10.2, "Kiến trúc dịch giọng nói theo thời gian thực",
            fontsize=26, weight="bold", color=INK)
    ax.text(0.7, 9.77, "Mỗi khối làm gì? Mũi tên cho biết dữ liệu được truyền đi.",
            fontsize=15, color=MUTED)

    lane(ax, 7.05, 2.4, "LUỒNG 1 · Nhận audio và nhận dạng lời nói")
    block(ax, 0.8, 7.32, 3.6, "Nhận audio",
          ["File hoặc microphone", "Đọc từng mẩu âm thanh"])
    block(ax, 5.6, 7.32, 4.6, "Nhận dạng lời nói",
          ["Qwen3-ASR 1.7B", "Nghe audio → viết thành chữ"], "#2463c5", "#edf4ff")
    block(ax, 11.4, 7.32, 4.9, "Chốt đoạn để dịch",
          ["Giữ phần chữ đã ổn định", "Chọn đoạn, tránh gửi lặp"])
    arrow(ax, (4.4, 8.095), (5.6, 8.095), "Audio (PCM)")
    arrow(ax, (10.2, 8.095), (11.4, 8.095), "Chữ tạm")

    block(ax, 11.4, 5.92, 4.9, "Hàng đợi chữ",
          ["Giữ các đoạn tiếng Anh theo thứ tự"], "#b46a0e", "#fff6e9", height=0.96)
    arrow(ax, (13.85, 7.32), (13.85, 6.88), "Gửi đoạn", (15.15, 7.07))

    lane(ax, 3.25, 2.5, "LUỒNG 2 · Dịch → tạo giọng → phát trực tiếp")
    block(ax, 11.4, 3.49, 4.9, "Dịch sang tiếng Việt",
          ["NLLB-200 distilled 600M", "Chữ tiếng Anh → chữ tiếng Việt"], "#744ac4", "#f4efff")
    block(ax, 5.6, 3.49, 4.6, "Tạo giọng nói",
          ["VieNeu v3 Turbo / MMS-TTS", "Chữ tiếng Việt → audio"], "#087e86", "#eaf8f6")
    block(ax, 0.8, 3.49, 3.6, "Phát ra loa",
          ["Nhận audio trong RAM", "Phát trực tiếp (--play)"], "#b46a0e", "#fff6e9")
    arrow(ax, (13.85, 5.92), (13.85, 5.04), "Lấy đoạn", (15.15, 5.45))
    arrow(ax, (11.4, 4.265), (10.2, 4.265), "Chữ Việt")
    arrow(ax, (5.6, 4.265), (4.4, 4.265), "PCM / chunk")

    ax.text(8.85, 2.86, "Phát xong một đoạn mới dịch và tạo giọng cho đoạn kế tiếp.",
            ha="center", fontsize=14, color=MUTED)
    block(ax, 5.6, 0.75, 4.6, "Nhật ký & cấu hình",
          ["Text + thời gian xử lý", "Chỉ lưu JSON"])

    # Both execution lanes record text and timings in the shared JSON event log.
    route(ax, [(7.9, 8.87), (7.9, 9.58), (16.85, 9.58),
               (16.85, 1.525), (10.2, 1.525)], dashed=True)
    route(ax, [(2.6, 3.49), (2.6, 2.48), (7.9, 2.48), (7.9, 2.3)], dashed=True)

    ax.text(0.7, 0.32, "Mũi tên liền: dữ liệu chính  ·  Nét đứt: nhật ký  ·  "
            "Gọi hàm và hàng đợi trong Python", fontsize=12, color=MUTED)
    ax.text(0.7, 0.01, "Hai luồng chạy đồng thời: ASR vẫn nghe khi luồng dịch/tạo giọng/phát đang bận.",
            fontsize=12, color=MUTED)

    output = Path(__file__).resolve().parents[1] / "docs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = output / f"architecture.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        if extension == "svg":
            # Matplotlib adds spaces at the ends of SVG path-data lines.
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
