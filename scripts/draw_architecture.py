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
    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98)
    ax.set(xlim=(0, 17.7), ylim=(0, 12))
    ax.axis("off")

    ax.text(0.7, 11.4, "Kiến trúc dịch giọng nói theo thời gian thực",
            fontsize=26, weight="bold", color=INK)
    ax.text(0.7, 10.96, "Mỗi khối làm gì? Mũi tên cho biết dữ liệu được truyền đi.",
            fontsize=15, color=MUTED)

    lane(ax, 8.0, 2.45, "LUỒNG 1 · Nhận audio và nhận dạng lời nói")
    block(ax, 0.8, 8.3, 3.6, "Nhận audio",
          ["File hoặc microphone", "Đọc từng mẩu âm thanh"])
    block(ax, 5.6, 8.3, 4.6, "Nhận dạng lời nói",
          ["Qwen3-ASR 1.7B", "Nghe audio → viết thành chữ"], "#2463c5", "#edf4ff")
    block(ax, 11.4, 8.3, 4.9, "Chốt đoạn để dịch",
          ["Giữ phần chữ đã ổn định", "Chọn đoạn, tránh gửi lặp"])
    arrow(ax, (4.4, 9.075), (5.6, 9.075), "Audio (PCM)")
    arrow(ax, (10.2, 9.075), (11.4, 9.075), "Chữ tạm")

    block(ax, 11.4, 6.9, 4.9, "Hàng đợi chữ",
          ["Giữ các đoạn tiếng Anh theo thứ tự"], "#b46a0e", "#fff6e9", height=0.96)
    arrow(ax, (13.85, 8.3), (13.85, 7.86), "Gửi đoạn", (15.15, 8.05))

    lane(ax, 4.5, 2.3, "LUỒNG 2 · Dịch đoạn chữ, sau đó tạo giọng")
    block(ax, 11.4, 4.72, 4.9, "Dịch sang tiếng Việt",
          ["NLLB-200 distilled 600M", "Chữ tiếng Anh → chữ tiếng Việt"], "#744ac4", "#f4efff")
    block(ax, 5.6, 4.72, 4.6, "Tạo giọng nói",
          ["MMS-TTS tiếng Việt (VITS)", "Chữ tiếng Việt → audio"], "#087e86", "#eaf8f6")
    block(ax, 0.8, 4.72, 3.6, "Hàng đợi audio",
          ["Giữ audio từng đoạn", "Chờ lượt lưu và phát"], "#b46a0e", "#fff6e9")
    arrow(ax, (13.85, 6.9), (13.85, 6.27), "Lấy đoạn", (15.15, 6.55))
    arrow(ax, (11.4, 5.495), (10.2, 5.495), "Chữ Việt")
    arrow(ax, (5.6, 5.495), (4.4, 5.495), "Audio Việt")

    lane(ax, 1.06, 2.55, "LUỒNG 3 · Lưu kết quả và phát âm thanh")
    block(ax, 0.8, 1.38, 3.6, "Lưu audio",
          ["Ghi WAV từng đoạn", "Và file ghép translated.wav"], "#b46a0e", "#fff6e9")
    block(ax, 5.6, 1.38, 4.6, "Phát ra loa",
          ["Phát các đoạn đúng thứ tự", "Hoạt động khi bật --play"], "#b46a0e", "#fff6e9")
    block(ax, 11.4, 1.38, 4.9, "Theo dõi kết quả",
          ["Text gốc + bản dịch", "Thời gian xử lý từng bước"])
    route(ax, [(2.6, 4.72), (2.6, 4.05), (0.7, 4.05),
               (0.7, 3.07), (2.6, 3.07), (2.6, 2.93)])
    ax.text(2.8, 4.34, "Audio + dữ liệu đoạn", fontsize=11.5, color=MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    arrow(ax, (4.4, 2.155), (5.6, 2.155), "File WAV")

    # ASR sends partial text; the audio worker logs translation and timing data.
    route(ax, [(7.9, 9.85), (7.9, 10.62), (16.85, 10.62),
               (16.85, 2.155), (16.3, 2.155)], dashed=True)
    route(ax, [(2.6, 1.38), (2.6, 0.85), (13.85, 0.85),
               (13.85, 1.38)], dashed=True)

    ax.text(0.7, 0.4, "Mũi tên liền: dữ liệu chính  ·  Nét đứt: nhật ký  ·  "
            "Gọi hàm và hàng đợi trong Python", fontsize=12, color=MUTED)
    ax.text(0.7, 0.05, "Ba luồng chạy đồng thời: vẫn nghe đoạn mới khi đang dịch và phát đoạn trước.",
            fontsize=12, color=MUTED)

    output = Path(__file__).resolve().parents[1] / "docs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = output / f"architecture.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
