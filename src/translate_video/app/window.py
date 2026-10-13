"""Always-on-top control window."""

from dataclasses import asdict
import math
import os

from PySide6.QtCore import QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSlider, QTextEdit, QVBoxLayout, QWidget,
)

from .client import EngineClient, Settings

SOURCES = [("auto", "Tự động nhận diện"), ("English", "English"), ("Chinese", "中文 (Chinese)")]
SEGMENTATION = [("punctuation", "Cắt theo dấu phẩy (trễ thấp)"), ("semantic", "Cắt theo ngữ nghĩa (trọn ý hơn)")]
STAGES = {"asr": "Qwen3-ASR", "nllb": "NLLB dịch", "tts": "VieNeu TTS"}
STYLE = """
QWidget { background: #1e2127; color: #e6e6e6; font-size: 13px; }
QPushButton { background: #2f6fed; border: none; border-radius: 6px; padding: 8px 14px; font-weight: 600; }
QPushButton:disabled { background: #3a3f4a; color: #8a8f99; }
QPushButton#secondary { background: #3a3f4a; font-weight: 400; padding: 5px 10px; }
QTextEdit { background: #15181d; border: 1px solid #2c313a; border-radius: 6px; }
QProgressBar { background: #15181d; border: 1px solid #2c313a; border-radius: 4px; height: 8px; }
QProgressBar::chunk { background: #38b36a; border-radius: 3px; }
QLabel#muted { color: #8a8f99; }
"""


def level_percent(rms: float) -> int:
    return max(0, min(100, round((20 * math.log10(max(rms, 1e-6)) + 60) / 60 * 100)))


class MainWindow(QWidget):
    def __init__(self, client: EngineClient):
        super().__init__()
        self.client = client
        self.state = "stopped"   # stopped | loading | ready | listening | error
        self.want_listening = False
        self.output_dir = None
        self.settings = QSettings("translate-video", "app")
        self.setWindowTitle("Translate Video — EN → VI")
        self.setStyleSheet(STYLE)
        self.resize(self.settings.value("size", None) or QSize(460, 560))
        self.build()
        self.restore()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)
        self.render()

    # ---- layout ----
    def build(self):
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.button = QPushButton("Khởi động")
        self.button.clicked.connect(self.toggle)
        self.status = QLabel("Chưa khởi động")
        self.status.setWordWrap(True)
        top.addWidget(self.button)
        top.addWidget(self.status, 1)
        layout.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        self.live = QLabel("")
        self.live.setObjectName("muted")
        self.live.setWordWrap(True)
        layout.addWidget(self.live)
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.document().setMaximumBlockCount(400)
        layout.addWidget(self.transcript, 1)

        self.volume = self.slider("Âm lượng giọng dịch", 0, 200, 100, self.on_volume)
        self.speed = self.slider("Tốc độ đọc cơ bản", 100, 140, 110, self.on_speed, suffix="x", scale=100)
        self.opacity = self.slider("Độ trong suốt cửa sổ", 35, 100, 100, self.on_opacity)
        layout.addLayout(self.volume[0])
        layout.addLayout(self.speed[0])
        layout.addLayout(self.opacity[0])

        source = QHBoxLayout()
        source.addWidget(QLabel("Ngôn ngữ nguồn"))
        self.source = QComboBox()
        for key, label in SOURCES:
            self.source.addItem(label, key)
        source.addWidget(self.source, 1)
        layout.addLayout(source)

        segmentation = QHBoxLayout()
        segmentation.addWidget(QLabel("Cách cắt câu"))
        self.segmentation = QComboBox()
        for key, label in SEGMENTATION:
            self.segmentation.addItem(label, key)
        segmentation.addWidget(self.segmentation, 1)
        layout.addLayout(segmentation)

        options = QHBoxLayout()
        self.pin = QCheckBox("Ghim trên cùng")
        self.pin.toggled.connect(self.on_pin)
        self.save_audio = QCheckBox("Lưu audio dịch")
        options.addWidget(self.pin)
        options.addWidget(self.save_audio)
        options.addStretch(1)
        self.logs = QPushButton("Mở thư mục log")
        self.logs.setObjectName("secondary")
        self.logs.clicked.connect(self.open_logs)
        options.addWidget(self.logs)
        layout.addLayout(options)

        self.footer = QLabel("")
        self.footer.setObjectName("muted")
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)

    def slider(self, text, low, high, value, handler, suffix="%", scale=1):
        row = QHBoxLayout()
        label = QLabel(text)
        label.setMinimumWidth(150)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)

        def show(v):
            return f"{v / scale:.2f}x" if suffix == "x" else f"{v}%"

        value_label = QLabel(show(value))
        value_label.setMinimumWidth(40)

        def changed(v):
            value_label.setText(show(v))
            handler(v)

        slider.valueChanged.connect(changed)
        row.addWidget(label)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        return row, slider

    def restore(self):
        def get(key, default, kind):
            try:
                return self.settings.value(key, default, kind)
            except Exception:
                return default

        self.volume[1].setValue(get("volume", 100, int))
        self.speed[1].setValue(get("speed", 110, int))
        self.source.setCurrentIndex(max(self.source.findData(get("source", "auto", str)), 0))
        index = self.segmentation.findData(get("segmentation", "punctuation", str))
        self.segmentation.setCurrentIndex(max(index, 0))
        self.opacity[1].setValue(get("opacity", 100, int))
        pinned = get("pin", True, bool)
        self.pin.setChecked(pinned)
        self.save_audio.setChecked(get("save_audio", False, bool))
        self.on_pin(pinned)
        position = self.settings.value("pos", None)
        if position is not None:
            self.move(position)

    # ---- user actions ----
    def toggle(self):
        if self.state == "listening":
            self.want_listening = False
            self.client.send("stop")
            self.set_status("Đang dừng…")
            self.button.setEnabled(False)
            return
        self.want_listening = True
        self.live.setText("")
        self.client.ensure_running()
        settings = self.current_settings()
        if self.state in ("stopped", "error"):
            self.state = "loading"
            self.client.send("load", settings)
        else:
            self.client.send("start", settings)
        self.render()

    def on_volume(self, value):
        if self.client.alive:
            self.client.send("gain", value / 100)

    def on_speed(self, value):
        if self.client.alive:
            self.client.send("speed", value / 100)

    def current_settings(self):
        return asdict(Settings(
            gain=self.volume[1].value() / 100, save_audio=self.save_audio.isChecked(),
            speed=self.speed[1].value() / 100, max_speed=max(1.4, self.speed[1].value() / 100),
            segmentation=self.segmentation.currentData(), source_language=self.source.currentData()))

    def on_opacity(self, value):
        self.setWindowOpacity(value / 100)

    def on_pin(self, pinned):
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowStaysOnTopHint, pinned)
        if visible:
            self.show()   # changing window flags hides the window

    def open_logs(self):
        target = self.output_dir or os.path.join(str(self.client.root), "output")
        os.makedirs(target, exist_ok=True)
        os.startfile(target)

    # ---- engine events ----
    def poll(self):
        for event in self.client.poll():
            self.handle(event)
        if self.state in ("loading", "listening") and self.client.process is not None and not self.client.alive:
            self.fail("Engine đã dừng đột ngột. Xem .cache\\engine.log")

    def handle(self, event):
        kind = event["kind"]
        if kind == "loading":
            self.set_status(f"Đang tải model {event['step']}/{event['steps']}: {STAGES.get(event['stage'], event['stage'])}…")
            self.progress.setRange(0, event["steps"])
            self.progress.setValue(event["step"] - 1)
        elif kind == "ready":
            seconds = sum(event["load_seconds"].values())
            self.state = "ready"
            self.footer.setText(f"Đã tải model trong {seconds:.0f}s · GPU {event['gpu']}")
            if self.want_listening:
                self.client.send("start", self.current_settings())
        elif kind == "status":
            state = event["state"]
            if state in ("starting", "calibrating"):
                self.state = "loading"
                self.set_status("Đang hiệu chuẩn âm thanh (sẽ có tiếng 'xì' rất nhỏ)…" if state == "calibrating"
                                else "Đang mở thiết bị âm thanh…")
            elif state == "listening":
                self.state = "listening"
                self.output_dir = event.get("output")
                self.set_status("Đang nghe toàn bộ âm thanh của máy")
            elif state == "idle" and self.state != "error":
                self.state = "ready"
                self.live.setText("")
                self.set_status("Đã dừng. Model vẫn được giữ trong bộ nhớ")
        elif kind == "calibration":
            if event["lag_ms"] is None:
                self.footer.setText("Không hiệu chuẩn được khử echo: sẽ tắt tiếng thu khi giọng dịch đang phát")
            else:
                self.footer.setText(f"Khử echo OK · thiết bị {event['device']} · độ trễ {event['lag_ms']:.0f} ms")
        elif kind == "level":
            self.progress.setRange(0, 100)
            self.progress.setValue(level_percent(event["rms"]))
        elif kind == "pipeline":
            self.on_pipeline(event["event"])
        elif kind == "error":
            self.fail(event["message"])
        self.render()

    def on_pipeline(self, event):
        kind = event["type"]
        if kind == "partial":
            self.live.setText("… " + event["text"])
        elif kind == "translated":
            self.live.setText("")
            self.append(event["source"], event["translation"])
        elif kind == "segment":
            self.footer.setText(
                f"Trễ chốt câu→loa {event.get('commit_to_speaker_ms', 0) / 1000:.1f}s · "
                f"tốc độ đọc {event['speed']:.2f}x · dịch {event['translation_ms']:.0f} ms · "
                f"TTS chunk đầu {event.get('tts_first_chunk_ms', 0):.0f} ms")
        elif kind == "dropped":
            self.note(f"Bỏ qua một câu cũ vì trễ quá lâu ({event['waited_ms'] / 1000:.0f}s)")
        elif kind == "skipped":
            self.note("Bỏ qua câu đã là tiếng Việt")
        elif kind == "translation_failed":
            self.note("Dịch lỗi một câu, đã bỏ qua và tiếp tục")
        elif kind == "tts_failed":
            self.note("Đọc lỗi một câu, đã bỏ qua và tiếp tục")
        elif kind == "revision_reset":
            self.note("ASR đổi nội dung đã chốt; đã đồng bộ lại")

    def append(self, source, translation):
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.transcript.setTextCursor(cursor)
        self.transcript.append(f'<span style="color:#8a8f99">{escape(source)}</span><br>'
                               f'<b style="color:#7fe0a3">{escape(translation)}</b>')
        self.transcript.verticalScrollBar().setValue(self.transcript.verticalScrollBar().maximum())

    def note(self, text):
        self.transcript.append(f'<i style="color:#d9a441">{escape(text)}</i>')

    def fail(self, message):
        self.state = "error"
        self.want_listening = False
        self.set_status("Lỗi: " + message)
        self.render()

    def set_status(self, text):
        self.status.setText(text)

    # ---- state -> widgets ----
    def render(self):
        labels = {"stopped": "Khởi động", "loading": "Đang chuẩn bị…", "ready": "Bắt đầu nghe",
                  "listening": "Dừng", "error": "Thử lại"}
        self.button.setText(labels[self.state])
        self.button.setEnabled(self.state != "loading")
        for box in (self.segmentation, self.source):
            box.setEnabled(self.state != "listening")   # applies when listening starts
        if self.state == "stopped":
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.set_status("Nhấn Khởi động để tải model (~1 phút)")
        elif self.state == "loading" and self.progress.maximum() == 0:
            self.progress.setRange(0, 3)
        elif self.state == "ready":
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

    def closeEvent(self, event):
        self.settings.setValue("volume", self.volume[1].value())
        self.settings.setValue("speed", self.speed[1].value())
        self.settings.setValue("segmentation", self.segmentation.currentData())
        self.settings.setValue("source", self.source.currentData())
        self.settings.setValue("opacity", self.opacity[1].value())
        self.settings.setValue("pin", self.pin.isChecked())
        self.settings.setValue("save_audio", self.save_audio.isChecked())
        self.settings.setValue("pos", self.pos())
        self.settings.setValue("size", self.size())
        self.timer.stop()
        self.status.setText("Đang đóng…")
        self.client.shutdown()
        event.accept()


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
