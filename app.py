import warnings

warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype() is deprecated"
)
import sys
import time
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QComboBox, QMessageBox
)
import mido

from engine import EngineWorker
from midi import init_midi, send_note, close_midi

class GestureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Gesture MIDI Controller")
        self.setFixedSize(400, 300)
        self.setWindowIcon(QIcon("logo.png"))
        self.worker = None
        self.midi_initialized = False
        self.running = False

        layout = QVBoxLayout()

        # -------- Camera selector --------
        layout.addWidget(QLabel("Camera"))
        self.cam_box = QComboBox()
        self.cam_box.addItems(["0", "1", "2"])
        layout.addWidget(self.cam_box)

        # -------- MIDI selector --------
        layout.addWidget(QLabel("MIDI Output"))
        self.midi_box = QComboBox()

        # 🔥 FILTER OUT MICROSOFT GS WAVETABLE (IMPORTANT)
        midi_ports = [
            p for p in mido.get_output_names()
            if "Microsoft GS" not in p
        ]
        self.midi_box.addItems(midi_ports)

        layout.addWidget(self.midi_box)

        # =======Selector of the orientation================

        layout.addWidget(QLabel("Orientation"))
        self.orientation_box = QComboBox()
        self.orientation_box.addItems(["Landscape","Portrait"])
        layout.addWidget(self.orientation_box)

        # -------- Status --------
        self.status = QLabel("Idle")
        layout.addWidget(self.status)

        # -------- Buttons --------
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

        self.setLayout(layout)

        self.start_btn.clicked.connect(self.start_engine)
        self.stop_btn.clicked.connect(self.stop_engine)

    # =========================
    # ENGINE CONTROL
    # =========================
    
    def start_engine(self):
        if self.running:
            QMessageBox.warning(self, "Running", "Engine already running")
            return

        cam = int(self.cam_box.currentText())  # ✅ always fresh
        midi_name = self.midi_box.currentText()
        orientation = self.orientation_box.currentText()


        try:
            init_midi(midi_name)
            self.midi_initialized = True
        except Exception as e:
            QMessageBox.critical(self, "MIDI Error", str(e))
            return

        # ✅ EngineWorker now takes ONLY camera number
        self.worker = EngineWorker(cam,orientation)
        self.worker.status_signal.connect(self.status.setText)
        self.worker.gesture_signal.connect(self.on_gesture)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

        self.running = True

    def stop_engine(self):
        if not self.running:
            return

        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

        self.cleanup_midi()
        self.running = False


    # =========================
    # GESTURE → MIDI
    # =========================

    def on_gesture(self, gesture):
        print("GESTURE:", gesture)

        if not self.midi_initialized:
            return

        try:
            if gesture == "closed":
                send_note(60)   # C4
            elif gesture == "two":
                send_note(64)   # E4
            elif gesture == "four":
                send_note(67)   # G4
            elif gesture == "one":
                send_note(69)
            elif gesture == "three":
                send_note(71)
        except Exception as e:
            print("MIDI send error:", e)

    # =========================
    # CLEANUP
    # =========================

    def cleanup_midi(self):
        if self.midi_initialized:
            close_midi()
            self.midi_initialized = False
    def on_worker_finished(self):
        self.worker = None
        self.status.setText("Idle")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureApp()
    window.show()
    sys.exit(app.exec())


### to debug - if one hand is nothing and other hand is a gesture the nothing hand is also fed into the tree