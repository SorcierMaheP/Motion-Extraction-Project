import os
import sys
import time
import platform
import numpy as np

import cv2 as cv
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)
import qdarktheme


class Thread(QThread):
    origFrame = Signal(QImage)
    procFrame = Signal(QImage)
    blendFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.status = True
        self.update = True
        self.cap = True

    def run(self):
        self.cap = cv.VideoCapture(*cam)
        currentFrameNumber = 0
        self.cap.set(cv.CAP_PROP_FPS, 60)

        _, previousFrame = self.cap.read()
        previousFrame = cv.cvtColor(previousFrame, cv.COLOR_BGR2GRAY)

        while self.status:
            # capture each frame from the video feed -----------------------------------------------------------------------------------------------------------
            if self.update:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                if frame is None:
                    break

                #  process each frame here:   ----------------------------------------------------------------------------------------------------------------------
                orig_frame = frame
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = cv.GaussianBlur(frame, (5, 5), 0)

                currentFrameNumber += 1

                frameList.append(frame)

                if currentFrameNumber >= prevFrameNumber:

                    currentFrameNumber = 0
                    previousFrame = frameList.pop(0)
                    previousFrame = cv.bitwise_not(previousFrame)
                    frameList.clear()

                proc_frame = cv.addWeighted(frame, 0.5, previousFrame, 0.5, 0)

                _, mask = cv.threshold(proc_frame, 140, 255, cv.THRESH_BINARY)
                height, width = mask.shape

                solid_color_frame = np.full(
                    (height, width, 3), (255, 0, 0), dtype=np.uint8
                )

                mask = cv.bitwise_and(solid_color_frame, solid_color_frame, mask=mask)
                blend_frame = cv.addWeighted(orig_frame, 0.4, mask, 0.6, 0)

                # Creating and scaling QImage
                h, w, ch = orig_frame.shape
                img_orig = QImage(orig_frame.data, w, h, w * ch, QImage.Format_BGR888)
                scaled_img_orig = img_orig.scaled(800, 600, Qt.KeepAspectRatio)

                h, w = proc_frame.shape
                img_proc = QImage(proc_frame.data, w, h, w, QImage.Format_Grayscale8)
                scaled_img_proc = img_proc.scaled(800, 600, Qt.KeepAspectRatio)

                h, w, ch = blend_frame.shape
                img_blend = QImage(blend_frame.data, w, h, w * ch, QImage.Format_BGR888)
                scaled_img_blend = img_blend.scaled(800, 600, Qt.KeepAspectRatio)

                # Emit signals
                self.origFrame.emit(scaled_img_orig)
                self.procFrame.emit(scaled_img_proc)
                self.blendFrame.emit(scaled_img_blend)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        font = QFont("JetBrains Mono", 14)
        QApplication.instance().setFont(font)
        # Title and dimensions
        self.setWindowTitle("Motion Extractor")
        self.setGeometry(0, 0, 1600, 900)

        # Create a label for the display camera
        self.label1 = QLabel(self)
        self.label1.setFixedSize(500, 600)
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setStyleSheet(
            "border :5px solid ;"
            "border-top-color : red; "
            "border-left-color :pink;"
            "border-right-color :yellow;"
            "border-bottom-color : green"
        )

        # Add text label for label1
        self.label1_text = QLabel("Original Video Feed", self)
        self.label1_text.setAlignment(Qt.AlignCenter)

        self.label2 = QLabel(self)
        self.label2.setFixedSize(500, 600)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setStyleSheet(
            "border: 5px solid ;"
            "border-top-color: #888888; "  # Grey
            "border-left-color: #CCCCCC; "  # Light grey
            "border-right-color: #666666; "  # Dark grey
            "border-bottom-color: #AAAAAA;"  # Medium grey
        )

        # Add text label for label2
        self.label2_text = QLabel("Motion Extracted Video Feed", self)
        self.label2_text.setAlignment(Qt.AlignCenter)

        self.label3 = QLabel(self)
        self.label3.setFixedSize(500, 600)
        self.label3.setAlignment(Qt.AlignCenter)
        self.label3.setStyleSheet(
            "border: 5px solid ;"
            "border-top-color: red; "  # Red
            "border-left-color: #CCCCCC; "  # Light grey
            "border-right-color: #666666; "  # Dark grey
            "border-bottom-color: green;"  # Green
        )

        # Add text label for label3
        self.label3_text = QLabel("Blended Video Feed", self)
        self.label3_text.setAlignment(Qt.AlignCenter)

        # Thread in charge of updating the image
        self.th = Thread()
        self.th.origFrame.connect(self.setOrigImage)
        self.th.procFrame.connect(self.setProcImage)
        self.th.blendFrame.connect(self.setBlendImage)
        self.th.finished.connect(self.close)

        model_layout = QHBoxLayout()

        # Buttons layout
        buttons_layout = QHBoxLayout()
        self.button1 = QPushButton("START")
        self.button1.setStyleSheet("color: green")
        self.button2 = QPushButton("CLOSE")
        button_width = 100
        button_height = 40
        self.button1.setFixedSize(button_width, button_height)
        self.button2.setFixedSize(button_width, button_height)
        buttons_layout.addWidget(self.button2)
        buttons_layout.addWidget(self.button1)
        buttons_layout.setContentsMargins(20, 5, 5, 20)

        # Defining file picker button
        self.file_picker_button = QPushButton("Open File")
        self.file_picker_button.clicked.connect(self.open_file_dialog)
        self.file_picker_button.setFixedSize(200, 40)

        # Vertical layout for label1 and its text label
        label1_layout = QVBoxLayout()
        label1_layout.setAlignment(Qt.AlignTop)
        label1_layout.addWidget(self.label1_text)
        label1_layout.addWidget(self.label1)

        # Vertical layout for label2 and its text label
        label2_layout = QVBoxLayout()
        label2_layout.setAlignment(Qt.AlignTop)
        label2_layout.addWidget(self.label2_text)
        label2_layout.addWidget(self.label2)

        # Vertical layout for label3 and its text label
        label3_layout = QVBoxLayout()
        label3_layout.setAlignment(Qt.AlignTop)
        label3_layout.addWidget(self.label3_text)
        label3_layout.addWidget(self.label3)

        # Horizontal layout for all label layouts
        labels_layout = QHBoxLayout()
        labels_layout.addLayout(label1_layout)
        labels_layout.addLayout(label2_layout)
        labels_layout.addLayout(label3_layout)
        labels_layout.setContentsMargins(0, 20, 0, 0)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.file_picker_button)
        layout.addLayout(labels_layout)
        layout.addLayout(buttons_layout, 0)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Connections
        self.button1.clicked.connect(self.toggle_thread)
        self.button2.clicked.connect(self.kill_thread)
        self.button2.setEnabled(False)

    @Slot()
    def open_file_dialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "All Files (*)",
            options=options,
        )
        if fileName:
            global cam
            cam = (fileName,)
            self.file_picker_button.setEnabled(False)
            self.file_picker_button.setStyleSheet("color:grey")

    @Slot()
    def kill_thread(self):
        print("Finishing...")
        self.th.wait(1)
        self.button2.setEnabled(False)
        self.button1.setEnabled(False)
        self.button1.setStyleSheet("color:grey")
        self.button2.setStyleSheet("color:grey")
        cv.destroyAllWindows()
        self.th.status = False

    @Slot()
    def toggle_thread(self):
        if not self.button2.isEnabled():
            self.button2.setStyleSheet("color:red")
            self.button2.setEnabled(True)
        if self.button1.text() == "START":
            print("Starting...")
            if self.th.update:
                self.th.start()
            self.th.update = True
            self.button1.setText("STOP")
            self.button1.setStyleSheet("color: blue")
        else:
            print("Stopping...")
            self.th.update = False
            self.button1.setText("START")
            self.button1.setStyleSheet("color: green")

    @Slot(QImage)
    def setOrigImage(self, image):
        pixmap = QPixmap.fromImage(image)
        self.label1.setPixmap(pixmap)

    @Slot(QImage)
    def setProcImage(self, image):
        pixmap = QPixmap.fromImage(image)
        self.label2.setPixmap(pixmap)

    @Slot(QImage)
    def setBlendImage(self, image):
        pixmap = QPixmap.fromImage(image)
        self.label3.setPixmap(pixmap)


if __name__ == "__main__":
    # list to store previous frames, needed to compute temporal changes
    # using it as a buffer here, with length = prevFrameNumber
    frameList = []

    # number of previous frames to process:
    prevFrameNumber = 3

    # treshold for the final output:
    treshold = 150

    # detects os of user:
    userPlatform = platform.system()

    # selecting the videocapture backend in Linux is necessary in some cases
    # change 0 to whatever webcam id is required
    if userPlatform == "Linux":
        cam = (0, cv.CAP_V4L2)
    elif userPlatform in ["Windows", "Darwin"]:
        cam = (0,)

    app = QApplication()
    qdarktheme.setup_theme("auto")
    w = Window()
    w.show()
    sys.exit(app.exec())
