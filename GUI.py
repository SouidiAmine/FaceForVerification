import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QComboBox, QMessageBox, QFrame
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from deepface import DeepFace
import cv2


class FaceVerificationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Verification - DeepFace")
        self.setGeometry(300, 150, 950, 550)
        self.setStyleSheet("""
            QWidget {
                background-color: #f4f6fa;
                font-family: Segoe UI;
            }
            QPushButton {
                background-color: #0078D7;
                color: white;
                border-radius: 8px;
                padding: 8px 15px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #005a9e; }
            QComboBox {
                padding: 5px;
                border-radius: 5px;
                border: 1px solid #ccc;
                background-color: white;
            }
            QLabel {
                font-size: 15px;
            }
        """)

        # Variables
        self.img1_path = None
        self.img2_path = None
        self.model_name = "Facenet"

        # Title
        title = QLabel("🔍 Face Verification System")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: #0078D7;")

        # Model selector
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Facenet", "VGG-Face"])
        self.model_selector.currentTextChanged.connect(self.set_model)

        # Image labels
        self.label_img1 = QLabel()
        self.label_img2 = QLabel()
        for lbl in [self.label_img1, self.label_img2]:
            lbl.setFixedSize(300, 300)
            lbl.setFrameShape(QFrame.Box)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color: #fff; color: #aaa; border-radius: 10px;")
            lbl.setText("No image loaded")

        # Buttons
        self.btn_load_img1 = QPushButton("📁 Load Image 1")
        self.btn_load_img2 = QPushButton("📁 Load Image 2")
        self.btn_webcam = QPushButton("📷 Capture from Webcam")
        self.btn_verify = QPushButton("✅ Verify Identity")

        for b in [self.btn_load_img1, self.btn_load_img2, self.btn_webcam, self.btn_verify]:
            b.setCursor(Qt.PointingHandCursor)

        # Result label
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Segoe UI", 16, QFont.Bold))

        # Layouts
        layout_main = QVBoxLayout()
        layout_imgs = QHBoxLayout()
        layout_buttons = QHBoxLayout()

        layout_imgs.addWidget(self.label_img1)
        layout_imgs.addWidget(self.label_img2)

        layout_buttons.addWidget(self.btn_load_img1)
        layout_buttons.addWidget(self.btn_load_img2)
        layout_buttons.addWidget(self.btn_webcam)
        layout_buttons.addWidget(self.btn_verify)

        layout_main.addWidget(title)
        layout_main.addWidget(QLabel("Select Model:"))
        layout_main.addWidget(self.model_selector)
        layout_main.addSpacing(10)
        layout_main.addLayout(layout_imgs)
        layout_main.addSpacing(10)
        layout_main.addLayout(layout_buttons)
        layout_main.addSpacing(15)
        layout_main.addWidget(self.result_label)

        self.setLayout(layout_main)

        # Connections
        self.btn_load_img1.clicked.connect(self.load_image1)
        self.btn_load_img2.clicked.connect(self.load_image2)
        self.btn_webcam.clicked.connect(self.capture_webcam)
        self.btn_verify.clicked.connect(self.verify_faces)

    def set_model(self, model_name):
        self.model_name = model_name

    def load_image(self, label_attr):
        path, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            setattr(self, label_attr + "_path", path)
            label = getattr(self, "label_" + label_attr)
            pixmap = QPixmap(path).scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pixmap)
            label.setStyleSheet("border: 2px solid #0078D7; border-radius: 10px;")

    def load_image1(self): 
        self.load_image("img1")

    def load_image2(self): 
        self.load_image("img2")

    def capture_webcam(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            path = "webcam_capture.jpg"
            cv2.imwrite(path, frame)
            self.img2_path = path
            pixmap = QPixmap(path).scaled(300, 300, Qt.KeepAspectRatio)
            self.label_img2.setPixmap(pixmap)
        else:
            QMessageBox.warning(self, "Error", "Unable to access the webcam.")

    def verify_faces(self):
        
        if not self.img1_path or not self.img2_path:

            QMessageBox.warning(self, "Error", "Please load both images before verifying.")
            return
        try:
            self.result_label.setText("⏳ Verifying...")
            self.result_label.setStyleSheet("color: black; font-weight: bold;")
            QApplication.processEvents()  # 🟢 Force UI to update here
            print("Verifying...")
            result = DeepFace.verify(
                img1_path=self.img1_path,
                img2_path=self.img2_path,
                model_name=self.model_name
            )
            verified = result["verified"]
            if verified:
                self.result_label.setText("✅ Same Person Detected")
                self.result_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.result_label.setText("❌ Different Person")
                self.result_label.setStyleSheet("color: red; font-weight: bold;")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Verification failed:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceVerificationApp()
    window.show()
    sys.exit(app.exec_())
