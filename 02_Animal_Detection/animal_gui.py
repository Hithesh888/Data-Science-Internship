import sys
import os
import cv2
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout,
    QWidget, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer


MODEL_PATH = r"D:\intern\project2\animal_detection\animal_model-4\weights\best.pt"

model = YOLO(MODEL_PATH)

CARNIVORES = {
    "Bear",
    "Leopard",
    "Lion",
    "Tiger"
}


class AnimalDetectionGUI(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Animal Detection System")
        self.setGeometry(100, 100, 1200, 800)

        self.cap = None
        self.timer = QTimer()

        self.current_image = None
        self.video_output = None

        self.setup_ui()

        self.timer.timeout.connect(self.process_video_frame)


    def setup_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)


        title = QLabel("🐾 ANIMAL DETECTION SYSTEM")
        title.setAlignment(Qt.AlignCenter)

        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)

        title.setFont(title_font)

        main_layout.addWidget(title)


        button_layout = QHBoxLayout()

        self.image_button = QPushButton("📷 Upload Image")
        self.video_button = QPushButton("🎥 Upload Video")
        self.save_button = QPushButton("💾 Save Result")
        self.clear_button = QPushButton("❌ Clear")

        for button in [
            self.image_button,
            self.video_button,
            self.save_button,
            self.clear_button
        ]:
            button.setMinimumHeight(45)

        button_layout.addWidget(self.image_button)
        button_layout.addWidget(self.video_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.clear_button)

        main_layout.addLayout(button_layout)


        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.StyledPanel)

        preview_layout = QVBoxLayout()
        preview_frame.setLayout(preview_layout)

        self.preview = QLabel("Upload an image or video")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(900, 500)

        self.preview.setStyleSheet("""
            QLabel {
                border: 2px solid #777;
                background-color: #eeeeee;
                color: #555555;
            }
        """)

        preview_layout.addWidget(self.preview)

        main_layout.addWidget(preview_frame)


        self.animal_label = QLabel("Animals Detected: 0")
        self.carnivore_label = QLabel("Carnivores: 0")

        info_font = QFont()
        info_font.setPointSize(16)
        info_font.setBold(True)

        self.animal_label.setFont(info_font)
        self.carnivore_label.setFont(info_font)

        main_layout.addWidget(self.animal_label)
        main_layout.addWidget(self.carnivore_label)


        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(self.status_label)



        self.image_button.clicked.connect(self.upload_image)
        self.video_button.clicked.connect(self.upload_video)
        self.save_button.clicked.connect(self.save_result)
        self.clear_button.clicked.connect(self.clear_screen)


    def upload_image(self):

        self.stop_video()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )

        if not file_path:
            return

        image = cv2.imread(file_path)

        if image is None:
            QMessageBox.critical(
                self,
                "Error",
                "Could not read the selected image."
            )
            return

        self.status_label.setText("Status: Detecting image...")

        result = model(image, conf=0.40, verbose=False)[0]

        processed_image, animal_count, carnivore_count = \
            self.draw_detections(image, result)

        self.current_image = processed_image

        self.display_image(processed_image)

        self.animal_label.setText(
            f"Animals Detected: {animal_count}"
        )

        self.carnivore_label.setText(
            f"Carnivores: {carnivore_count}"
        )

        self.status_label.setText(
            "Status: Image detection completed"
        )

        # Carnivore popup
        if carnivore_count > 0:

            QMessageBox.warning(
                self,
                "⚠ Carnivore Alert",
                f"{carnivore_count} carnivorous animal(s) detected!"
            )


    def upload_video(self):

        self.stop_video()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv)"
        )

        if not file_path:
            return

        self.cap = cv2.VideoCapture(file_path)

        if not self.cap.isOpened():

            QMessageBox.critical(
                self,
                "Error",
                "Could not open the selected video."
            )

            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        output_path = os.path.join(
            os.path.dirname(file_path),
            "detected_" + os.path.basename(file_path)
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        self.video_output = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        self.video_output_path = output_path

        self.status_label.setText(
            "Status: Processing video..."
        )

        interval = int(1000 / fps)

        self.timer.start(max(interval, 1))


    def process_video_frame(self):

        if self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:

            self.stop_video()

            QMessageBox.information(
                self,
                "Video Completed",
                f"Video detection completed.\n\n"
                f"Saved to:\n{self.video_output_path}"
            )

            return


        result = model(
            frame,
            conf=0.40,
            verbose=False
        )[0]


        processed_frame, animal_count, carnivore_count = \
            self.draw_detections(frame, result)


        self.current_image = processed_frame


        self.display_image(processed_frame)

        self.animal_label.setText(
            f"Animals Detected: {animal_count}"
        )

        self.carnivore_label.setText(
            f"Carnivores: {carnivore_count}"
        )


        if self.video_output is not None:

            self.video_output.write(
                processed_frame
            )

    def draw_detections(self, image, result):

        output = image.copy()

        animal_count = 0
        carnivore_count = 0


        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            animal = model.names[class_id]

            animal_count += 1



            if animal in CARNIVORES:

                color = (0, 0, 255)

                carnivore_count += 1

            else:

                color = (0, 255, 0)


            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                3
            )


            label = (
                f"{animal} "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                output,
                (x1, max(0, y1 - 30)),
                (
                    x1 + len(label) * 12,
                    y1
                ),
                color,
                -1
            )


            cv2.putText(
                output,
                label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )



        cv2.putText(
            output,
            f"Carnivores: {carnivore_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )


        return (
            output,
            animal_count,
            carnivore_count
        )


    def display_image(self, image):

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        height, width, channels = \
            rgb_image.shape

        bytes_per_line = channels * width

        q_image = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(q_image)

        pixmap = pixmap.scaled(
            self.preview.width(),
            self.preview.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.preview.setPixmap(pixmap)


    def save_result(self):

        if self.current_image is None:

            QMessageBox.information(
                self,
                "Nothing to Save",
                "Please detect an image or video first."
            )

            return


        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            "",
            "JPEG Image (*.jpg);;PNG Image (*.png)"
        )

        if file_path:

            cv2.imwrite(
                file_path,
                self.current_image
            )

            QMessageBox.information(
                self,
                "Saved",
                "Detection result saved successfully."
            )


    def stop_video(self):

        self.timer.stop()

        if self.cap is not None:

            self.cap.release()

            self.cap = None


        if self.video_output is not None:

            self.video_output.release()

            self.video_output = None


    def clear_screen(self):

        self.stop_video()

        self.preview.clear()

        self.preview.setText(
            "Upload an image or video"
        )

        self.animal_label.setText(
            "Animals Detected: 0"
        )

        self.carnivore_label.setText(
            "Carnivores: 0"
        )

        self.status_label.setText(
            "Status: Ready"
        )

        self.current_image = None


    def closeEvent(self, event):

        self.stop_video()

        event.accept()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = AnimalDetectionGUI()

    window.show()

    sys.exit(app.exec_())