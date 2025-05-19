import sys
import cv2
import pandas as pd
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import uic

class InfoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV 실시간 영상 + 학생 정보")
        self.setGeometry(100, 100, 900, 500)

        # 전체 레이아웃
        main_layout = QHBoxLayout()

        # 왼쪽: OpenCV 영상 보여줄 QLabel
        self.video_label = QLabel()
        self.video_label.setFixedSize(600, 450)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label)

        # 오른쪽: 학생 정보 표시
        student_info = self.load_excel_info()
        self.info_label = QLabel(student_info)
        self.info_label.setStyleSheet("background-color: #DFF6FF; padding: 15px; font-size: 14px;")
        self.info_label.setAlignment(Qt.AlignTop)
        main_layout.addWidget(self.info_label)

        self.setLayout(main_layout)

        # OpenCV 타이머로 프레임 업데이트
        self.cap = cv2.VideoCapture(0)  # 0: 기본 웹캠
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 간격 (약 30fps)

    def load_excel_info(self):
        try:
            df = pd.read_excel("data/student_data.xlsx")
            student = df.iloc[0]
            return f"""
            <b>INFORMATION</b><br><br>
            과목: {student['과목']}<br>
            날짜: {student['날짜']}<br>
            이름: {student['이름']}<br>
            학번: {student['학번']}<br>
            """
        except Exception as e:
            return f"엑셀 파일을 불러올 수 없습니다: {e}"

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # OpenCV 이미지 → Qt 이미지로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InfoApp()
    window.show()
    sys.exit(app.exec_())
