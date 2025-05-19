from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QMovie, QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidgetItem
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtMultimedia import *
import pandas as pd
import sys
import cv2

form_class = uic.loadUiType("AttendanceCheck.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()

        # OpenCV
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 타이머 구현
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)

        # 사운드 효과
        self.player = QMediaPlayer()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile("/home/msi/py_project/10second_alarm.mp3")))
        self.player.setVolume(50)

        # self.sound = QSoundEffect()
        # self.sound.setSource(QUrl.fromLocalFile("/home/msi/py_project/click.wav"))
        # #print("QUrl:", QUrl.fromLocalFile("click.wav").toString())
        # self.sound.setVolume(0.5)

        #데이터 불러오기
        self.df = self.load_csv("/home/msi/py_project/data/data_file.csv")
        self.csv_loaded = False
        # GUI 구현
        self.setupUi(self)

        self.movie = QMovie("image/cbnu_character.gif")
        self.uangLabel.setMovie(self.movie)
        self.movie.start()

        cbnu_logo = QPixmap("image/cbnu_logo.png")
        self.cbnuLabel.setPixmap(cbnu_logo)
        self.cbnuLabel.setScaledContents(True)

        # 첫 시작화면 페이지 지정
        self.stackedWidget.setCurrentIndex(0)
        # Button 구현
        self.startButton.clicked.connect(self.button_clicked)
        self.PrevButton.clicked.connect(self.PrevButton_clicked)
    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
    def start_Timer(self):
        self.clock_timer.start(1000)
        self.update_clock()

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.videoLabel.clear()
    def stop_Timer(self):
        self.clock_timer.stop()

    def load_csv(self, file_path):
        try:
            # CSV 파일 전체 불러오기
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print("CSV 불러오기 실패:", e)
            return None

    def display_student_info_from_df(self, df, target_name="유태현"):
        try:
            currentTime = QDateTime.currentDateTime()
            today = currentTime.toString("yyyy-MM-dd")

            # 학생 필터링
            target_idx = df[df['이름'] == target_name].index
            if target_idx.empty:
                print(f"{target_name} 정보를 찾을 수 없습니다.")
                return

            # 날짜 열 없으면 추가
            if today not in df.columns:
                df[today] = ""

            # 출석 기록
            df.at[target_idx[0], today] = "출석"

            # 화면 표시용 항목/값
            headers = ['이름', '학번', '학과', today]
            row_data = [
                df.at[target_idx[0], '이름'],
                df.at[target_idx[0], '학번'],
                df.at[target_idx[0], '학과'],
                df.at[target_idx[0], today]
            ]

            # QTableWidget에 세로로 출력
            self.InformationTable.setRowCount(len(headers))
            self.InformationTable.setColumnCount(2)
            self.InformationTable.setHorizontalHeaderLabels(["항목", "값"])

            for row in range(len(headers)):
                self.InformationTable.setItem(row, 0, QTableWidgetItem(headers[row]))
                self.InformationTable.setItem(row, 1, QTableWidgetItem(str(row_data[row])))

            # CSV 저장
            df.to_csv("/home/msi/py_project/data/data_file.csv", index=False, encoding='utf-8-sig')
            self.df = df
            print("출석 완료 및 저장됨.")

        except Exception as e:
            print("출석 처리 중 오류:", e)
    def update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        if len(faces) == 1:
            x, y, w, h = faces[0]

            face_roi = frame[y:y+h, x:x+w]
            h1, w1, ch1 = face_roi.shape
            bytesPerLine1 = ch1 * w1
            qimg1 = QImage(face_roi.tobytes(), w1, h1, bytesPerLine1, QImage.Format_RGB888)
            pixmap1 = QPixmap.fromImage(qimg1)
            self.roilabel.setPixmap(pixmap1)
            self.roilabel.setScaledContents(True)

            cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
            if not self.csv_loaded:
                if self.df is not None:
                    self.display_student_info_from_df(self.df,target_name="유태현")
                    self.csv_loaded = True
        # else:
            # self.roilabel.clear()
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        qimg = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.videoLabel.setPixmap(pixmap)

        if (len(faces) > 0):
            self.videoLabel.setStyleSheet("border: 4px solid red")
        else:
            self.videoLabel.setStyleSheet("border: 4px solid black")

    def update_clock(self):
        currentTime = QDateTime.currentDateTime()
        formatted_time = currentTime.toString("yyyy-MM-dd hh:mm:ss")
        self.clockLabel.setText(formatted_time)

        second = currentTime.time().second()
        minute = currentTime.time().minute()
        if minute == 0 and second >=51 and second <=59:
            self.player.play()
        if minute ==0 and second >= 50 and second <=59:
            progress = (second - 50) / 9.0
            red_intensity = int(255 * progress)
            color_style = f"color: rgb({red_intensity}, 0, 0);"
            self.clockLabel.setStyleSheet(color_style)

        else:
            self.clockLabel.setStyleSheet("color: black;")
    def button_clicked(self):
        print("Button clicked")
        self.stackedWidget.setCurrentIndex(1)
        self.start_camera()
        # videoLabel 테투리 색깔 지정
        self.videoLabel.setStyleSheet("border: 4px solid black;")
        self.start_Timer()
        # self.player.play()
    def PrevButton_clicked(self):
        self.stackedWidget.setCurrentIndex(0)
        self.stop_camera()
        self.stop_Timer()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()

