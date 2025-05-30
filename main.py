# 유사도 판단은 recognize_and_check_attendance()에서 이루어지며,
# 일정 기준 (예: 0.3 이상) 유사도가 넘으면 출석으로 인정됩니다.
# 해당 유사도는 sim 변수에 저장되며 process_camera_frame() 내에서 비교합니다.

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QMovie, QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidgetItem
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtMultimedia import *
import pandas as pd
import sys
import cv2
import threading
import time
from TTS import speak
from face_attendance import FaceAttendanceSystem

import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

form_class = uic.loadUiType("AttendanceCheck.ui")[0]

class GUI(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.attendance_system = FaceAttendanceSystem()
        self.attendance_timer = QTimer()
        self.attendance_timer.timeout.connect(self.process_camera_frame)

        self.cap = None
        self.current_name = None
        self.check_face = None
        self.tts_running = False
        self.waiting_for_result = False
        self.result_received = False
        self.recognition_start_time = None

        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)

        self.clock_menu_timer = QTimer()
        #self.clock_menu_timer.timeout.connect(self.update_mainmenu_clock)

        self.player = QMediaPlayer()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile("/home/a/python_project/Automatic-Attendance-System/10second_alarm.mp3")))
        self.player.setVolume(50)

        self.df = self.load_csv("data/data_file.csv")
        self.csv_loaded = False

        self.setupUi(self)

        self.movie = QMovie("image/cbnu_character.gif")
        self.uangLabel.setMovie(self.movie)
        self.movie.start()

        cbnu_logo = QPixmap("image/cbnu_logo.png")
        self.cbnuLabel.setPixmap(cbnu_logo)
        self.cbnuLabel.setScaledContents(True)

        self.clock_menu_timer.start(1000)
        self.update_mainmenu_clock()
        self.stackedWidget.setCurrentIndex(0)

        self.startButton.clicked.connect(self.button_clicked)
        self.prevButton.clicked.connect(self.prevButton_clicked)

    def start_Timer(self):
        self.clock_timer.start(1000)
        self.update_clock()

    def stop_Timer(self):
        self.clock_timer.stop()

    def load_csv(self, file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print("CSV 불러오기 실패:", e)
            return None

    def display_student_info_from_df(self, df, target_name, status="출석"):
        try:
            currentTime = QDateTime.currentDateTime()
            today = currentTime.toString("yyyy-MM-dd")

            target_idx = df[df['이름'] == target_name].index
            if target_idx.empty:
                print(f"{target_name} 정보를 찾을 수 없습니다.")
                return

            if today not in df.columns:
                df[today] = ""

            df.at[target_idx[0], today] = status

            headers = ['이름', '학번', '학과', today]
            row_data = [
                df.at[target_idx[0], '이름'],
                df.at[target_idx[0], '학번'],
                df.at[target_idx[0], '학과'],
                df.at[target_idx[0], today]
            ]

            self.InformationTable.setRowCount(len(headers))
            self.InformationTable.setColumnCount(2)
            self.InformationTable.setHorizontalHeaderLabels(["항목", "값"])

            for row in range(len(headers)):
                self.InformationTable.setItem(row, 0, QTableWidgetItem(headers[row]))
                self.InformationTable.setItem(row, 1, QTableWidgetItem(str(row_data[row])))

            df.to_csv("data/data_file.csv", index=False, encoding='utf-8-sig')
            self.df = df
            print(f"{status} 완료 및 저장됨: {target_name}")

        except Exception as e:
            print("출석 처리 중 오류:", e)

    def update_clock(self):
        currentTime = QDateTime.currentDateTime()
        formatted_time = currentTime.toString("yyyy-MM-dd hh:mm:ss")
        self.clockLabel.setText(formatted_time)
        second = currentTime.time().second()
        minute = currentTime.time().minute()
        if minute == 0 and 51 <= second <= 59:
            self.player.play()
        if minute == 0 and 50 <= second <= 59:
            progress = (second - 50) / 9.0
            red_intensity = int(255 * progress)
            color_style = f"color: rgb({red_intensity}, 0, 0);"
            self.clockLabel.setStyleSheet(color_style)
        else:
            self.clockLabel.setStyleSheet("color: black;")

    def update_mainmenu_clock(self):
        currentTime = QDateTime.currentDateTime()
        formatted_time = currentTime.toString("yyyy-MM-dd hh:mm:ss")
        #self.timerLabel.setText(formatted_time)
        self.timerLabel.setStyleSheet("color: black;")

    def process_camera_frame(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)

        ret, frame = self.cap.read()
        if not ret or not self.waiting_for_result:
            return

        # ▶ 전체 프레임 실시간 영상 표시
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        self.videoLabel.setPixmap(QPixmap.fromImage(qimg))
        self.videoLabel.setScaledContents(True)
        # ▶ 출석 판단 및 얼굴 crop
        result, sim, face_crop = self.attendance_system.recognize_and_check_attendance(frame, self.check_face)
        elapsed_time = time.time() - self.recognition_start_time

        if face_crop is not None:
            crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            ch, cw, cc = crop_rgb.shape
            qimg_crop = QImage(crop_rgb.data, cw, ch, cc * cw, QImage.Format_RGB888)
            self.captureLabel.setPixmap(QPixmap.fromImage(qimg_crop))

        if result is True:
            print(f"✅ 출석: {self.current_name}, 유사도={sim:.2f} (기준: >= 0.3)")
            self.display_student_info_from_df(self.df, self.current_name, status="출석")
            self.result_received = True
            self.waiting_for_result = False
            #self.attendance_timer.stop()

        elif result is False:
            print(f"❌ 유사도 낮음: {sim:.2f} (기준: >= 0.3)")

        if elapsed_time > 3.0 and not self.result_received:
            print(f"⚠️ 인식 실패 - 결석 처리: {self.current_name}")
            self.display_student_info_from_df(self.df, self.current_name, status="결석")
            self.waiting_for_result = False
            #self.attendance_timer.stop()

    def call_names_with_tts(self):
        self.tts_running = True
        thread = threading.Thread(target=self.call_names_from_csv, daemon=True)
        thread.start()

    def call_names_from_csv(self):
        try:
            if self.df is None:
                print("CSV 데이터 없음")
                return

            name_list = self.df['이름'].dropna().tolist()
            speak("지금부터 출석을 부르겠습니다.", volume_gain=8, wait=2, stop_flag=lambda: self.tts_running)

            for name in name_list:
                if not self.tts_running:
                    break

                print("TTS:", name)
                self.nameLabel.setText(name)
                student_row = self.df[self.df['이름'] == name]
                self.IDLabel.setText(str(student_row['학번'].values[0]))
                self.departmentLabel.setText(str(student_row['학과'].values[0]))
                self.current_name = name
                self.check_face = self.attendance_system.load_check_face(f"faces/{name}.jpg")

                speak(name, volume_gain=8, speed=1.1, wait=1.5, stop_flag=lambda: self.tts_running)

                self.recognition_start_time = time.time()
                self.result_received = False
                self.waiting_for_result = True


                while self.waiting_for_result:
                    QtCore.QThread.msleep(100)
                    QApplication.processEvents()

        except Exception as e:
            print("이름 호출 중 오류:", e)

    def button_clicked(self):
        print("Button clicked")
        self.stackedWidget.setCurrentIndex(1)
        self.start_Timer()
        self.call_names_with_tts()
        self.attendance_timer.start(100)

    def prevButton_clicked(self):
        self.tts_running = False
        self.attendance_timer.stop()
        self.stackedWidget.setCurrentIndex(0)
        self.nameLabel.setText("")
        self.IDLabel.setText("")
        self.departmentLabel.setText("")

    def closeEvent(self, event):
        self.tts_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    app.exec_()
