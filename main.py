from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QMovie, QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidgetItem
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtMultimedia import *
import pandas as pd
import sys
import cv2
from TTS import speak
import threading

form_class = uic.loadUiType("AttendanceCheck.ui")[0]

class GUI(QMainWindow, form_class):
    def __init__(self):
        super().__init__()

        # 타이머 구현
        #정각 표시
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        #메뉴 시간 표시
        self.clock_menu_timer = QTimer()
        self.clock_menu_timer.timeout.connect(self.update_clock_timer)

        # 사운드 효과
        self.player = QMediaPlayer()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile("/home/msi/py_project/10second_alarm.mp3")))
        self.player.setVolume(50)

        #데이터 불러오기
        self.df = self.load_csv("data/data_file.csv")
        self.csv_loaded = False
        # GUI 구현
        self.setupUi(self)

        self.movie = QMovie("image/cbnu_character.gif")
        self.uangLabel.setMovie(self.movie)
        self.movie.start()

        cbnu_logo = QPixmap("image/cbnu_logo.png")
        self.cbnuLabel.setPixmap(cbnu_logo)
        self.cbnuLabel.setScaledContents(True)

        self.clock_menu_timer.start(1000)
        self.update_clock_timer()
        # 첫 시작화면 페이지 지정
        self.stackedWidget.setCurrentIndex(0)
        # Button 구현
        self.startButton.clicked.connect(self.button_clicked)
        self.prevButton.clicked.connect(self.prevButton_clicked)

    def start_Timer(self):
        self.clock_timer.start(1000)
        self.update_clock()

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
            df.to_csv("data/data_file.csv", index=False, encoding='utf-8-sig')
            self.df = df
            print("출석 완료 및 저장됨.")

        except Exception as e:
            print("출석 처리 중 오류:", e)

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


    def update_clock_timer(self):
        currentTime = QDateTime.currentDateTime()
        formatted_time = currentTime.toString("yyyy-MM-dd hh:mm:ss")
        self.timerLabel.setText(formatted_time)
        self.timerLabel.setStyleSheet("color: black;")

    def button_clicked(self):
        print("Button clicked")
        self.stackedWidget.setCurrentIndex(1)

        # videoLabel 테투리 색깔 지정
        #self.videoLabel.setStyleSheet("border: 4px solid black;")
        self.start_Timer()
        self.call_names_with_tts()
        # self.player.play()
    def prevButton_clicked(self):
        self.stackedWidget.setCurrentIndex(0)

    def call_names_with_tts(self):
        thread = threading.Thread(target=self.call_names_from_csv)
        thread.start()

    def call_names_from_csv(self):
        try:
            if self.df is None:
                print("CSV 데이터 없음")
                return

            # 이름 컬럼에서 리스트 추출
            name_list = self.df['이름'].dropna().tolist()

            # 호출 시작 멘트
            speak("지금부터 출석을 부르겠습니다.", volume_gain=8, wait=2)

            for name in name_list:
                print("TTS:", name)

                student_row = self.df[self.df['이름'] == name]
                if student_row.empty:
                    continue

                self.nameLabel.setText(name)
                self.IDLabel.setText(str(student_row['학번'].values[0]))
                self.departmentLabel.setText(str(student_row['학과'].values[0]))
                speak(name, volume_gain=8, speed=1.1, wait=1.5)

        except Exception as e:
            print("이름 호출 중 오류:", e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    app.exec_()

