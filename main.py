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

from breezy.switch import switch

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
        self.check_img = None
        self.tts_running = False
        self.waiting_for_result = False
        self.result_received = False
        self.recognition_start_time = None
        self.start_attendance_system = False
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)

        # 메인메뉴 타이머 구현
        # self.clock_menu_timer = QTimer()
        # self.clock_menu_timer.timeout.connect(self.update_mainmenu_clock)
        # self.clock_menu_timer.start(1000)
        # self.update_mainmenu_clock()

        self.effect = QSoundEffect()
        self.effect.setSource(QUrl.fromLocalFile("/home/a/python_project/Automatic-Attendance-System/data/10second_alarm.wav"))
        self.effect.setVolume(0.9)

        self.df = None
        self.current_lecture_hour = None
        self.current_lecture_name = None
        self.csv_loaded = False
        self.alarm_played = False
        self.attendance_done = True
        self.setupUi(self)

        self.movie = QMovie("image/cbnu_character.gif")
        self.uangLabel.setMovie(self.movie)
        self.uangLabel.setScaledContents(False)
        self.movie.start()

        cbnu_logo = QPixmap("image/cbnu_logo.png")
        self.default_img = QPixmap("image/default.jpg")
        self.cbnuLabel.setPixmap(cbnu_logo)
        self.cbnuLabel.setScaledContents(True)
        self.videoLabel.setStyleSheet("border: 10px solid black")
        self.captureLabel.setStyleSheet("border: 10px solid black")
        self.captureLabel.setPixmap(self.default_img)
        self.captureLabel.setScaledContents(True)
        # self.UserManualTitleLabel.setStyleSheet("border: 4px solid black")
        # self.UserManualText.setStyleSheet("border: 4px solid black")

        self.stackedWidget.setCurrentIndex(0)

        self.startButton.clicked.connect(self.button_clicked)
        self.prevButton.clicked.connect(self.prevButton_clicked)
        self.UserManualButton.clicked.connect(self.UserManualbutton_clicked)
        self.manualPrevButton.clicked.connect(self.manualPrevButton_clicked)
        self.sim_list = []
        self.day_map = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}

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

            df.to_csv("data/data_file.csv", index=False, encoding='utf-8-sig')
            self.df = df
            print(f"{status} 완료 및 저장됨: {target_name}")

        except Exception as e:
            print("출석 처리 중 오류:", e)

    def load_today_schedule(self, day_str, file_path="timetable.txt"):
        try:
            schedule = {}  # 예: {15: "자동제어", 17: "자료구조"}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        day = parts[0].strip()
                        hour = int(parts[1].strip())
                        lecture = parts[2].strip()
                        if day == day_str:
                            schedule[hour] = lecture
            return schedule
        except Exception as e:
            print("시간표 파일 읽기 오류:", e)
        return {}

    def get_next_class_hour(self, current_hour, today_hours):
        future_hours = [h for h in today_hours if h > current_hour]
        return min(future_hours) if future_hours else None
    # 타이머 시간 도달시 빨간색 색깔 변하게 하는 코드
    # 정각 도달하는 시간 소리 표시
    def update_clock(self):
        currentTime = QDateTime.currentDateTime()

        second = currentTime.time().second()
        minute = currentTime.time().minute()
        hour = currentTime.time().hour()

        day = self.day_map[currentTime.date().dayOfWeek()]
        formatted_time = currentTime.toString("yyyy-MM-dd hh:mm:ss") + " " + day +"요일"
        self.clockLabel.setText(formatted_time)

        time_schedule = self.load_today_schedule(day, "/home/a/python_project/Automatic-Attendance-System/data/timetable.txt")
        if hour in time_schedule:
            attendance_check = "Y" if self.attendance_done else "N"
            self.currentClassLabel.setText(f"수업 : {time_schedule[hour]} | 출석체크 여부 : {attendance_check}")
        else:
            self.currentClassLabel.setText("수업 : 없음")
        if time_schedule is not None:
            today_hours = sorted(time_schedule.keys())
            next_hour = self.get_next_class_hour(hour, today_hours)

            if next_hour is not None:
                next_class = time_schedule[next_hour]
                print(f"🕐 다음 수업은 오늘 {next_hour}, {next_class}입니다.")
                print(f"출석 체크 여부 : {self.attendance_done}")

                # 👉 이전에 로드한 적 없는 시간이면 로드
                if self.attendance_done and next_hour != self.current_lecture_hour:
                    self.current_lecture_hour = next_hour
                    self.current_lecture_name = next_class
                    self.df = self.load_csv(f"data/{self.current_lecture_name}.csv")
                    self.attendance_done = False
                    print(f"📂 [{self.current_lecture_name}] 수업 출석부 로드 완료.")
            else:
                print("📭 오늘 더 이상 수업 없음")
        else:
            print(f"{day}요일에 해당하는 수업이 없습니다.")
        if  hour == (next_hour-1) and minute == 41 and second >50 and not self.alarm_played:
            print(f"[{minute}:{second}] 🔊 Alarm starting")
            self.effect.stop()
            self.effect.play()
            self.alarm_played = True

        if  hour == (next_hour-1) and minute == 42 and second == 1 and self.alarm_played:
            print(f"[{minute}:{second}] alarm stopping, start attendance_system")
            self.alarm_played = False
            self.start_attendance_system = True

        if minute == 59 and 50 <= second <= 59:
            progress = (second - 50) / 9.0
            red_intensity = int(255 * progress)
            color_style = f"color: rgb({red_intensity}, 0, 0);"
            self.clockLabel.setStyleSheet(color_style)
        else:
            self.clockLabel.setStyleSheet("color: black;")

        if (self.start_attendance_system and not self.attendance_done):
            self.call_names_with_tts()
            self.start_attendance_system = False

    # 메인메뉴 타이머 구현
    # def update_mainmenu_clock(self):
    #     currentTime = QDateTime.currentDateTime()
    #     formatted_time = currentTime.toString("yyyy-MM-dd hh:mm:ss")
    #     self.timerLabel.setText(formatted_time)
    #     self.timerLabel.setStyleSheet("color: black;")

    def process_camera_frame(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            # 실제 적용된 해상도 출력
            print("WIDTH:", self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            print("HEIGHT:", self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = self.cap.read()
        if not ret:
            return

        # 출석 인식이 필요한 경우에만 얼굴 비교
        if self.waiting_for_result:
            result, sim, bbox = self.attendance_system.recognize_and_check_attendance(frame, self.check_face)
            elapsed_time = time.time() - self.recognition_start_time
            #### 이 부분 디버거 모드 선택할 수 있게 해서 껐다 킬 수 있게 하기
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 40), 10)
            if sim is not None:
                self.sim_list.append(sim)
        # 실시간 프레임 항상 표시
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        # print(w, h, ch, bytes_per_line)
        qimg = QImage(frame_rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        self.videoLabel.setPixmap(QPixmap.fromImage(qimg))
        self.videoLabel.setScaledContents(True)


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
                self.attendanceLabel.setText("")
                self.videoLabel.setStyleSheet("border: 10px solid black")
                self.nameLabel.setText(name)
                student_row = self.df[self.df['이름'] == name]
                self.IDLabel.setText(str(student_row['학번'].values[0]))
                self.departmentLabel.setText(str(student_row['학과'].values[0]))
                self.current_name = name
                self.check_img = cv2.imread(f"faces/{name}.jpg")
                self.check_face = self.attendance_system.load_check_face(f"faces/{name}.jpg")
                if self.check_img is not None:
                    tmp = cv2.cvtColor(self.check_img, cv2.COLOR_BGR2RGB)
                    target_w = self.captureLabel.width()
                    target_h = self.captureLabel.height()

                    resized = cv2.resize(self.check_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                    ch, cw, cc = rgb.shape
                    qimg_crop = QImage(rgb.data, cw, ch, cc * cw, QImage.Format_RGB888)
                    self.captureLabel.setPixmap(QPixmap.fromImage(qimg_crop))
                    self.captureLabel.setScaledContents(True)
                self.result_received = False
                self.waiting_for_result = False  # 아직 인식 시작 안 함

                # TTS 먼저 출력
                tts_thread = threading.Thread(
                    target=speak,
                    args=(name,),
                    kwargs={
                        "volume_gain": 8,
                        "speed": 1.1,
                        "wait": 1.5,
                        "stop_flag": lambda: self.tts_running
                    }
                )
                tts_thread.start()
                tts_thread.join()  # TTS 재생 끝날 때까지 기다림

                # 3초 인식 시작
                self.recognition_start_time = time.time()
                self.result_received = False
                self.waiting_for_result = True

                self.sim_list = []  # 유사도 저장 리스트 초기화

                self.recognition_start_time = time.time()
                self.result_received = False
                self.waiting_for_result = True

                while self.waiting_for_result:
                    elapsed = time.time() - self.recognition_start_time
                    if elapsed >= 3.0:
                        self.waiting_for_result = False
                        break
                    QtCore.QThread.msleep(100)
                    QApplication.processEvents()

                # 평균 유사도 판단
                if self.sim_list:
                    avg_sim = sum(self.sim_list) / len(self.sim_list)
                    print(f"📊 평균 유사도: {avg_sim:.2f}")
                    if avg_sim >= 0.3:
                        print(f"✅ 출석: {self.current_name}, 평균 유사도={avg_sim:.2f} (기준: >= 0.3)")
                        self.attendanceLabel.setText("출석")
                        self.videoLabel.setStyleSheet("border: 10px solid #39FF14")
                        self.display_student_info_from_df(self.df, self.current_name, status="출석")

                        speak(f"출석 확인", volume_gain=6, wait=1.0)
                    else:
                        self.attendanceLabel.setText("결석")
                        self.videoLabel.setStyleSheet("border: 10px solid red")
                        print(f"❌ 평균 유사도 낮음: {avg_sim:.2f} (기준: >= 0.3)")
                        self.display_student_info_from_df(self.df, self.current_name, status="결석")
                        speak(f"결석 처리", volume_gain=6, wait=1.0)
                else:
                    self.attendanceLabel.setText("결석")
                    self.videoLabel.setStyleSheet("border: 10px solid red")
                    print(f"⚠️ 인식 실패 - 유사도 없음: {self.current_name}")
                    self.display_student_info_from_df(self.df, self.current_name, status="결석")
                    speak(f"결석 처리", volume_gain=6, wait=1.0)
                    QtCore.QThread.msleep(100)
                    QApplication.processEvents()
            self.attendance_done = True
            self.videoLabel.setStyleSheet("border: 10px solid black")
            speak("출석이 완료되었습니다.", volume_gain=8, wait=2, stop_flag=lambda: self.tts_running)
            self.attendanceLabel.setText("")
            self.nameLabel.setText("")
            self.IDLabel.setText("")
            self.departmentLabel.setText("")
            self.captureLabel.setPixmap(self.default_img)
        except Exception as e:
            print("이름 호출 중 오류:", e)


    def button_clicked(self):
        print("Button clicked")
        self.stackedWidget.setCurrentIndex(1)
        self.start_Timer()
        self.attendance_timer.start(100)
        threading.Thread(
            target=speak,
            args=("지금부터 출석을 부르겠습니다.",),
            kwargs={
                "volume_gain": 8,
                "wait": 2,
                "stop_flag": lambda: self.tts_running
            },
            daemon=True
        ).start()


    def prevButton_clicked(self):
        self.tts_running = False
        self.attendance_timer.stop()
        self.clock_timer.stop()
        self.stackedWidget.setCurrentIndex(0)
        self.nameLabel.setText("")
        self.IDLabel.setText("")
        self.departmentLabel.setText("")

    def UserManualbutton_clicked(self):
        self.stackedWidget.setCurrentIndex(2)

    def manualPrevButton_clicked(self):
        self.stackedWidget.setCurrentIndex(0)
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
