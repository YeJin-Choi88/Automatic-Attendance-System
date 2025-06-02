# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QSizePolicy, QVBoxLayout
#
# app = QApplication([])
# window = QWidget()
#
# btn = QPushButton("확장되는 버튼")
# btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#
# layout = QVBoxLayout()
# layout.addWidget(btn)
#
# window.setLayout(layout)
# window.resize(400, 300)
# window.show()
#
# app.exec_()

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget

app = QApplication([])
window = uic.loadUi("test.ui")
window.show()
app.exec_()