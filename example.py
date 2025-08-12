from PyQt5 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)

window = QtWidgets.QWidget()
window.setWindowTitle('PyQt5 example')
window.setGeometry(100, 100, 280, 80)
window.show()

sys.exit(app.exec_())
