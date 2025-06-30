from gui.viewer import MRICSIViewer
from PyQt5.QtWidgets import QApplication
import sys


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet("""
            QWidget {
                background-color: #f5f7fa;
                font-family: 'Segoe UI';
                font-size: 12pt;
                color: #333;
            }

            QLabel {
                font-weight: bold;
            }

            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }

            QPushButton:hover {
                background-color: #357ABD;
            }

            QPushButton:pressed {
                background-color: #2C5F9E;
            }

            QLineEdit, QTextEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
                background-color: #fff;
            }

            QComboBox {
                padding: 5px;
                border-radius: 4px;
                background-color: #fff;
                border: 1px solid #ccc;
            }

            QGroupBox {
                border: 1px solid #d0d0d0;
                margin-top: 10px;
                padding: 10px;
                border-radius: 8px;
            }

            QGroupBox:title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                font-weight: bold;
                color: #4A4A4A;
            }
        """)
    viewer = MRICSIViewer()
    viewer.show()
    sys.exit(app.exec_())
