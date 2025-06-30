from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal ,QPoint,QEvent
from PyQt5.QtGui import QIcon

class ClickableImageLabel(QLabel):
    # clicked = pyqtSignal(int, int)  # emits x,y when clicked

    # def mousePressEvent(self, event):
    #     if event.button() == Qt.LeftButton:
    #         self.clicked.emit(event.pos().x(), event.pos().y())

    clicked = pyqtSignal(int, int, Qt.MouseButton)

    def mousePressEvent(self, event):
        if event.button() in [Qt.LeftButton, Qt.RightButton]:
            x = event.pos().x()
            y = event.pos().y()
            self.clicked.emit(x, y, event.button())



class DualSliceViewer(QWidget):
    def __init__(self, matrix1, matrix2):
        super().__init__()
        self.matrix1 = matrix1
        self.matrix1 = np.flipud(np.rot90(self.matrix1, axes=(0, 1)))
        self.matrix2 = matrix2
        self.max1 = matrix1.shape[0] - 1
        self.max2 = matrix2.shape[0] - 1
        self.slice1_idx = 0
        self.slice2_idx = 0

        self.initUI()

    def initUI(self):
        
        self.setWindowTitle("🧠 MRI + DMI Experiment Viewer")
        self.setWindowIcon(QIcon("icon.png"))  # optional icon

        self.max1 = self.matrix1.shape[2] - 1  # axis 2 is the slice axis
        # --- Slider ---
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(self.max1)
        self.slider1.valueChanged.connect(self.update_matrix1)

        self.label1 = QLabel(f"MRI Slice: 0")

        # --- Matplotlib Figure ---
        
        self.fig = Figure(figsize=(5, 5))
        self.axes = self.fig.add_subplot(111)
        self.axes.set_aspect('equal')
        self.axes.axis('off')
        self.axes.set_position([0, 0, 1, 1])  # fill canvas

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFixedSize(500, 500)  # force square display

        layout = QVBoxLayout()
        layout.addWidget(self.label1)
        layout.addWidget(self.slider1)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.update_display()

        # self.show()  # You usually call this from outside the class, not inside initUI

    def update_matrix1(self, idx):
        self.slice1_idx = idx
        self.label1.setText(f"MRI Slice: {idx}")
        self.update_display()

    def update_display(self):
        self.axes.clear()
        self.axes.imshow(self.matrix1[:, :, self.slice1_idx], cmap='gray', aspect='equal')
        self.axes.set_title("MRI")
        self.axes.axis('off')
        self.canvas.draw()