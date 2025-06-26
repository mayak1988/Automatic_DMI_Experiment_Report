import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QSlider, QLabel,
    QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class DualSliceViewer(QWidget):
    def __init__(self, matrix1, matrix2):
        super().__init__()
        self.matrix1 = matrix1
        self.matrix2 = matrix2
        self.max1 = matrix1.shape[0] - 1
        self.max2 = matrix2.shape[0] - 1
        self.slice1_idx = 0
        self.slice2_idx = 0

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Dual 3D Matrix Viewer")

        # Sliders
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(self.max1)
        self.slider1.valueChanged.connect(self.update_matrix1)
        self.label1 = QLabel(f"Matrix 1 Slice: 0")

        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(self.max2)
        self.slider2.valueChanged.connect(self.update_matrix2)
        self.label2 = QLabel(f"Matrix 2 Slice: 0")

        # Plot setup
        self.fig, self.axes = plt.subplots(1, 2, figsize=(8, 4))
        self.canvas = FigureCanvas(self.fig)
        self.update_display()

        # Layout
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.label1)
        slider_layout.addWidget(self.slider1)
        slider_layout.addWidget(self.label2)
        slider_layout.addWidget(self.slider2)

        main_layout = QVBoxLayout()
        main_layout.addLayout(slider_layout)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)
        self.show()

    def update_matrix1(self, idx):
        self.slice1_idx = idx
        self.label1.setText(f"Matrix 1 Slice: {idx}")
        self.update_display()

    def update_matrix2(self, idx):
        self.slice2_idx = idx
        self.label2.setText(f"Matrix 2 Slice: {idx}")
        self.update_display()

    def update_display(self):
        self.axes[0].clear()
        self.axes[1].clear()

        self.axes[0].imshow(self.matrix1[self.slice1_idx], cmap='gray')
        self.axes[0].set_title("Matrix 1")
        self.axes[1].imshow(self.matrix2[self.slice2_idx], cmap='gray')
        self.axes[1].set_title("Matrix 2")

        for ax in self.axes:
            ax.axis('off')

        self.canvas.draw()

# --- Run GUI ---
if __name__ == '__main__':
    # Example matrices with different slice counts
    matrix1 = np.random.rand(30, 128, 128)
    matrix2 = np.random.rand(45, 128, 128)

    app = QApplication(sys.argv)
    viewer = DualSliceViewer(matrix1, matrix2)
    sys.exit(app.exec_())
