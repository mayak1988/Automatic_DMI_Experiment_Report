import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QPushButton, QFileDialog, QSlider, QScrollArea, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal ,QPoint,QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QPolygon, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import re
from reportlab.pdfbase.pdfmetrics import stringWidth
from segment_anything import sam_model_registry, SamPredictor
from skimage import measure
from PIL import Image
import matplotlib.cm as cm



class DualSliceViewer(QWidget):
    """
    A PyQt widget to view and scroll through slices of two 3D image matrices side-by-side.

    This viewer currently displays one matrix (`matrix1`) with a slider to scroll through slices.
    The matrix is preprocessed by rotating and flipping for consistent orientation.

    Attributes:
    -----------
    matrix1 : np.ndarray
        First 3D image volume to visualize (e.g., MRI).
    matrix2 : np.ndarray
        Second 3D image volume (e.g., DMI), currently loaded but not displayed.
    max1 : int
        Maximum slice index for matrix1.
    max2 : int
        Maximum slice index for matrix2.
    slice1_idx : int
        Current slice index for matrix1 displayed.
    slice2_idx : int
        Current slice index for matrix2 (not currently displayed).

    UI Elements:
    ------------
    slider1 : QSlider
        Horizontal slider to scroll through slices of matrix1.
    label1 : QLabel
        Displays current slice number for matrix1.
    canvas : FigureCanvas
        Matplotlib canvas showing the current slice image.
    axes : matplotlib.axes.Axes
        Matplotlib axes used for displaying the image.

    Methods:
    --------
    __init__(matrix1, matrix2)
        Initializes the viewer, processes matrix1 orientation, and sets up the UI.
    initUI()
        Sets up PyQt widgets and layout, connects slider to update function.
    update_matrix1(idx)
        Slot connected to slider; updates the current slice index and label, then refreshes display.
    update_display()
        Clears the Matplotlib axes and redraws the current slice of matrix1.
    """
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
        """
        Update the displayed slice index and refresh the image display.

        Parameters:
        -----------
        idx : int
            The new slice index selected by the slider.
        """
        self.slice1_idx = idx
        self.label1.setText(f"MRI Slice: {idx}")
        self.update_display()



    def update_display(self):
        """Clear the axes and display the current slice of matrix1."""
        self.axes.clear()
        self.axes.imshow(self.matrix1[:, :, self.slice1_idx], cmap='gray', aspect='equal')
        self.axes.set_title("MRI")
        self.axes.axis('off')
        self.canvas.draw()

