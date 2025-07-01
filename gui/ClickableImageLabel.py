
from PyQt5.QtWidgets import  QLabel
from PyQt5.QtCore import Qt, pyqtSignal




class ClickableImageLabel(QLabel):
    """
    A QLabel subclass that emits a signal with mouse click coordinates and button.

    Signals:
    --------
    clicked(int, int, Qt.MouseButton)
        Emitted when the label is clicked with the x, y coordinates of the click
        relative to the label and the mouse button pressed (left or right).
    """

    clicked = pyqtSignal(int, int, Qt.MouseButton)

    def mousePressEvent(self, event):
        if event.button() in [Qt.LeftButton, Qt.RightButton]:
            x = event.pos().x()
            y = event.pos().y()
            self.clicked.emit(x, y, event.button())
