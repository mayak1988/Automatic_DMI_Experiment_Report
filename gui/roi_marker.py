import sys, io,os,re,scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QPushButton, QFileDialog, QSlider, QScrollArea, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal ,QPoint,QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QPolygon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import re
from reportlab.pdfbase.pdfmetrics import stringWidth
import cv2
from segment_anything import sam_model_registry, SamPredictor
from skimage import measure
from PIL import Image
import matplotlib.cm as cm

from gui.widgets import DualSliceViewer,ClickableImageLabel
from gui.image_utils import rotate_and_flip_image

class ROIMarker(QWidget):
    finished = pyqtSignal()  # Signal to notify when "Finish" is clicked

    def __init__(self, slice_index, image,predictor, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Mark ROIs - Slice {slice_index}")
        self.slice_index = slice_index
        self.image = rotate_and_flip_image(image)
        self.rois = []
        self.current_roi = []
        self.predictor = predictor
        self.segmentation_mode = False
        self.drawing_mode = False
        self.segmented_masks = {} 

        pixmap = self.numpy_to_pixmap(self.image)
        self.pixmap = pixmap
        self.original_pixmap = pixmap.copy()  # This should be the clean untouched image

        # === GUI Layout ===
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # QLabel to display image — now using ClickableImageLabel
        self.image_label = ClickableImageLabel()
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setScaledContents(True)

        # Connect click signal for segmentation
        # self.image_label.clicked.connect(self.handle_image_click)
        self.image_label.clicked.connect(lambda x, y, b: self.handle_image_click(x, y, b))

        # Scroll area for zoom/resize
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)

        layout.addWidget(scroll_area)

        # === Buttons ===
        button_layout = QHBoxLayout()

        draw_btn = QPushButton("Draw ROIs")
        draw_btn.clicked.connect(self.draw_rois)

        delete_btn = QPushButton("Delete ROIs")
        delete_btn.clicked.connect(self.delete_rois)

        segment_btn = QPushButton("Segmentation")
        segment_btn.clicked.connect(self.enable_segmentation_mode)

        finish_btn = QPushButton("Finish")
        finish_btn.clicked.connect(self.finish_and_close)

        button_layout.addWidget(draw_btn)
        button_layout.addWidget(delete_btn)
        button_layout.addWidget(segment_btn)
        button_layout.addWidget(finish_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def grayscale_to_rgb(image):
        """Convert 2D grayscale image to 3-channel RGB."""
        if image.ndim != 2:
            raise ValueError("Input must be a 2D grayscale image.")
        return np.stack([image] * 3, axis=-1)

    def numpy_to_pixmap(self, array):
        """Convert a 2D numpy array to QPixmap for display."""
        norm = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
        img = norm.astype(np.uint8)
        qimage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimage)

    def draw_rois(self):
        print("🖊️ Drawing mode ON")
        self.drawing_mode = True
        self.segmentation_mode = False  # Make sure segmentation isn't active
        self.current_roi = []

    def delete_rois(self):
        print("🗑️ Delete ROIs")
        # Clear drawn ROIs
        self.rois = []
        self.current_roi = []

        # Also clear segmented masks for this slice if exists
        if hasattr(self, 'segmented_masks') and self.slice_index in self.segmented_masks:
            self.segmented_masks[self.slice_index] = []

        self.update_image()

    def enable_segmentation_mode(self):
        print("🧠 Segmentation mode ON. Click on image to segment.")
        self.segmentation_mode = True
        self.drawing_mode = False  # disable ROI drawing

    def handle_image_click(self, x, y,button):
        if self.segmentation_mode:
            print(f"🧠 Running segmentation at ({x}, {y})")
            self.segmentation_mode = False  # Optional: one-time segment
            # Map GUI click to image coordinates:
            label_size = self.image_label.size()
            pixmap_size = self.pixmap.size()  # size of the pixmap shown on QLabel

            scale_x = self.image.shape[1] / label_size.width()
            scale_y = self.image.shape[0] / label_size.height()

            mapped_x = int(x * scale_x)
            mapped_y = int(y * scale_y)

            # Call SAM
            self.run_sam_segmentation(mapped_x, mapped_y)

        elif self.drawing_mode:
            print(f"✏️ Drawing point at ({x}, {y})")

            label_size = self.image_label.size()
            scale_x = self.image.shape[1] / label_size.width()
            scale_y = self.image.shape[0] / label_size.height()
            mapped_x = int(x * scale_x)
            mapped_y = int(y * scale_y)
            point = QPoint(mapped_x, mapped_y)

            if button == Qt.LeftButton:
                print(f"✏️ Drawing point at ({mapped_x}, {mapped_y})")
                self.current_roi.append(point)
                self.update_image()

            elif button == Qt.RightButton:
                if len(self.current_roi) >= 3:
                    print("✅ ROI finalized.")
                    self.rois.append(self.current_roi.copy())
                else:
                    print("⚠️ ROI too small, discarded.")
                self.current_roi = []
                self.update_image()

        else:
            print("🖱️ Click ignored, not in segmentation or drawing mode.")

    def run_sam_segmentation(self, x, y):
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        
        # Define bounding box size (adjust as needed)
        box_size = 60
        x_min = max(x - box_size // 2, 0)
        y_min = max(y - box_size // 2, 0)
        x_max = min(x + box_size // 2, self.image.shape[1] - 1)
        y_max = min(y + box_size // 2, self.image.shape[0] - 1)
        
        bbox = np.array([x_min, y_min, x_max, y_max])

        img_rgb = self.prepare_rgb_image(self.image)
        self.predictor.set_image(img_rgb)

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=bbox[None, :],  # SAM expects batch of boxes
            multimask_output=True
        )

        # Choose the best mask, or your preferred one
        best_mask = masks[np.argmax(scores)]
        # best_mask = masks[2]
        overlay = self.overlay_mask_on_image(self.image, best_mask)

        pixmap = self.numpy_rgb_to_qpixmap(overlay)
        self.image_label.setPixmap(pixmap)

        if self.slice_index not in self.segmented_masks:
            self.segmented_masks[self.slice_index] = []

        self.segmented_masks[self.slice_index].append(best_mask)
        print(f"🧠 Stored SAM mask for slice {self.slice_index} (total: {len(self.segmented_masks[self.slice_index])})")

    def overlay_mask_on_image(self, image, mask, color=(0, 255, 0), alpha=0.5):
        # Convert float64 image to uint8
        img_8u = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        img_8u = img_8u.astype(np.uint8)

        # Convert grayscale to RGB
        image_rgb = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2RGB)

        # Create colored mask
        colored_mask = np.zeros_like(image_rgb)
        colored_mask[mask] = color

        # Overlay mask on image with transparency
        overlay = cv2.addWeighted(colored_mask, alpha, image_rgb, 1 - alpha, 0)

        return overlay

    def prepare_rgb_image(self, img):
        """Ensure grayscale image is converted to RGB and uint8 for SAM input."""
        import cv2
        import numpy as np

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)

        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    def numpy_rgb_to_qpixmap(self, img_rgb):
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)
    
    def finish_and_close(self):
        # Ensure segmented_masks exists
        if not hasattr(self, 'segmented_masks'):
            self.segmented_masks = {}

        # Combine manually drawn ROIs and SAM masks if any
        if self.rois:
            print(f"🔴 Saving {len(self.rois)} manual ROIs for slice {self.slice_index}")
            if self.slice_index not in self.segmented_masks:
                self.segmented_masks[self.slice_index] = []
            self.segmented_masks[self.slice_index].extend(self.rois)

        if self.slice_index in self.segmented_masks:
            print(f"✅ Total masks for slice {self.slice_index}: {len(self.segmented_masks[self.slice_index])}")
        else:
            print(f"⚠️ No masks found for slice {self.slice_index}")

        print("📤 Emitting finished signal...")
        self.finished.emit()
        self.close()

    def eventFilter(self, source, event):
        if source is self.image_label:
            if self.drawing_mode and event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    pos = event.pos()
                    self.current_roi.append(pos)
                    self.update_image()

                elif event.button() == Qt.RightButton:
                    if len(self.current_roi) > 2:
                        self.rois.append(self.current_roi)
                    self.current_roi = []
                    self.update_image()

            elif self.segmentation_mode and event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    x = event.pos().x()
                    y = event.pos().y()
                    self.map_and_run_sam(x, y)
            return True
        return super().eventFilter(source, event)

    def update_image(self):
        print(f"Updating image: QLabel size {self.image_label.size()}, pixmap size {self.original_pixmap.size()}")
        print(f"Number of ROIs: {len(self.rois)}, current ROI points: {len(self.current_roi)}")

        pixmap_copy = self.original_pixmap.copy()
        painter = QPainter(pixmap_copy)
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)

        for roi in self.rois:
            polygon = QPolygon(roi)
            painter.drawPolygon(polygon)

        if self.current_roi:
            for i in range(len(self.current_roi) - 1):
                painter.drawLine(self.current_roi[i], self.current_roi[i + 1])

        painter.end()
        self.image_label.setPixmap(pixmap_copy)

    def parse_input(self, text):
        return [int(x.strip()) for x in text.split(',') if x.strip().isdigit()]


    def Mark_Rois(self):
        self.marked_slices = self.parse_input(self.mri_input.text())  # or wherever your input is
        self.slice_queue = [int(s) for s in self.marked_slices if str(s).isdigit()]
        self.current_roi_index = 0

        
        self.launch_next_roi_window()
    
    def start_marking_rois(self):
        self.marked_slices = self.get_selected_mri_slices()  # however you get them
        self.current_roi_index = 0
        self.launch_next_roi_window()

    def launch_next_roi_window(self):
        if self.current_roi_index >= len(self.marked_slices):
            print("✅ All ROIs marked.")
            return

        slice_index = self.marked_slices[self.current_roi_index]
        mri_slice = self.mri[:, :, slice_index]  # assuming self.mri is 3D

        self.roi_window = ROIMarker(slice_index, mri_slice, predictor=self.predictor)
        self.roi_window.finished.connect(self.on_roi_window_closed)
        self.roi_window.show()

    def on_roi_window_closed(self):
        slice_index = self.roi_window.slice_index
        masks = self.roi_window.segmented_masks.get(slice_index, [])

        if masks:
            if slice_index not in self.saved_masks:
                self.saved_masks[slice_index] = []
            self.saved_masks[slice_index].extend(masks)
            print(f"✅ Saved {len(masks)} masks for slice {slice_index}")

        self.current_roi_index += 1
        self.launch_next_roi_window()

        



