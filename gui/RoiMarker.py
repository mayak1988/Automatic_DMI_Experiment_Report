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
from gui.ClickableImageLabel import ClickableImageLabel

from config.config import SAM_CHECKPOINT_PATH,model


class ROIMarker(QWidget):
    """
    A PyQt5 widget for interactively marking Regions of Interest (ROIs) on a given MRI image.

    This widget displays a single MRI slice and allows the user to manually select or draw ROIs.
    The selected ROIs can be used later for signal analysis, segmentation, or visualization.

    Attributes:
        mri_image (np.ndarray): The 2D MRI slice to be displayed and marked.
        roi_mask (np.ndarray): A binary mask (same shape as mri_image) where selected ROI pixels are marked as 1.
        current_roi (list): A list of points (QPoint or (x, y) tuples) defining the currently drawn ROI.
        parent (QWidget): Optional parent widget.

    Methods:
        initUI(): Initializes the user interface layout and image display.
        mousePressEvent(event): Starts a new ROI by registering the first point.
        mouseMoveEvent(event): Adds points to the current ROI as the mouse moves.
        mouseReleaseEvent(event): Finalizes the ROI and updates the mask.
        paintEvent(event): Draws the image and ROI outlines on the widget.
        get_roi_mask(): Returns the binary mask of the selected ROI.
        clear_rois(): Clears all ROIs from the view and mask.
    """
    finished = pyqtSignal()  # Signal to notify when "Finish" is clicked

    def __init__(self, slice_index, image,predictor, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Mark ROIs - Slice {slice_index}")
        self.slice_index = slice_index
        self.image = self.rotate_and_flip_image(image)
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
        """
        Converts a 2D grayscale image to a 3-channel RGB image.

        This function replicates the single grayscale channel across the three RGB channels,
        producing an RGB image that visually appears the same as the grayscale input.

        Parameters:
        ----------
        image : np.ndarray
            A 2D numpy array representing a grayscale image.

        Returns:
        -------
        np.ndarray
            A 3D numpy array of shape (H, W, 3) representing the RGB image.

        Raises:
        ------
        ValueError
            If the input image is not a 2D array.
        """
        if image.ndim != 2:
            raise ValueError("Input must be a 2D grayscale image.")
        return np.stack([image] * 3, axis=-1)

    def rotate_and_flip_image(self,image):
        """
        Rotates the input image 90 degrees counterclockwise and flips it vertically.

        This function is typically used to reorient image slices for consistent visualization
        or alignment with anatomical conventions. The image is first rotated 90 degrees
        counterclockwise, then flipped along the vertical axis (Y-axis).

        Parameters:
        ----------
        image : np.ndarray
            A 2D numpy array representing a single image slice.

        Returns:
        -------
        np.ndarray
            The transformed image after rotation and vertical flipping.
        """
        # Rotate the image 90 degrees
        rotated_image = np.rot90(image)
        # Flip the image along the Y-axis (vertical flip)
        flipped_image = np.flipud(rotated_image)
        return flipped_image
   
    def numpy_to_pixmap(self, array):
        """
        Converts a 2D NumPy array to a QPixmap for display in PyQt.

        This function normalizes the input array to the [0, 255] range,
        converts it to an 8-bit grayscale image (QImage), and then wraps it
        into a QPixmap for use in PyQt GUIs.

        Parameters:
        ----------
        array : np.ndarray
            A 2D NumPy array representing grayscale image data.

        Returns:
        -------
        QPixmap
            A QPixmap object that can be displayed in PyQt widgets.

        Raises:
        ------
        ValueError
            If the input is not a 2D array.
        """
        norm = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
        img = norm.astype(np.uint8)
        qimage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimage)

    def draw_rois(self):
        """
        Enables manual ROI (Region of Interest) drawing mode in the GUI.

        This method activates drawing mode so the user can manually define ROIs
        on the image. It also ensures that segmentation mode is turned off to
        prevent conflicts. The current ROI container is reset for new input.

        Actions:
        --------
        - Sets `self.drawing_mode` to True.
        - Sets `self.segmentation_mode` to False.
        - Resets `self.current_roi` to an empty list.
        - Prints a status message to the console.
        """
        print("🖊️ Drawing mode ON")
        self.drawing_mode = True
        self.segmentation_mode = False  # Make sure segmentation isn't active
        self.current_roi = []

    def delete_rois(self):
        """
        Deletes all drawn and segmented ROIs for the current image slice.

        This method clears manually drawn ROIs and, if present, any segmented
        ROI masks associated with the current slice. After clearing, it updates
        the displayed image to reflect the changes.

        Actions:
        --------
        - Clears `self.rois` and `self.current_roi`.
        - If `self.segmented_masks` exists and contains masks for the current
        `self.slice_index`, those are also cleared.
        - Calls `self.update_image()` to refresh the display.
        - Prints a status message to the console.
        """
        print("🗑️ Delete ROIs")
        # Clear drawn ROIs
        self.rois = []
        self.current_roi = []

        # Also clear segmented masks for this slice if exists
        if hasattr(self, 'segmented_masks') and self.slice_index in self.segmented_masks:
            self.segmented_masks[self.slice_index] = []

        self.update_image()

    def enable_segmentation_mode(self):
        """
        Enables segmentation mode in the GUI for automatic or manual segmentation.

        This method activates segmentation mode, allowing the user to click on the
        image to generate segmented regions. It also disables manual ROI drawing mode
        to prevent mode conflicts.

        Actions:
        --------
        - Sets `self.segmentation_mode` to True.
        - Sets `self.drawing_mode` to False.
        - Prints a status message to the console.
        """
        print("🧠 Segmentation mode ON. Click on image to segment.")
        self.segmentation_mode = True
        self.drawing_mode = False  # disable ROI drawing

    def handle_image_click(self, x, y,button):
        """
        Handles mouse click events on the displayed image for segmentation or ROI drawing.

        Depending on the current mode, this method processes clicks to either
        perform image segmentation or add points to a manually drawn ROI.

        Parameters:
        -----------
        x : int or float
            X coordinate of the mouse click relative to the QLabel widget.
        y : int or float
            Y coordinate of the mouse click relative to the QLabel widget.
        button : Qt.MouseButton
            Mouse button clicked (e.g., Qt.LeftButton, Qt.RightButton).

        Behavior:
        ---------
        - If in segmentation mode:
            * Converts click coordinates to image space.
            * Runs the SAM segmentation at the clicked location.
            * Optionally disables segmentation mode after one use.
        - If in drawing mode:
            * Converts click coordinates to image space.
            * Left-click adds a point to the current ROI polygon.
            * Right-click finalizes the ROI if it has at least 3 points,
            otherwise discards it.
        - If neither mode is active:
            * Prints a message that the click is ignored.

        Notes:
        ------
        - Coordinate mapping accounts for QLabel resizing relative to the image.
        - `self.update_image()` is called to refresh the display after changes.
        """
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
        """
        Runs SAM (Segment Anything Model) segmentation on the image at the given coordinates.

        Given a click position (x, y) on the image, this method:
        - Defines a bounding box centered on (x, y) to localize the segmentation.
        - Prepares the image in RGB format suitable for the SAM predictor.
        - Uses the SAM predictor to generate multiple segmentation masks around the point.
        - Selects the best scoring mask.
        - Overlays the selected mask on the original image.
        - Updates the displayed image with the overlay.
        - Stores the mask in `self.segmented_masks` keyed by the current slice index.

        Parameters:
        -----------
        x : int
            X-coordinate (column) in image pixel space.
        y : int
            Y-coordinate (row) in image pixel space.

        Notes:
        ------
        - The bounding box size is fixed (default 60 pixels) but can be adjusted.
        - The method assumes `self.image` is the current grayscale or RGB image slice.
        - The `self.predictor` must be pre-initialized with the SAM model.
        - The overlay uses a helper method `overlay_mask_on_image`.
        - The result is immediately displayed in `self.image_label`.
        - Segmentation masks are accumulated per slice in `self.segmented_masks`.

        Raises:
        -------
        - IndexError if coordinates fall outside image bounds (handled by bounding box clipping).
        - AttributeError if `self.predictor` or other dependencies are missing.
        """
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
        """
        Overlays a binary mask on a grayscale image with a specified color and transparency.

        Parameters:
        -----------
        image : np.ndarray
            A 2D grayscale image array (typically float64 or similar).
        mask : np.ndarray (bool)
            A binary mask array of the same shape as `image`, where True values indicate
            the region to overlay.
        color : tuple of int, optional
            RGB color to use for the mask overlay (default is green: (0, 255, 0)).
        alpha : float, optional
            Transparency factor for the overlay mask (range 0.0 to 1.0, default 0.5).

        Returns:
        --------
        np.ndarray
            An RGB image array with the colored mask overlay applied with the specified transparency.

        Notes:
        ------
        - The input grayscale image is normalized to 8-bit unsigned integers.
        - The mask is applied by coloring the pixels where `mask` is True.
        - OpenCV's `addWeighted` function is used for alpha blending.
        """
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
        """
        Converts a grayscale or non-uint8 image to an 8-bit 3-channel RGB image suitable for SAM.

        This method normalizes the input image to the 0-255 range and converts it to uint8 if needed.
        If the image is grayscale (2D), it converts it to RGB by duplicating the channel.

        Parameters:
        -----------
        img : np.ndarray
            Input image, either grayscale (2D) or color (3D), with any dtype.

        Returns:
        --------
        np.ndarray
            3-channel RGB image of dtype uint8 normalized to [0, 255].

        Notes:
        ------
        - Input images with dtype other than uint8 are normalized and converted.
        - Grayscale images are converted to RGB by channel replication.
        """

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)

        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    def numpy_rgb_to_qpixmap(self, img_rgb):
        """
        Converts a 3-channel RGB NumPy array to a QPixmap for display in PyQt.

        Parameters:
        -----------
        img_rgb : np.ndarray
            A 3D numpy array of shape (height, width, 3) with dtype uint8 representing
            an RGB image.

        Returns:
        --------
        QPixmap
            A QPixmap object that can be used for displaying the image in PyQt widgets.

        Notes:
        ------
        - The input image must be in RGB order (not BGR).
        - The image data is not copied, so the original array must remain in scope while the
        QPixmap is in use.
        """
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)
    
    def finish_and_close(self):
        """
        Finalizes the current ROI and segmentation work, saves masks, and closes the window.

        This method ensures that:
        - The `segmented_masks` attribute exists.
        - Any manually drawn ROIs for the current slice are appended to the segmented masks.
        - Prints status messages about the number of saved masks.
        - Emits a `finished` signal to notify listeners that processing is complete.
        - Closes the GUI window.

        Notes:
        ------
        - Designed to be called when the user finishes ROI/segmentation editing.
        - Merges manual ROIs (`self.rois`) with existing SAM-based masks (`self.segmented_masks`).
        - Relies on `self.slice_index` to identify the current image slice.
        - Emits a Qt signal `self.finished`.
        """
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
        """
        Event filter to handle mouse interactions on the image label for drawing and segmentation modes.

        Parameters:
        -----------
        source : QObject
            The object that sent the event (expected to be `self.image_label`).
        event : QEvent
            The event object to be processed.

        Behavior:
        ---------
        - When the source is `self.image_label`:
            * In drawing mode:
                - Left mouse click adds a point to the current ROI.
                - Right mouse click finalizes the ROI if it has more than 2 points,
                then resets the current ROI.
                - Updates the image display after modifications.
            * In segmentation mode:
                - Left mouse click triggers SAM segmentation at the clicked position.
        - Returns True if the event is handled here to prevent further propagation.
        - Otherwise, passes the event to the superclass eventFilter for default handling.

        Returns:
        --------
        bool
            True if the event was handled, False otherwise.
        """
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
        """
        Updates the displayed image by redrawing all stored ROIs and the currently drawn ROI.

        This method creates a copy of the original pixmap, then uses a QPainter
        to overlay:
        - All finalized ROIs in red polygons.
        - The current ROI in-progress as connected red lines between points.

        It then sets the updated pixmap back to the image label for display.

        Additional:
        ---------
        - Prints debug info including QLabel and pixmap sizes and ROI counts.
        """
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

