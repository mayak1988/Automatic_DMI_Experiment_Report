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

def rotate_and_flip_image(image):
    # Rotate the image 90 degrees
    rotated_image = np.rot90(image)
    # Flip the image along the Y-axis (vertical flip)
    flipped_image = np.flipud(rotated_image)
    return flipped_image

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


class MRICSIViewer(QWidget):
    def __init__(self):
        super().__init__()

        # Set a wider default size (e.g., 1400x900)
        self.resize(900, 500)  

        self.predictor = None 

        self.initUI()
        # Load SAM model
        self.sam_checkpoint = r'C:\Users\luciogrp\Documents\GitHub\Automatic_DMI_Experiment_Report\data\sam_vit_b_01ec64.pth'
        self.sam_model  = sam_model_registry["vit_b"](checkpoint=self.sam_checkpoint)
        self.sam_model.to("cpu") # Or "cpu" if no GPU available
        self.predictor = SamPredictor(self.sam_model)
        self.roi_colors = plt.cm.tab20.colors  # or tab10.colors for 10
        self.mri = None
        self.csi = None
        # Define substances
        self.substances = ['HDO', 'Glucose', 'Lactate']
        self.slice_queue = []       # list of slice indices to show
        self.current_roi_index = 0  # where we are in the list
        self.saved_masks = {}

    def initUI(self):
        self.setWindowTitle("MRI + CSI Viewer")

        main_layout = QVBoxLayout()  # Create only once!

        # --- Experiment Metadata ---
        meta_layout = QHBoxLayout()
        self.date_input = QLineEdit()
        self.date_input.setPlaceholderText("Experiment Date (YYYY-MM-DD)")
        self.participant_input = QLineEdit()
        self.participant_input.setPlaceholderText("Participant ID")
        self.description_input = QTextEdit()
        self.description_input.setPlaceholderText("Experiment Description")
        self.description_input.setFixedHeight(60)

        meta_layout.addWidget(QLabel("Date:"))
        meta_layout.addWidget(self.date_input)
        meta_layout.addWidget(QLabel("Participant:"))
        meta_layout.addWidget(self.participant_input)

        main_layout.addLayout(meta_layout)
        main_layout.addWidget(QLabel("Description:"))
        main_layout.addWidget(self.description_input)
        

        # Add the coil_type_combo to main_layout (or meta_layout, depending on design)
        self.coil_type_combo = QComboBox()
        self.coil_type_combo.addItems(["Surface Coil", "Volume Coil"])

        # Add combo box to the main_layout so it shows up in your UI
        main_layout.addWidget(QLabel("Coil Type:"))
        main_layout.addWidget(self.coil_type_combo)
        
        # --- Load Button ---
        load_btn = QPushButton("Load .mat File")
        load_btn.clicked.connect(self.load_mat)
        main_layout.addWidget(load_btn)

        # --- Show Viewer ---
        self.button = QPushButton("Show Viewer")
        self.button.clicked.connect(self.open_viewer)
        main_layout.addWidget(self.button)

        # --- Manual Slice Input ---
        input_layout = QHBoxLayout()
        self.mri_input = QLineEdit()
        self.mri_input.setPlaceholderText("MRI slices (e.g. 1,3,5)")
        # self.csi_input = QLineEdit()
        # self.csi_input.setPlaceholderText("CSI slices (e.g. 2,3,5)")
        input_layout.addWidget(self.mri_input)
        # input_layout.addWidget(self.csi_input)
        main_layout.addLayout(input_layout)

        display_btn = QPushButton("Mark ROIs")
        display_btn.clicked.connect(self.Mark_Rois)
        main_layout.addWidget(display_btn)

        display_btn1 = QPushButton("Generate Report")
        display_btn1.clicked.connect(self.Generate_Report)
        main_layout.addWidget(display_btn1)


        exit_btn = QPushButton("Exit")
        exit_btn.clicked.connect(QApplication.quit)
        main_layout.addWidget(exit_btn)


        # ✅ Set the final layout ONCE at the end
        self.setLayout(main_layout)
        self.show()

    def open_viewer(self):
        self.viewer = DualSliceViewer(self.mri, self.csi)
        self.viewer.show()

    def load_mat(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open .mat File", "", "MAT-files (*.mat)")
        if not path: 
            return
        mat = scipy.io.loadmat(path)
        self.mri = np.abs(mat['res'][0, 0]['anaimage'])    # If entire array is complex
        self.csi = np.abs(mat['res'][0, 0]['SepImage'])
        print("Loaded MRI shape:", self.mri.shape)
        print("Loaded CSI shape:", self.csi.shape)

    def map_mri_to_csi_pairs(self,selected_mri_slices, num_mri_slices, num_csi_slices):
        csi_to_mri = {}
        for mri_idx in selected_mri_slices:
            csi_idx = round(mri_idx * (num_csi_slices / num_mri_slices))
            csi_idx = min(csi_idx, num_csi_slices - 1)
            if csi_idx not in csi_to_mri:
                csi_to_mri[csi_idx] = mri_idx  # keep first MRI that maps to CSI

        # Return as two aligned lists: one-to-one MRI and CSI slices
        csi_slices = sorted(csi_to_mri.keys())
        mri_slices = [csi_to_mri[csi] for csi in csi_slices]
        return mri_slices, csi_slices
    
    def normalize_csi_slice(self,csi_slice):
        csi_slice_abs = np.abs(csi_slice)  # Take the magnitude of the complex CSI data
        min_val = np.min(csi_slice_abs)
        max_val = np.max(csi_slice_abs)
        return (csi_slice_abs - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    
    # Display the anatomical images for the selected slices only
    def display_anatomic_images(self,anatomic, selected_slices,save_as=None):
        # Function to rotate image 90 degrees and flip it along the Y-axis

        
        num_images_per_row = 4
        num_rows = (len(selected_slices) + num_images_per_row - 1) // num_images_per_row  # Calculate rows based on selected slices
        
        fig, axes = plt.subplots(num_rows, num_images_per_row, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        for idx, slice_num in enumerate(selected_slices):
            ax = axes[idx]
            # Apply rotation and flip before displaying
            rotated_flipped_image = rotate_and_flip_image(anatomic[:, :, slice_num])
            ax.imshow(rotated_flipped_image, cmap='gray')
            ax.set_title(f'Slice {slice_num + 1}')
            ax.axis('off')

        # Turn off remaining axes if there are fewer slices than spaces
        for i in range(len(selected_slices), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

            # Save the figure as a JPEG if a save path is specified
        if save_as:
            plt.savefig(save_as, format='jpeg', dpi=300)  # Save with 300 dpi for high quality
            print(f"Figure saved as {save_as}")


        plt.show()
    
    # Overlay CSI on anatomical images for the selected slices and all scans
    def overlay_csi_on_anatomic(self,anatomic, csi, anatomic_slice_idx, csi_slice_idx, substances,save_folder=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from scipy.ndimage import zoom

        num_scans = csi.shape[4]
        ncols = 4
        nrows = (num_scans + ncols - 1) // ncols

        # Prepare anatomical background image once
        anatomic_img = rotate_and_flip_image(anatomic[:, :, anatomic_slice_idx])

        for substance_idx, substance_name in enumerate(substances):
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
            axes = axes.flatten()

            for scan_idx in range(num_scans):
                ax = axes[scan_idx]
                csi_slice = csi[:, :, csi_slice_idx, substance_idx, scan_idx]
                csi_slice_norm = self.normalize_csi_slice(csi_slice)

                # Resize CSI from (64, 64) to (512, 512)
                zoom_factor_x = anatomic_img.shape[0] / csi_slice_norm.shape[0]
                zoom_factor_y = anatomic_img.shape[1] / csi_slice_norm.shape[1]
                csi_resized = zoom(csi_slice_norm, (zoom_factor_x, zoom_factor_y), order=1)  # bilinear

                # Now overlay
                
                ax.imshow(csi_resized, cmap='jet', alpha=0.7)
                ax.imshow(anatomic_img, cmap='gray', alpha=0.6)
                ax.set_title(f" Scan {scan_idx + 1}")
                ax.axis('off')

            # Turn off unused subplots
            for ax in axes[num_scans:]:
                ax.axis('off')

            # plt.suptitle(f"{substance_name} - All Scans", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            # Save one image per substance
            
            if save_folder:
                save_path = os.path.join(save_folder, f"Anat_{anatomic_slice_idx}_CSI_{csi_slice_idx}_{substance_name}.jpg")
            else:
                save_path = f"Anat_{anatomic_slice_idx}_CSI_{csi_slice_idx}_{substance_name}.jpg"

            plt.savefig(save_path, format='jpeg', dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"Figure saved as {save_path}")

            plt.close()
    
    def insert_overlay_images_to_pdf(self,pdf, report_folder, mri_slices, csi_slices, substances,
                                  overlay_func, mri_data, csi_data, current_y, page_height):
        """
        Generate overlay images and insert them into a PDF.

        Parameters:
            pdf: reportlab.pdfgen.canvas.Canvas
                The PDF canvas object.
            report_folder: str
                Directory to save and load overlay images.
            mri_slices: list of int
                Selected MRI slice indices.
            csi_slices: list of int
                Corresponding CSI slice indices.
            substances: list of str
                Substance names (e.g. ['H2O', 'Glu', 'Lac']).
            overlay_func: callable
                Function that generates and saves overlay images.
            mri_data: np.ndarray
                The anatomical MRI volume.
            csi_data: np.ndarray
                The CSI volume.
            current_y: float
                Current vertical position on the PDF page.
            page_height: float
                Total page height in points (e.g. A4 height).

        Returns:
            float: updated current_y position after inserting images.
        """


        for i, (anatomic_slice_idx, csi_slice_idx) in enumerate(zip(mri_slices, csi_slices)):
            # Generate overlay images for this slice pair
            overlay_func(
                mri_data,
                csi_data,
                anatomic_slice_idx,
                csi_slice_idx,
                substances,
                save_folder=report_folder 
            )

            for substance_name in substances:
                overlay_jpg_path = os.path.join(
                    report_folder,
                    f"Anat_{anatomic_slice_idx}_CSI_{csi_slice_idx}_{substance_name}.jpg"
                )

                if not os.path.exists(overlay_jpg_path):
                    print(f"⚠️ Warning: Image not found: {overlay_jpg_path}")
                    continue

                overlay_img = ImageReader(overlay_jpg_path)
                img_width = 500
                img_height = 400

                if current_y - img_height < 50:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 12)
                    current_y = page_height - 100

                # Define label text
                label_text = f"Overlay: MRI Slice {anatomic_slice_idx}, CSI Slice {csi_slice_idx}, {substance_name}"
                font_name = "Helvetica"
                font_size = 12
                pdf.setFont(font_name, font_size)

                # Calculate image and text position
                img_x = 50
                img_center_x = img_x + img_width / 2
                text_width = stringWidth(label_text, font_name, font_size)
                text_x = img_center_x - text_width / 2
                text_y = current_y  # Start at the current Y

                # Draw label above image
                pdf.drawString(text_x, text_y, label_text)

                # Draw image below label
                pdf.drawImage(overlay_img, img_x, text_y - img_height - 10,
                            width=img_width, height=img_height)

                # Update current_y for next image block
                current_y = text_y - img_height - 30

        return current_y
    
    def plot_roi_intensities(self, roi_averages, substances, save_folder):
        num_scans = len(next(iter(roi_averages.values()))[0])  # assume all ROIs have same scan count
        x = np.arange(num_scans)

        for substance in substances:
            fig, ax = plt.subplots(figsize=(8, 5))
            all_roi_series = roi_averages[substance]

            for idx, roi_values in enumerate(all_roi_series):
                color = self.roi_colors[idx % len(self.roi_colors)]
                ax.plot(x, roi_values, label=f'ROI {idx+1}', marker='o', color=color)

            ax.set_title(f'{substance} Intensity Over Time')
            ax.set_xlabel('Scan Number')
            ax.set_ylabel('Mean Intensity')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()

            save_path = os.path.join(save_folder, f"{substance}_intensity_plot.jpg")
            plt.savefig(save_path, dpi=150)
            plt.close()

    def safe_report_name(self,date_str):
        """
        Convert a user-provided date to a safe folder name.
        Example: '2025/06/26' ➝ '2025_06_26'
        If invalid, returns 'default_report'.
        """
        if isinstance(date_str, str):
            date_str = date_str.strip()
            pattern = r"^\d{4}/\d{2}/\d{2}$"
            if re.match(pattern, date_str):
                return date_str.replace("/", "_")  # Safe name

        # Invalid format fallback
        return "default_report"
    
    def convert_roi_to_mask(self, rois, image_shape):
        import numpy as np
        import cv2

        mask = np.zeros(image_shape, dtype=np.uint8)

        for roi in rois:
            # If it's a numpy array (mask) already, just combine it
            if isinstance(roi, np.ndarray) and roi.dtype == bool:
                mask = np.logical_or(mask, roi).astype(np.uint8)
            # Else, assume it's a polygon (list of QPoint)
            elif isinstance(roi, list):
                pts = []
                for p in roi:
                    if hasattr(p, 'x') and hasattr(p, 'y'):
                        pts.append([int(p.x()), int(p.y())])
                    else:
                        raise ValueError(f"Unexpected point format in ROI: {p}")
                pts_np = np.array(pts, dtype=np.int32)
                cv2.fillPoly(mask, [pts_np], 1)
            else:
                raise ValueError(f"Unrecognized ROI format: {type(roi)}")

        return mask.astype(bool)

    def plot_slice_with_rois(self, mri_slice, masks, save_path=None, title=None):
        # Flip the image for consistent display
        flipped_mri = rotate_and_flip_image(mri_slice)

        fig, ax = plt.subplots()
        ax.imshow(flipped_mri, cmap='gray')

        for idx, mask in enumerate(masks):
            if mask.shape != flipped_mri.shape:
                print(f"⚠️ Mask shape {mask.shape} doesn't match MRI shape {flipped_mri.shape}. Skipping.")
                continue

            contours = measure.find_contours(mask, level=0.5)
            color = self.roi_colors[idx % len(self.roi_colors)]

            for contour in contours:   
                ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)

        if title:
            ax.set_title(title)
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def calculate_roi_intensity(self, csi, rois_with_slices, substances, num_scans):
        roi_averages = {substance: [] for substance in substances}  # Dict: substance → list of [per-scan avg values for each ROI]

        for substance_idx, substance in enumerate(substances):
            for slice_idx, roi_mask in rois_with_slices:
            
                roi_values = []

                for scan_num in range(num_scans):
                    csi_slice = csi[:, :, slice_idx, substance_idx, scan_num]  # 2D

                    # Resize mask if needed
                    if roi_mask.shape != csi_slice.shape:
                        roi_mask_resized = cv2.resize(
                            roi_mask.astype(np.uint8),
                            (csi_slice.shape[1], csi_slice.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                    else:
                        roi_mask_resized = roi_mask

                    roi_values.append(np.mean(csi_slice[roi_mask_resized > 0]))

                # 👇 Store the list of per-scan intensities
                roi_averages[substance].append(roi_values)
        
        return roi_averages
        
    def Generate_Report(self):
        # Ask for location - default s reports folder
        # Ask for file name - default name is date_report 
        # print locaton
        reports_base_folder = "reports"  # Your existing folder

        # Ensure the reports folder exists
        os.makedirs(reports_base_folder, exist_ok=True)

        # Create report folder name based on date input or fallback name
                    # Example usage:
        raw_date = self.date_input.text().strip()
        # Remove slashes and other unwanted characters
        report_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_date)

        # Optional: fallback to default if empty or looks wrong
        if not re.match(r'^\d{4}_\d{2}_\d{2}$', report_name):
            report_name = "default"
        
        report_folder = os.path.join(reports_base_folder, f"{report_name}_report")

        # Create the report folder if it doesn't exist
        os.makedirs(report_folder, exist_ok=True)


            # === Get all inputs ===
        date = self.date_input.text()
        participant = self.participant_input.text()
        description = self.description_input.toPlainText()
        mrislices = self.parse_input(self.mri_input.text())
        # If mrislices is a list of strings or numbers:
        selected_mri_slices = [int(s) for s in mrislices if str(s).strip().isdigit()]
            # Override CSI slice indices if surface coil
        coil_type = self.coil_type_combo.currentText()
        if coil_type.lower() == "surface coil":
            csi_slices = [1] * len(selected_mri_slices)
            mri_slices = selected_mri_slices
        else:  # set all CSI indices to 1
            mri_slices, csi_slices = self.map_mri_to_csi_pairs(selected_mri_slices,self.mri.shape[2], self.csi.shape[2])  # assuming they match; later can add self.csi_input


        if not mri_slices:
            print("No slices entered.")
            return


        
        
        # user_input = date.strip()
        # report_name = self.safe_report_name(user_input)
        

        default_name = f"{report_name}_report.pdf" if date else "experiment_report.pdf"
        default_path = os.path.join(report_folder, default_name)

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Report PDF",
            default_path,  # <-- start here inside created folder
            "PDF files (*.pdf)"
        )
        
        if not save_path:
            return
        
        # === Start PDF ===
        pdf = canvas.Canvas(save_path, pagesize=A4)
        width, height = A4
        pdf.setFont("Helvetica", 12)

        # === Header Metadata ===
        pdf.drawString(50, height - 50, f"🧪 MRI + DMI Experiment Report")
        pdf.drawString(50, height - 80, f"Date: {date}")
        pdf.drawString(50, height - 100, f"Participant: {participant}")
        pdf.drawString(50, height - 120, f"Description:")
        text_obj = pdf.beginText(70, height - 140)
        for line in description.splitlines():
            text_obj.textLine(line)
        pdf.drawText(text_obj)

        pdf.drawString(50, height - 200, f"MRI Shape: {self.mri.shape}")
        pdf.drawString(50, height - 220, f"CSI Shape: {self.csi.shape}")
        pdf.drawString(50, height - 240, f"Slices selected: {mri_slices}")

        # # === draw_mri_grid ===

        # anatomic_jpg_path = os.path.join(report_folder, "anatomic_slices.jpg")

        # # Save anatomical image inside report folder
        # slice_start = 7
        # slice_end = self.mri.shape[2] - 5
        # self.display_anatomic_images(self.mri, list(range(slice_start, slice_end)), save_as=anatomic_jpg_path)


        # # Load image to get actual size
        # anatomic_image = Image.open(anatomic_jpg_path)
        # img_orig_width, img_orig_height = anatomic_image.size

        # # Set desired width and compute height to preserve aspect ratio
        # img_width = 500
        # aspect_ratio = img_orig_height / img_orig_width
        # img_height = int(img_width * aspect_ratio)

        # # Draw text and image as before
        # # Constants
        # x_pos = 50
        # bottom_margin = 50

        # y_pos = height - 270

        # # Calculate available space
        # if y_pos - img_height < bottom_margin:
        #     pdf.showPage()
        #     y_pos = height - 100  # Reset top position
        #     pdf.setFont("Helvetica", 12)

        
        
        # img_center_x = x_pos + img_width / 2
        # text = "Anatomical MRI Slices"
        # font_name = "Helvetica"
        # font_size = 12
        # pdf.setFont(font_name, font_size)
        # text_width = stringWidth(text, font_name, font_size)
        # text_x = img_center_x - text_width / 2
        # text_y = y_pos

        # # Draw label above image
        # pdf.drawString(text_x, text_y, text)

        # # Draw the image just below the label with correct aspect ratio
        # img = ImageReader(anatomic_jpg_path)
        # pdf.drawImage(img, x_pos, text_y - img_height - 10, width=img_width, height=img_height)

        # # Update current_y
        # current_y = text_y - img_height - 30

        # === draw_mri_grid ===

        anatomic_jpg_path = os.path.join(report_folder, "anatomic_slices.jpg")

        # Save anatomical image inside report folder
        slice_start = 7
        slice_end = self.mri.shape[2] - 5
        self.display_anatomic_images(self.mri, list(range(slice_start, slice_end)), save_as=anatomic_jpg_path)

        # Load image to get actual size
        anatomic_image = Image.open(anatomic_jpg_path)
        img_orig_width, img_orig_height = anatomic_image.size

        # Page size and margins
        page_width, page_height = height, width  # Assuming height and width are your page dimensions
        left_margin = 50
        right_margin = 50
        top_margin = 50
        bottom_margin = 50

        max_width = page_width - left_margin - right_margin
        max_height = page_height - top_margin - bottom_margin - 50  # extra 50 for the text height + padding

        # Calculate scale factor to fit image inside max dimensions while preserving aspect ratio
        scale_w = max_width / img_orig_width
        scale_h = max_height / img_orig_height
        scale = min(scale_w, scale_h, 1)  # Don't upscale if smaller than max

        # Calculate scaled image dimensions
        img_width = img_orig_width * scale
        img_height = img_orig_height * scale

        # Calculate X position to center image and text horizontally
        img_x = (page_width - img_width) / 2

        # Calculate Y positions
        text_y = page_height - top_margin
        img_y = text_y - img_height - 10  # 10 pts padding below text

        # If image bottom goes below margin, start new page and reset positions
        if img_y < bottom_margin:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            text_y = page_height - top_margin
            img_y = text_y - img_height - 10

        # Draw text centered above image
        text = "Anatomical MRI Slices"
        font_name = "Helvetica"
        font_size = 12
        pdf.setFont(font_name, font_size)
        text_width = stringWidth(text, font_name, font_size)
        text_x = (page_width - text_width) / 2
        pdf.drawString(text_x, text_y, text)

        # Draw the image scaled and centered below the text
        img = ImageReader(anatomic_jpg_path)
        pdf.drawImage(img, img_x, img_y, width=img_width, height=img_height)

        # Update current_y for further content below
        current_y = img_y - 30


        # Display CSI overlays on the selected anatomical slice for all scans
        # === Overlay each MRI/CSI pair ===
        overlay_img_height = 300
        overlay_img_width = 500
        y_pos = height - 270
        current_y = y_pos - img_height - 100  # Start below the anatomical image

        current_y = self.insert_overlay_images_to_pdf(
            pdf=pdf,
            report_folder=report_folder,
            mri_slices=mri_slices,
            csi_slices=csi_slices,
            substances=self.substances,
            overlay_func=self.overlay_csi_on_anatomic,
            mri_data=self.mri,
            csi_data=self.csi,
            current_y=current_y,
            page_height=height
        )

                # Define a color map (e.g., tab10 gives 10 distinct colors)

        for slice_index, masks in self.saved_masks.items():
            mri_slice = self.mri[:, :, slice_index]
            save_path = os.path.join(report_folder, f"mri_rois_slice_{slice_index}.jpg")

                # Convert all ROIs (QPoint lists) to binary masks matching mri_slice shape
            binary_masks = []
            for roi in masks:
                # Convert polygon ROI to binary mask
                mask = self.convert_roi_to_mask(masks, mri_slice.shape)
                binary_masks.append(mask)

            self.plot_slice_with_rois(
                mri_slice=mri_slice,
                masks=binary_masks,
                save_path=save_path,
                title=f"MRI Slice {slice_index} with ROIs"
            )

            # Add to PDF
            img = ImageReader(save_path)
            img_width = 400
            img_height = 400
            if current_y - img_height < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                current_y = height - 100

            pdf.drawImage(img, 50, current_y - img_height,
                        width=img_width, height=img_height)
            current_y -= img_height + 60



        all_rois_masks = []

        for mri_idx, csi_idx in zip(self.marked_slices, csi_slices):
            masks = self.saved_masks.get(mri_idx, [])
            mri_slice_shape = self.mri[:, :, mri_idx].shape  # shape of MRI slice for mask size reference

            for mask in masks:
                # If mask is polygon points (list of QPoint or tuples), convert to numpy mask
                if isinstance(mask, list) or isinstance(mask, tuple):
                    np_mask = self.convert_roi_to_mask([mask], mri_slice_shape)  # expecting list of ROIs for this function
                elif isinstance(mask, np.ndarray):
                    np_mask = mask
                else:
                    raise ValueError(f"Unexpected mask format: {type(mask)}")

                all_rois_masks.append((csi_idx, np_mask))

        roi_averages = self.calculate_roi_intensity(self.csi, all_rois_masks, self.substances, self.csi.shape[4])
        

        # 2. Plot and save the ROI intensities graph
        self.plot_roi_intensities(roi_averages, self.substances, report_folder)

        # 3. Insert the plot image into the PDF
        for substance in self.substances:
            plot_img_path = os.path.join(report_folder, f"{substance}_intensity_plot.jpg")

            if os.path.exists(plot_img_path):
                img = ImageReader(plot_img_path)

                # Set desired image size
                img_width = 300  # smaller width
                img_height = 225  # smaller height (maintain aspect ratio)

                # Check for space; if not enough, create new page
                if current_y - img_height - 20 < 50:
                    pdf.showPage()
                    pdf.setFont("Helvetica", 12)
                    current_y = height - 100

                # Add label above image
                pdf.drawString(50, current_y, f"{substance} ROI Intensity Over Time:")
                current_y -= 15  # small gap after text

                # Draw image
                pdf.drawImage(img, 50, current_y - img_height,
                            width=img_width, height=img_height)
                current_y -= img_height + 40  # update current_y with spacing after image

            else:
                print(f"⚠️ Plot image not found: {plot_img_path}")

         # === Save PDF ===
        pdf.save()
        print(f"Report saved to: {save_path}")
      
    def parse_input(self, text):
        return [int(x.strip()) for x in text.split(',') if x.strip().isdigit()]

    def normalize(self, img):
        img = img - np.min(img)
        if np.max(img) > 0:
            img = img / np.max(img)
        return img

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
