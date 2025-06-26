import sys, io,os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QPushButton, QFileDialog, QSlider, QScrollArea, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal ,QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QPolygon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import datetime
import re
from reportlab.pdfbase.pdfmetrics import stringWidth
import cv2




def rotate_and_flip_image(image):
    # Rotate the image 90 degrees
    rotated_image = np.rot90(image)
    # Flip the image along the Y-axis (vertical flip)
    flipped_image = np.flipud(rotated_image)
    return flipped_image

class ROIMarker(QWidget):
    finished = pyqtSignal()  # Signal to notify when "Finish" is clicked

    def __init__(self, slice_index, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Mark ROIs - Slice {slice_index}")
        self.slice_index = slice_index
        self.image = rotate_and_flip_image(image)
        self.rois = []
        self.current_roi = []
        self.drawing = False

        pixmap = self.numpy_to_pixmap(self.image)
        self.pixmap = pixmap
        self.original_pixmap = pixmap.copy()  # This should be the clean untouched image

        # === GUI Layout ===
        layout = QVBoxLayout()

        # QLabel to display image
        self.image_label = QLabel()
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setFixedSize(self.pixmap.size())  # Fix label size to pixmap size
        self.image_label.setScaledContents(False)

        # Enable mouse events
        self.image_label.installEventFilter(self)
        self.image_label.setMouseTracking(True)

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
        segment_btn.clicked.connect(self.run_segmentation)

        finish_btn = QPushButton("Finish")
        finish_btn.clicked.connect(self.finish_and_close)

        button_layout.addWidget(draw_btn)
        button_layout.addWidget(delete_btn)
        button_layout.addWidget(segment_btn)
        button_layout.addWidget(finish_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def numpy_to_pixmap(self, array):
        """Convert a 2D numpy array to QPixmap for display."""
        norm = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
        img = norm.astype(np.uint8)
        qimage = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimage)

    def draw_rois(self):
        print("🖊️ Drawing mode ON")
        self.drawing = True
        self.current_roi = []

    def delete_rois(self):
        print("🗑️ Delete ROIs")
        self.rois = []
        self.update_image()

    def run_segmentation(self):
        print("🧠 Segmentation clicked (not implemented)")

    def finish_and_close(self):
        print("✅ Finished marking ROIs")
        self.finished.emit()
        self.close()

    def eventFilter(self, source, event):
        if source is self.image_label and self.drawing:
            if event.type() == event.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    pos = event.pos()
                    # Map label pos to pixmap coordinates
                    pixmap_size = self.image_label.pixmap().size()
                    label_size = self.image_label.size()

                    # Calculate scale factors
                    scale_x = pixmap_size.width() / label_size.width()
                    scale_y = pixmap_size.height() / label_size.height()

                    # Map pos to pixmap coords
                    mapped_x = int(pos.x() * scale_x)
                    mapped_y = int(pos.y() * scale_y)

                    self.current_roi.append(QPoint(mapped_x, mapped_y))
                    self.update_image()
                elif event.button() == Qt.RightButton:
                    if len(self.current_roi) > 2:
                        self.rois.append(self.current_roi)
                    self.current_roi = []
                    self.update_image()
            elif event.type() == event.MouseMove:
                # Optional for preview
                pass
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
        self.setWindowTitle("Dual 3D Matrix Viewer")

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

    # def update_matrix2(self, idx):
    #     self.slice2_idx = idx
    #     self.label2.setText(f"CSI Slice: {idx}")
    #     self.update_display()

    def update_display(self):
        self.axes.clear()
        self.axes.imshow(self.matrix1[:, :, self.slice1_idx], cmap='gray', aspect='equal')
        self.axes.set_title("MRI")
        self.axes.axis('off')
        self.canvas.draw()


class MRICSIViewer(QWidget):
    def __init__(self):
        super().__init__()
        
        self.initUI()
        self.mri = None
        self.csi = None
        # Define substances
        self.substances = ['HDO', 'Glucose', 'Lactate']
        self.slice_queue = []       # list of slice indices to show
        self.current_roi_index = 0  # where we are in the list

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

            plt.suptitle(f"{substance_name} - All Scans", fontsize=16)
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
        from reportlab.lib.utils import ImageReader
        import os

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
                img_height = 300

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

        # === draw_mri_grid ===
            
        anatomic_jpg_path = os.path.join(report_folder, "anatomic_slices.jpg")

        # Save anatomical image inside report folder
        slice_start = 7
        slice_end = self.mri.shape[2] - 5
        self.display_anatomic_images(self.mri, list(range(slice_start, slice_end)), save_as=anatomic_jpg_path)


        img = ImageReader(anatomic_jpg_path)
        img_width = 500
        img_height = 300
        x_pos = 50
        y_pos = height - 270  # top Y coordinate for image block

        # Center text above image
        img_center_x = x_pos + img_width / 2
        text = "Anatomical MRI Slices"
        font_name = "Helvetica"
        font_size = 12
        pdf.setFont(font_name, font_size)
        text_width = stringWidth(text, font_name, font_size)
        text_x = img_center_x - text_width / 2
        text_y = y_pos

        # Draw label above image
        pdf.drawString(text_x, text_y, text)

        # Draw the image just below the label
        pdf.drawImage(img, x_pos, text_y - img_height - 10, width=img_width, height=img_height)

        # Update current_y if you're tracking vertical position
        current_y = text_y - img_height - 30


        # Display CSI overlays on the selected anatomical slice for all scans
        # === Overlay each MRI/CSI pair ===
        overlay_img_height = 300
        overlay_img_width = 500
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

        self.roi_window = ROIMarker(slice_index, mri_slice)
        self.roi_window.finished.connect(self.on_roi_window_closed)
        self.roi_window.show()

    def on_roi_window_closed(self):
        self.current_roi_index += 1
        self.launch_next_roi_window()
    

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MRICSIViewer()
    # viewer.resize(1000, 800)
    viewer.show()
    sys.exit(app.exec_())