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


from gui.visualization import display_anatomic_images,overlay_csi_on_anatomic,plot_roi_intensities,plot_slice_with_rois,calculate_roi_intensity
from gui.roi_marker import ROIMarker
from gui.image_utils import map_mri_to_csi_pairs,normalize_csi_slice,rotate_and_flip_image,normalize,convert_roi_to_mask

def Generate_Report(self):


    reports_base_folder = "reports"
    os.makedirs(reports_base_folder, exist_ok=True)

    raw_date = self.date_input.text().strip()
    report_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_date)

    if not re.match(r'^\d{4}_\d{2}_\d{2}$', report_name):
        report_name = "default"

    report_folder = os.path.join(reports_base_folder, f"{report_name}_report")
    os.makedirs(report_folder, exist_ok=True)

    date = self.date_input.text()
    participant = self.participant_input.text()
    description = self.description_input.toPlainText()
    mrislices = self.parse_input(self.mri_input.text())
    selected_mri_slices = [int(s) for s in mrislices if str(s).strip().isdigit()]

    coil_type = self.coil_type_combo.currentText()
    if coil_type.lower() == "surface coil":
        csi_slices = [1] * len(selected_mri_slices)
        mri_slices = selected_mri_slices
    else:
        mri_slices, csi_slices = map_mri_to_csi_pairs(selected_mri_slices, self.mri.shape[2], self.csi.shape[2])

    if not mri_slices:
        print("No slices entered.")
        return

    default_name = f"{report_name}_report.pdf" if date else "experiment_report.pdf"
    default_path = os.path.join(report_folder, default_name)

    save_path, _ = QFileDialog.getSaveFileName(
        self,
        "Save Report PDF",
        default_path,
        "PDF files (*.pdf)"
    )
    if not save_path:
        return

    pdf = canvas.Canvas(save_path, pagesize=A4)
    width, height = A4
    pdf.setFont("Helvetica", 12)

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

    # Save anatomical images inside report folder
    slice_start = 7
    slice_end = self.mri.shape[2] - 5
    display_anatomic_images(self.mri, list(range(slice_start, slice_end)), save_as=os.path.join(report_folder, "anatomic_slices.jpg"))

    anatomic_jpg_path = os.path.join(report_folder, "anatomic_slices.jpg")
    anatomic_image = Image.open(anatomic_jpg_path)
    img_orig_width, img_orig_height = anatomic_image.size

    # Page size and margins
    page_width, page_height = height, width
    left_margin = 50
    right_margin = 50
    top_margin = 50
    bottom_margin = 50

    max_width = page_width - left_margin - right_margin
    max_height = page_height - top_margin - bottom_margin - 50

    scale_w = max_width / img_orig_width
    scale_h = max_height / img_orig_height
    scale = min(scale_w, scale_h, 1)

    img_width = img_orig_width * scale
    img_height = img_orig_height * scale

    img_x = (page_width - img_width) / 2
    text_y = page_height - top_margin
    img_y = text_y - img_height - 10

    if img_y < bottom_margin:
        pdf.showPage()
        pdf.setFont("Helvetica", 12)
        text_y = page_height - top_margin
        img_y = text_y - img_height - 10

    text = "Anatomical MRI Slices"
    font_name = "Helvetica"
    font_size = 12
    pdf.setFont(font_name, font_size)
    text_width = stringWidth(text, font_name, font_size)
    text_x = (page_width - text_width) / 2
    pdf.drawString(text_x, text_y, text)

    img = ImageReader(anatomic_jpg_path)
    pdf.drawImage(img, img_x, img_y, width=img_width, height=img_height)

    current_y = img_y - 30

    # Overlay CSI on anatomical slices
    current_y = insert_overlay_images_to_pdf(
        pdf=pdf,
        report_folder=report_folder,
        mri_slices=mri_slices,
        csi_slices=csi_slices,
        substances=self.substances,
        overlay_func=overlay_csi_on_anatomic,
        mri_data=self.mri,
        csi_data=self.csi,
        current_y=current_y,
        page_height=height
    )

    for slice_index, masks in self.saved_masks.items():
        mri_slice = self.mri[:, :, slice_index]
        save_path_mask = os.path.join(report_folder, f"mri_rois_slice_{slice_index}.jpg")

        binary_masks = []
        for roi in masks:
            mask = convert_roi_to_mask([roi], mri_slice.shape)
            binary_masks.append(mask)

        plot_slice_with_rois(
            mri_slice=mri_slice,
            masks=binary_masks,
            save_path=save_path_mask,
            title=f"MRI Slice {slice_index} with ROIs"
        )

        img = ImageReader(save_path_mask)
        img_width = 400
        img_height = 400
        if current_y - img_height < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            current_y = height - 100

        pdf.drawImage(img, 50, current_y - img_height, width=img_width, height=img_height)
        current_y -= img_height + 60

    all_rois_masks = []
    for mri_idx, csi_idx in zip(self.marked_slices, csi_slices):
        masks = self.saved_masks.get(mri_idx, [])
        mri_slice_shape = self.mri[:, :, mri_idx].shape

        for mask in masks:
            if isinstance(mask, (list, tuple)):
                np_mask = convert_roi_to_mask([mask], mri_slice_shape)
            elif isinstance(mask, np.ndarray):
                np_mask = mask
            else:
                raise ValueError(f"Unexpected mask format: {type(mask)}")

            all_rois_masks.append((csi_idx, np_mask))

    roi_averages = calculate_roi_intensity(self.csi, all_rois_masks, self.substances, self.csi.shape[4])

    plot_roi_intensities(roi_averages, self.substances, report_folder)

    for substance in self.substances:
        plot_img_path = os.path.join(report_folder, f"{substance}_intensity_plot.jpg")
        if os.path.exists(plot_img_path):
            img = ImageReader(plot_img_path)
            img_width = 300
            img_height = 225

            if current_y - img_height - 20 < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 12)
                current_y = height - 100

            pdf.drawString(50, current_y, f"{substance} ROI Intensity Over Time:")
            current_y -= 15

            pdf.drawImage(img, 50, current_y - img_height, width=img_width, height=img_height)
            current_y -= img_height + 40
        else:
            print(f"⚠️ Plot image not found: {plot_img_path}")

    pdf.save()
    print(f"Report saved to: {save_path}")

def insert_overlay_images_to_pdf(pdf, report_folder, mri_slices, csi_slices, substances,
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