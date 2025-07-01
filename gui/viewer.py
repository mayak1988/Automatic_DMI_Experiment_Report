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
from scipy.ndimage import zoom
from gui.DualSliceViewer import DualSliceViewer
from gui.RoiMarker import ROIMarker
from config.config import SAM_CHECKPOINT_PATH,model


class MRICSIViewer(QWidget):
    def __init__(self):
        super().__init__()

        # Set a wider default size (e.g., 1400x900)
        self.resize(900, 500)  

        self.predictor = None 

        self.initUI()
        # Load SAM model
        # self.sam_checkpoint = r'C:\Users\luciogrp\Documents\GitHub\Automatic_DMI_Experiment_Report\data\sam_vit_b_01ec64.pth'
        self.sam_checkpoint = SAM_CHECKPOINT_PATH
        self.sam_model  = sam_model_registry[model](checkpoint=self.sam_checkpoint)
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
        """
        Open the DualSliceViewer window to explore MRI and CSI volumes.

        This initializes and displays a DualSliceViewer instance,
        passing the currently loaded MRI and CSI data volumes to it.
        """
        self.viewer = DualSliceViewer(self.mri, self.csi)
        self.viewer.show()

    def load_mat(self):
        """
        Open a .mat file and load MRI and CSI data into the application.

        - Opens a file dialog to select a .mat file.
        - Expects the .mat file to contain a 'res' structure with:
            - 'anaimage' under `res[0, 0]` for anatomical MRI data.
            - 'SepImage' under `res[0, 0]` for CSI data.
        - Applies `np.abs()` to convert from complex to magnitude (if needed).
        - Stores MRI and CSI as instance attributes: `self.mri` and `self.csi`.

        Prints:
        -------
        - Shapes of the loaded MRI and CSI arrays.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Open .mat File", "", "MAT-files (*.mat)")
        if not path: 
            return
        mat = scipy.io.loadmat(path)
        self.mri = np.abs(mat['res'][0, 0]['anaimage'])    # If entire array is complex
        self.csi = np.abs(mat['res'][0, 0]['SepImage'])
        print("Loaded MRI shape:", self.mri.shape)
        print("Loaded CSI shape:", self.csi.shape)

    def map_mri_to_csi_pairs(self,selected_mri_slices, num_mri_slices, num_csi_slices):
        """
        Maps selected MRI slice indices to corresponding CSI slice indices.

        This function computes a mapping from selected MRI slice indices to CSI slice indices
        based on the relative number of slices in each modality. If multiple MRI slices map
        to the same CSI slice, only the first one is kept in the mapping.

        Parameters:
        ----------
        selected_mri_slices : list[int]
            Indices of selected MRI slices.
        num_mri_slices : int
            Total number of MRI slices.
        num_csi_slices : int
            Total number of CSI slices.

        Returns:
        -------
        dict[int, int]
            A dictionary mapping CSI slice indices to the corresponding MRI slice indices.
        """
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
        """
        Normalizes a CSI slice to the range [0, 1] based on its magnitude.

        This function takes a complex-valued CSI (Chemical Shift Imaging) slice,
        computes its magnitude (absolute value), and normalizes the values to the [0, 1] range
        using min-max normalization.

        Parameters:
        ----------
        csi_slice : np.ndarray
            A 2D or 3D numpy array representing a single complex-valued CSI slice.

        Returns:
        -------
        np.ndarray
            A normalized array of the same shape with values scaled to the range [0, 1].
        """
        csi_slice_abs = np.abs(csi_slice)  # Take the magnitude of the complex CSI data
        min_val = np.min(csi_slice_abs)
        max_val = np.max(csi_slice_abs)
        return (csi_slice_abs - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    
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
    
    # Display the anatomical images for the selected slices only
    def display_anatomic_images(self,anatomic, selected_slices,save_as=None):
        """
        Displays multiple anatomical MRI slices in a grid, applying rotation and vertical flip to each slice.

        Parameters:
        -----------
        anatomic : np.ndarray
            3D numpy array representing anatomical MRI data with shape (height, width, num_slices).
        selected_slices : list of int
            List of slice indices to display.
        save_as : str or None, optional
            File path to save the resulting figure as a JPEG image. If None, the figure is not saved.

        Behavior:
        ---------
        - Rotates each selected slice 90 degrees and flips it vertically before display.
        - Arranges slices in a grid with 4 images per row.
        - Displays slice indices starting at 1 (slice_num + 1) in the subplot titles.
        - Turns off axis ticks for clarity.
        - Saves the figure as a high-quality JPEG if `save_as` is provided.
        - Shows the matplotlib figure window.

        Notes:
        ------
        - Uses `rotate_and_flip_image` function to preprocess each slice.
        - The number of rows is computed based on the number of selected slices.
        - Remaining subplot axes are turned off if the total axes exceed the number of slices.
        """

        
        num_images_per_row = 4
        num_rows = (len(selected_slices) + num_images_per_row - 1) // num_images_per_row  # Calculate rows based on selected slices
        
        fig, axes = plt.subplots(num_rows, num_images_per_row, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        for idx, slice_num in enumerate(selected_slices):
            ax = axes[idx]
            # Apply rotation and flip before displaying
            rotated_flipped_image = self.rotate_and_flip_image(anatomic[:, :, slice_num])
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
        """
        Overlays CSI metabolite intensity maps on anatomical MRI slices for multiple scans and substances.

        Parameters:
        -----------
        anatomic : np.ndarray
            3D numpy array representing anatomical MRI data (height x width x slices).
        csi : np.ndarray
            5D numpy array containing CSI data with shape 
            (height_csi, width_csi, csi_slices, substances, scans).
        anatomic_slice_idx : int
            Index of the anatomical MRI slice to display.
        csi_slice_idx : int
            Index of the CSI slice to overlay.
        substances : list of str
            List of substance names corresponding to the 4th dimension of `csi`.
        save_folder : str or None, optional
            Directory path where overlay images will be saved. If None, saves to current working directory.

        Behavior:
        ---------
        - For each substance:
        - Creates a figure with subplots arranged in a grid (4 columns).
        - For each scan:
            - Extracts the relevant CSI slice and normalizes it.
            - Resizes the CSI slice to match the anatomical image size using bilinear interpolation.
            - Overlays the normalized CSI data using a 'jet' colormap with transparency on the anatomical grayscale image.
            - Disables axis ticks for clean visualization.
        - Saves the overlay figure as a high-resolution JPEG image.
        - Closes the figure to free memory.

        Notes:
        ------
        - The anatomical slice is pre-processed with a rotation and vertical flip to match orientation.
        - Uses `normalize_csi_slice` for normalization and `zoom` from scipy.ndimage for resizing.
        - Overlay uses alpha blending to combine anatomical and CSI images visually.
        """
        num_scans = csi.shape[4]
        ncols = 4
        nrows = (num_scans + ncols - 1) // ncols

        # Prepare anatomical background image once
        anatomic_img = self.rotate_and_flip_image(anatomic[:, :, anatomic_slice_idx])

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
        """
        Plots and saves ROI intensity time series for each substance.

        Parameters:
        -----------
        roi_averages : dict
            Dictionary mapping each substance (str) to a list of ROI intensity arrays.
            Each ROI intensity array contains mean intensities over multiple scans/time points.
            Example: { 'Substance1': [roi1_values, roi2_values, ...], ... }
        substances : list of str
            List of substance names to plot.
        save_folder : str
            Directory path where plot images will be saved.

        Behavior:
        ---------
        - For each substance:
            - Creates a line plot showing intensity over scans/time for each ROI.
            - Uses distinct colors for each ROI.
            - Adds title, axis labels, legend, and grid.
            - Saves the plot as a JPEG image in `save_folder`.

        Notes:
        ------
        - Assumes all ROIs have the same number of scans/time points.
        - Uses a predefined `roi_colors` list for ROI color cycling.
        - The x-axis corresponds to scan indices starting from 0.
        """
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
                return date_str.replace("-", "_")  # Safe name

        # Invalid format fallback
        return "default_report"
    
    def convert_roi_to_mask(self, rois, image_shape):
        """
        Converts a list of ROIs into a binary mask.

        This function takes a list of ROIs (either boolean masks or polygons defined by points)
        and generates a single binary mask with all ROIs combined. Polygons are assumed to be lists
        of QPoint-like objects with `.x()` and `.y()` methods, and are filled using OpenCV.

        Parameters:
        ----------
        rois : list
            List of ROIs, where each ROI is either:
            - A boolean numpy array of shape `image_shape` representing a mask.
            - A list of point objects (e.g., QPoint) defining a polygon.
        image_shape : tuple
            Shape of the image (height, width) for the output mask.

        Returns:
        -------
        np.ndarray
            A binary mask (dtype=bool) with the same shape as `image_shape`,
            where pixels inside any ROI are `True`.
        """
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
        """
        Plots an MRI slice with overlaid ROI contours.

        Parameters:
        -----------
        mri_slice : np.ndarray
            2D numpy array representing a single MRI slice.
        masks : list of np.ndarray
            List of binary masks (2D arrays) corresponding to ROIs on the MRI slice.
        save_path : str or None, optional
            File path to save the figure. If None, the plot is displayed interactively.
        title : str or None, optional
            Title for the plot.

        Behavior:
        ---------
        - Rotates and flips the MRI slice for consistent display orientation.
        - For each ROI mask:
            - Checks if mask shape matches the MRI slice shape; skips if mismatched.
            - Finds contours at mask boundary using `skimage.measure.find_contours`.
            - Plots each contour as a colored line, cycling through `roi_colors`.
        - Adds the title if provided.
        - Removes axis ticks for a clean image.
        - Saves the figure to disk if `save_path` is provided; otherwise, shows the plot window.

        Notes:
        ------
        - Assumes `roi_colors` is a predefined list of colors.
        - Contour plotting uses matplotlib with linewidth=2 for visibility.
        """
        flipped_mri = self.rotate_and_flip_image(mri_slice)

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
        """
        Calculates mean ROI intensities over multiple scans for each substance.

        Parameters:
        -----------
        csi : np.ndarray
            5D CSI data array with shape (height, width, slices, substances, scans).
        rois_with_slices : list of tuples
            List of (slice_index, roi_mask) tuples where:
                - slice_index (int): index of the CSI slice corresponding to the ROI.
                - roi_mask (np.ndarray): 2D binary mask of the ROI (can be a different shape than CSI slice).
        substances : list of str
            List of substance names corresponding to the substances dimension in `csi`.
        num_scans : int
            Number of scans/time points in the CSI data.

        Returns:
        --------
        roi_averages : dict
            Dictionary mapping each substance to a list of ROI mean intensity arrays.
            Each list contains one array per ROI; each array holds mean intensities per scan.
            Format: {substance: [roi1_intensity_series, roi2_intensity_series, ...], ...}

        Behavior:
        ---------
        - For each substance and each ROI mask:
            - Resizes ROI mask to match CSI slice size if necessary using nearest-neighbor interpolation.
            - Extracts mean intensity values inside the ROI for each scan.
        - Returns all computed ROI intensity time series in a dictionary.

        Notes:
        ------
        - Assumes ROI masks correspond spatially to slices in CSI data but may have different resolutions.
        - Uses `cv2.resize` with INTER_NEAREST to preserve binary mask integrity.
        """
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
        """
        Generates a comprehensive PDF report for an MRI + DMI experiment.

        This method collects metadata and selected slices from the GUI input fields,
        processes and overlays MRI and CSI data, includes user-drawn ROI masks,
        calculates average ROI intensities for each substance, and compiles all of it
        into a structured multi-page PDF report saved in a dedicated folder.

        Functionality includes:
        ------------------------
        - Creating a timestamped report folder inside "reports/"
        - Validating and cleaning date input to use in file/folder naming
        - Mapping selected MRI slices to CSI slices depending on coil type
        - Displaying anatomical MRI slices and saving them as images
        - Overlaying CSI data on anatomical slices
        - Displaying and saving user-defined ROI masks over MRI slices
        - Calculating average intensity of substances in ROIs across CSI timepoints
        - Plotting substance intensity over time and embedding it into the PDF

        GUI Inputs:
        -----------
        - `self.date_input`: Date of the experiment (used for naming and display)
        - `self.participant_input`: Participant name
        - `self.description_input`: Description of the experiment
        - `self.mri_input`: List of MRI slice indices (as text)
        - `self.coil_type_combo`: Dropdown to select coil type ("surface coil" or others)
        - `self.substances`: List of substances to extract from CSI data
        - `self.mri`, `self.csi`: 4D/5D numpy arrays of MRI and CSI data
        - `self.saved_masks`: Dictionary of user-drawn ROI masks per MRI slice
        - `self.marked_slices`: List of MRI slices with ROI annotations

        Output:
        -------
        - A saved PDF file with the full report, located in:
        `reports/<YYYY_MM_DD>_report/<date>_report.pdf`
        - Additional saved images used in the report, stored in the same folder

        Notes:
        ------
        - If the date format is invalid, the folder is named "default_report"
        - A file dialog is used to confirm or change the PDF save path
        - All image and plot generation is handled before saving the final PDF

        Raises:
        -------
        - ValueError if an ROI or mask format is unrecognized
        """

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
        """
        Normalizes an image array to the range [0, 1].

        This function shifts the input array so that the minimum value becomes 0,
        then scales the values so that the maximum becomes 1. If the array has a 
        maximum of 0 after shifting (i.e., all values were originally the same), 
        it skips the division step to avoid division by zero.

        Parameters:
        ----------
        img : np.ndarray
            A numpy array representing the image data (can be 2D or 3D).

        Returns:
        -------
        np.ndarray
            The normalized image array with values in the range [0, 1].
        """
        img = img - np.min(img)
        if np.max(img) > 0:
            img = img / np.max(img)
        return img

    def Mark_Rois(self):
        """
        Initializes the ROI marking process by parsing user input slices and launching the first ROI window.

        This method:
        - Parses the slice indices input from the user (e.g., a text field).
        - Converts valid slice indices to integers and stores them in a queue.
        - Resets the current ROI index to start from the first slice.
        - Initiates the ROI marking GUI/window for the first slice by calling `launch_next_roi_window()`.

        Notes:
        ------
        - Assumes `self.mri_input` is a GUI text input widget containing slice numbers.
        - The `slice_queue` holds slices to be processed in order.
        - `launch_next_roi_window` should handle displaying the ROI marking interface for the current slice.
        """
        self.marked_slices = self.parse_input(self.mri_input.text())  # or wherever your input is
        self.slice_queue = [int(s) for s in self.marked_slices if str(s).isdigit()]
        self.current_roi_index = 0

        
        self.launch_next_roi_window()
    
    def start_marking_rois(self):
        """
        Begins the ROI marking process by retrieving selected MRI slices and launching the first ROI window.

        Actions:
        --------
        - Retrieves the list of MRI slices selected by the user.
        - Resets the current ROI index to zero.
        - Calls `launch_next_roi_window()` to start the ROI marking interface.

        Notes:
        ------
        - Assumes `get_selected_mri_slices()` returns a list of slice indices to process.
        - `launch_next_roi_window()` manages the display and interaction for marking ROIs on slices.
        """
        self.marked_slices = self.get_selected_mri_slices()  # however you get them
        self.current_roi_index = 0
        self.launch_next_roi_window()

    def launch_next_roi_window(self):
        """
        Launches the ROI marking window for the next MRI slice in the queue.

        Behavior:
        ---------
        - Checks if all slices in `self.marked_slices` have been processed.
        If yes, prints a completion message and returns.
        - Retrieves the current slice index from `self.marked_slices`.
        - Extracts the corresponding MRI slice data (assumed 3D array).
        - Creates an instance of the `ROIMarker` window for the current slice,
        passing the slice index, slice data, and SAM predictor.
        - Connects the `finished` signal of the ROI window to `on_roi_window_closed`
        to handle the window closing event.
        - Displays the ROI marking window to the user.

        Notes:
        ------
        - Assumes `self.mri` is a 3D NumPy array (height x width x slices).
        - The ROI window (`ROIMarker`) must support the `finished` signal.
        - This method manages sequential slice ROI marking through `current_roi_index`.
        """
        if self.current_roi_index >= len(self.marked_slices):
            print("✅ All ROIs marked.")
            return

        slice_index = self.marked_slices[self.current_roi_index]
        mri_slice = self.mri[:, :, slice_index]  # assuming self.mri is 3D

        self.roi_window = ROIMarker(slice_index, mri_slice, predictor=self.predictor)
        self.roi_window.finished.connect(self.on_roi_window_closed)
        self.roi_window.show()

    def on_roi_window_closed(self):
        """
        Handles the event when an ROI marking window is closed.

        This method:
        - Retrieves the slice index and any segmented masks from the just-closed ROI window.
        - Saves the masks to `self.saved_masks` under the corresponding slice index,
        appending if masks already exist for that slice.
        - Prints a confirmation message about the number of masks saved.
        - Increments the current ROI index to move to the next slice.
        - Launches the ROI marking window for the next slice, if any remain.

        Notes:
        ------
        - Assumes `self.roi_window.segmented_masks` is a dict keyed by slice indices,
        containing lists of mask arrays.
        - This function enables sequential processing of multiple slices for ROI marking.
        """
        slice_index = self.roi_window.slice_index
        masks = self.roi_window.segmented_masks.get(slice_index, [])

        if masks:
            if slice_index not in self.saved_masks:
                self.saved_masks[slice_index] = []
            self.saved_masks[slice_index].extend(masks)
            print(f"✅ Saved {len(masks)} masks for slice {slice_index}")

        self.current_roi_index += 1
        self.launch_next_roi_window()

        