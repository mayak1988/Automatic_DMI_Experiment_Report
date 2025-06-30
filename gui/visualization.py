from gui.image_utils import rotate_and_flip_image,normalize_csi_slice
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import zoom
from skimage import measure
import cv2

roi_colors = plt.cm.tab20.colors  # or tab10.colors for 10
# Display the anatomical images for the selected slices only
def display_anatomic_images(anatomic, selected_slices,save_as=None):
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
def overlay_csi_on_anatomic(anatomic, csi, anatomic_slice_idx, csi_slice_idx, substances,save_folder=None):

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
            csi_slice_norm = normalize_csi_slice(csi_slice)

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

def plot_roi_intensities(roi_averages, substances, save_folder):
    num_scans = len(next(iter(roi_averages.values()))[0])  # assume all ROIs have same scan count
    x = np.arange(num_scans)

    for substance in substances:
        fig, ax = plt.subplots(figsize=(8, 5))
        all_roi_series = roi_averages[substance]

        for idx, roi_values in enumerate(all_roi_series):
            color = roi_colors[idx % len(roi_colors)]
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

def plot_slice_with_rois( mri_slice, masks, save_path=None, title=None):
    # Flip the image for consistent display
    flipped_mri = rotate_and_flip_image(mri_slice)

    fig, ax = plt.subplots()
    ax.imshow(flipped_mri, cmap='gray')

    for idx, mask in enumerate(masks):
        if mask.shape != flipped_mri.shape:
            print(f"⚠️ Mask shape {mask.shape} doesn't match MRI shape {flipped_mri.shape}. Skipping.")
            continue

        contours = measure.find_contours(mask, level=0.5)
        color = roi_colors[idx % len(roi_colors)]

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

def calculate_roi_intensity(csi, rois_with_slices, substances, num_scans):
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

