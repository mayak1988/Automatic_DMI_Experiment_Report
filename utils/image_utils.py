import numpy as np 

def map_mri_to_csi_pairs(selected_mri_slices, num_mri_slices, num_csi_slices):
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

def normalize_csi_slice(csi_slice):
    csi_slice_abs = np.abs(csi_slice)  # Take the magnitude of the complex CSI data
    min_val = np.min(csi_slice_abs)
    max_val = np.max(csi_slice_abs)
    return (csi_slice_abs - min_val) / (max_val - min_val)  # Normalize to [0, 1]

def rotate_and_flip_image(image):
    # Rotate the image 90 degrees
    rotated_image = np.rot90(image)
    # Flip the image along the Y-axis (vertical flip)
    flipped_image = np.flipud(rotated_image)
    return flipped_image

def normalize(img):
    img = img - np.min(img)
    if np.max(img) > 0:
        img = img / np.max(img)
    return img

def convert_roi_to_mask(rois, image_shape):
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