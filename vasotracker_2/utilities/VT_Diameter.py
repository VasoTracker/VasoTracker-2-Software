##################################################
## VasoTracker 2 - Blood Vessel Diameter Measurement Software
##
## Author: Calum Wilson, Matthew D Lee, and Chris Osborne
## License: BSD 3-Clause License (See main file for details)
## Website: www.vasostracker.com
##
##################################################


## We found the following to be useful:
## https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch09s07.html
## http://code.activestate.com/recipes/82965-threads-tkinter-and-asynchronous-io/
## https://www.physics.utoronto.ca/~phy326/python/Live_Plot.py
## http://forum.arduino.cc/index.php?topic=225329.msg1810764#msg1810764
## https://stackoverflow.com/questions/9917280/using-draw-in-pil-tkinter
## https://stackoverflow.com/questions/37334106/opening-image-on-canvas-cropping-the-image-and-update-the-canvas

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional
import numpy as np
from skimage import measure
from .VTutils import diff2, process_ddts
from scipy.signal import medfilt

if TYPE_CHECKING:
    from vt_mvc import Roi, Caliper, RasterDrawState

def _line_profile_coordinates(src, dst, linewidth=1):
    """Return the coordinates of the profile of an image along a scan line.
    From skimage, under BSD 3-clause

    Parameters
    ----------
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line

    Returns
    -------
    coords : array, shape (2, N, C), float
        The coordinates of the profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.

    Notes
    -----
    This is a utility method meant to be used internally by skimage functions.
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = int(np.ceil(np.hypot(d_row, d_col) + 1))
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.stack([np.linspace(row_i - row_width, row_i + row_width,
                                      linewidth) for row_i in line_row])
    perp_cols = np.stack([np.linspace(col_i - col_width, col_i + col_width,
                                      linewidth) for col_i in line_col])
    return np.stack([perp_rows, perp_cols])

@dataclass
class ImageDiameters:
    # array (num_lines, 2)
    outer_diam_x: np.ndarray
    inner_diam_x: np.ndarray
    outer_diam_y: np.ndarray
    inner_diam_y: np.ndarray
    # bool array
    od_outliers: np.ndarray
    id_outliers: np.ndarray
    # float array
    outer_diam: np.ndarray
    inner_diam: np.ndarray
    avg_outer_diam: float
    avg_inner_diam: float


def calculate_diameter(
    image: np.ndarray,
    rds: "RasterDrawState",
    compute_id: bool, # id is checked
    default_detection_alg: bool, # org is checked
    lines_to_avg: int,
    num_lines: int,
    scale: float,
    smooth_factor: int,
    thresh_factor: float,
    filter_means: bool,
    rotate_tracking: bool,
    ultrasound_tracking: bool,
) -> Optional[ImageDiameters]:
     # Rotate the image by 90 degrees if rotate_tracking is True

    if rotate_tracking:
        image = np.rot90(image)  # This rotates the image 90 degrees counterclockwise
        nx, ny = image.shape  # Update the dimensions after rotation

    else:
        ny, nx = image.shape
    
    roi = rds.roi
    autocaliper = rds.autocaliper
    multi_roi = rds.multi_roi

    y_pos = []
    have_autocalipers = len(autocaliper) > 0
    single_roi = len(multi_roi) == 0


    if not have_autocalipers and single_roi:
        if roi is None:
            if rotate_tracking:
                # Transform full image selection to match rotated analysis space
                start_x, start_y, end_x, end_y = 0, 0, ny, nx  # Swap width/height
            else:
                start_x, start_y, end_x, end_y = 0, 0, nx, ny  # No rotation needed
            
        else:
            start_x, start_y, end_x, end_y = roi.fixed_corners()

            if rotate_tracking:
                # The user drew the ROI on the original image, but we analyze the rotated image
                # Transform ROI coordinates while ensuring they remain within bounds
                start_x_new = start_y
                start_y_new = nx - end_x  # Use nx instead of ny to ensure correct bounds
                end_x_new = end_y
                end_y_new = nx - start_x  # Use nx instead of ny to ensure correct bounds


                start_x, start_y, end_x, end_y = start_x_new, start_y_new, end_x_new, end_y_new

        # Ensure scanlines are spaced evenly along the y-axis in the rotated image
        total_height = end_y - start_y
        space_between_lines = total_height / (num_lines + 1)

        start = int(start_y + space_between_lines)  # Always space along y-axis
        diff = int(total_height / (num_lines + 1))
        end = int(end_y - space_between_lines)

        # Ensure correct number of lines
        if total_height % (num_lines + 1) == 0:
            end += 1

        data = [
            np.average(
                image[
                    y - int(lines_to_avg // 2): y + int(lines_to_avg / 2),
                    int(start_x): int(end_x)  # Keep X as the width direction
                ],
                axis=0
            )
            for y in range(start, end, diff)
        ]

        for y in range(start, end, diff):
            y_pos.append((y, y))

        start_x = [start_x] * len(data)  # Ensure tracking alignment



    # For multiple ROIs when rotate_tracking is True
    elif not have_autocalipers and not single_roi:
        data = []
        start_x = []
        for roi in multi_roi.values():
            x1, y1, x2, y2 = roi.fixed_corners()
            
            if rotate_tracking:
                # Transform ROI coordinates for rotated image
                x1_new = y1
                y1_new = nx - x2
                x2_new = y2
                y2_new = nx - x1
                x1, y1, x2, y2 = x1_new, y1_new, x2_new, y2_new
            
            scan = np.average(
                image[
                    int(y1) : int(y2),
                    int(x1) : int(x2),
                ],
                axis=0,
            )
            data.append(scan)
            start_x.append(x1)
            y_mean = 0.5 * (y1 + y2)
            y_pos.append((y_mean, y_mean))

        diff = 0

    elif have_autocalipers:
        data = []
        start_x = []
        for cal in autocaliper.values():
            data.append(
                measure.profile_line(
                    image, (cal.y1, cal.x1), (cal.y2, cal.x2), linewidth=lines_to_avg
                )
            )
            start_x.append(cal.x1)

        diff = 0
    else:
        return None

    # Smooth the data
    window = np.ones(smooth_factor)
    if ultrasound_tracking == 0:
        smoothed = [
            np.convolve(window / window.sum(), sig, mode="same") for sig in data
        ]
    else:
        # Define the median filter window size
        median_window = smooth_factor if smooth_factor % 2 == 1 else smooth_factor + 1  # Must be odd
        # Apply median filtering instead of moving average smoothing
        smoothed = [medfilt(sig, kernel_size=median_window) for sig in data]


    # Differentiate the data. There are other methods in VTutils...
    # But this one is much faster!
    ddts = [diff2(sig, 1) for sig in smoothed]  # Was 1 \\\\\ ULTRASOUND
    window = np.ones(smooth_factor)
    ddts = [np.convolve(window / window.sum(), sig, mode="same") for sig in ddts]

    window = np.ones(smooth_factor)
    if ultrasound_tracking == 0:
        ddts = [np.convolve(window / window.sum(), sig, mode="same") for sig in ddts]
    else:
        # Define the median filter window size
        median_window = smooth_factor if smooth_factor % 2 == 1 else smooth_factor + 1  # Must be odd
        # Apply median filtering instead of moving average smoothing
        ddts = [medfilt(sig, kernel_size=median_window) for sig in ddts]


    thresh = 0
    diams = process_ddts(
        ddts,
        thresh_factor,
        thresh,
        nx,
        scale,
        start_x,
        compute_id,
        default_detection_alg,
        ultrasound_tracking,
    )
    if diams.outer_diam_pos.ndim == 0:
        return None

    if have_autocalipers:
        od_x = []
        od_y = []
        id_x = []
        id_y = []
        for i, cal in enumerate(autocaliper.values()):
            coords = _line_profile_coordinates(
                (cal.y1, cal.x1), (cal.y2, cal.x2)
            ).squeeze()

            def convert_from_lp_coords(pos, xlist, ylist):
                if np.any(pos == 0):
                    xlist.append((0, 0))
                    ylist.append((0, 0))
                    return

                d_x1 = coords[1][pos[0] - start_x[i]]
                d_x2 = coords[1][pos[1] - start_x[i]]
                d_y1 = coords[0][pos[0] - start_x[i]]
                d_y2 = coords[0][pos[1] - start_x[i]]
                xlist.append((d_x1, d_x2))
                ylist.append((d_y1, d_y2))

            try:
                od_pos = diams.outer_diam_pos[i]
                convert_from_lp_coords(od_pos, od_x, od_y)
                id_pos = diams.inner_diam_pos[i]
                convert_from_lp_coords(id_pos, id_x, id_y)
            except IndexError:
                breakpoint()
        od_x = np.array(od_x)
        od_y = np.array(od_y)
        id_x = np.array(id_x)
        id_y = np.array(id_y)
    else:
        od_x = diams.outer_diam_pos
        id_x = diams.inner_diam_pos
        od_y = np.array(y_pos)
        id_y = np.array(y_pos)

    if filter_means:
        avg_outer_diam=np.mean(diams.outer_diam, where=~diams.od_outliers)
        avg_inner_diam=np.mean(diams.inner_diam, where=~diams.id_outliers)
    else:
        avg_outer_diam=np.mean(diams.outer_diam)
        avg_inner_diam=np.mean(diams.inner_diam)


    result = ImageDiameters(
        outer_diam_x=od_x,
        outer_diam_y=od_y,
        inner_diam_x=id_x,
        inner_diam_y=id_y,
        od_outliers=diams.od_outliers,
        id_outliers=diams.id_outliers,
        outer_diam=diams.outer_diam,
        inner_diam=diams.inner_diam,
        avg_outer_diam=avg_outer_diam,
        avg_inner_diam=avg_inner_diam,
    )
    return result
