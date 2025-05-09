##################################################
## VasoTracker 2 - Blood Vessel Diameter Measurement Software
##
## Author: Calum Wilson, Matthew D Lee, and Chris Osborne
## License: BSD 3-Clause License (See main file for details)
## Website: www.vasostracker.com
##
##################################################


from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import ndimage


# EDIT AT YOUR OWN RISK


def diff(sig, n):
    dx = 1 / n
    ddt = ndimage.gaussian_filter1d(sig, sigma=6, order=1, mode="nearest") / dx
    ddt = np.array(ddt)
    return ddt


def diff2(sig, n):
    dx = 1 / n
    ddt = (
        np.convolve(sig, [1, -1]) / dx
    )  # ndimage.gaussian_filter1d(sig, sigma=6, order=1, mode='nearest') / dx
    ddt = np.array(ddt)
    return ddt


def diff3(sig, n):
    dx = 1 / n
    ddt = (
        np.diff(sig) / dx
    )  # ndimage.gaussian_filter1d(sig, sigma=6, order=1, mode='nearest') / dx
    ddt = np.array(ddt)
    return ddt


# Peak finding function


def detect_peaks(
    x,
    mph=None,
    mpd=1,
    threshold=0,
    edge="rising",
    kpsh=False,
    valley=False,
    show=False,
    ax=None,
):
    """Detect peaks in data based on their amplitude and other features.
    Marcos Duarte, https://github.com/demotu/BMC
    CC-BY-4.0


    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype("float64")
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ["rising", "both"]:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ["falling", "both"]:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[
            np.in1d(
                ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True
            )
        ]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (
                    x[ind[i]] > x[ind] if kpsh else True
                )
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available.")
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, "b", lw=1)
        if ind.size:
            label = "valley" if valley else "peak"
            label = label + "s" if ind.size > 1 else label
            ax.plot(
                ind,
                x[ind],
                "+",
                mfc=None,
                mec="r",
                mew=2,
                ms=8,
                label="%d %s" % (ind.size, label),
            )
            ax.legend(loc="best", framealpha=0.5, numpoints=1)
        ax.set_xlim(-0.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel("Data #", fontsize=14)
        ax.set_ylabel("Amplitude", fontsize=14)
        mode = "Valley detection" if valley else "Peak detection"
        ax.set_title(
            "%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
            % (mode, str(mph), mpd, str(threshold), edge)
        )
        # plt.grid()
        plt.show()


class TimeIt:
    from datetime import datetime

    def __init__(self):
        self.name = None

    def __call__(self, name):
        self.name = name
        return self

    def __enter__(self):
        self.tic = self.datetime.now()
        return self

    def __exit__(self, name, *args, **kwargs):
        print(
            "process "
            + self.name
            + " runtime: {}".format(self.datetime.now() - self.tic)
        )  ##]]

@dataclass
class DdtResult:
    # int arrays
    outer_diam_pos: np.ndarray
    inner_diam_pos: np.ndarray
    # bool arrays
    od_outliers: np.ndarray
    id_outliers: np.ndarray
    # float arrays
    outer_diam: np.ndarray
    inner_diam: np.ndarray


def process_ddts2(
    ddts, thresh_factor, thresh, nx, scale, start_x, ID_mode, detection_mode
) -> DdtResult:
    outer_diameters1_pos = []  # array for diameter data
    outer_diameters2_pos = []
    inner_diameters1_pos = []  # array for diameter data
    inner_diameters2_pos = []
    ODS = []
    IDS = []
    scale = scale
    start_x = [int(x) for x in start_x]

    for j, ddt in enumerate(ddts):
        end_x = start_x[j] + len(ddts[0])

        # Get local extrema positions
        valley_indices = detect_peaks(ddt, mph=0.04, mpd=1, valley=True)
        peaks_indices = detect_peaks(ddt, mph=0.04, mpd=1, valley=False)
        # Get the local extrema values
        valleys = [ddt[indice] for indice in valley_indices]
        peaks = [ddt[indice] for indice in peaks_indices]

        try:
            if detection_mode == 0:
                if len(valleys) > 0:
                    arg1 = np.argmin(valleys)
                    OD1 = valley_indices[arg1]
                    OD1_ = OD1 + start_x[j]
                else:
                    OD1_ = 0  # Default value if no valley detected

                if len(peaks) > 0:
                    arg2 = np.argmax(peaks)
                    OD2 = peaks_indices[arg2]
                    OD2_ = OD2 + start_x[j]
                else:
                    OD2_ = nx  # Default value if no peak detected

            else:
                if len(peaks) > 0:
                    arg1 = np.argmax(peaks)
                    OD1 = peaks_indices[arg1]
                    OD1_ = OD1 + start_x[j]
                else:
                    OD1_ = 0  

                if len(valleys) > 0:
                    arg2 = np.argmin(valleys)
                    OD2 = valley_indices[arg2]
                    OD2_ = OD2 + start_x[j]
                else:
                    OD2_ = nx 

        except:
            print("we fucked it")
            OD1_ = 0
            OD2_ = nx

        # Inner diameter calculation
        try:
            if detection_mode == 0:
                test = [item for item in peaks_indices if item > OD1 and item < (OD1 + (OD2 - OD1) / 2)]
                ID1_ = test[0] + start_x[j] if test else np.NaN

                test2 = [item for item in valley_indices if item < OD2 and item > (OD1 + (OD2 - OD1) / 2)]
                ID2_ = test2[-1] + start_x[j] if test2 else np.NaN

            else:
                test = [item for item in valley_indices if item > OD1 and item < (OD1 + (OD2 - OD1) / 2)]
                ID1_ = test[0] + start_x[j] if test else np.NaN

                test2 = [item for item in peaks_indices if item < OD2 and item > (OD1 + (OD2 - OD1) / 2)]
                ID2_ = test2[-1] + start_x[j] if test2 else np.NaN

        except:
            ID1_ = 0
            ID2_ = 0

        OD = scale * (OD2_ - OD1_)
        ID = scale * (ID2_ - ID1_)

        if ID_mode == 0:
            ID1_ = np.NaN
            ID2_ = np.NaN
            ID = np.NaN

        outer_diameters1_pos.append(OD1_)
        outer_diameters2_pos.append(OD2_)
        inner_diameters1_pos.append(ID1_)
        inner_diameters2_pos.append(ID2_)
        ODS.append(OD)
        IDS.append(ID)

    ODS_zscore = is_outlier(np.asarray(ODS), thresh_factor)
    IDS_zscore = is_outlier(np.asarray(IDS), thresh_factor)

    # ODlist_inliers = [val for val, outlier in zip(ODlist, ODS_zscore) if not outlier]
    # IDlist_inliers = [val for val, outlier in zip(IDlist, IDS_zscore) if not outlier]

    # average_OD = np.mean(ODlist_inliers)
    # average_ID = np.mean(IDlist_inliers)

    outer_diam_pos = np.column_stack((outer_diameters1_pos, outer_diameters2_pos))
    inner_diam_pos = np.column_stack((inner_diameters1_pos, inner_diameters2_pos))

    return DdtResult(
        outer_diam_pos=outer_diam_pos,
        inner_diam_pos=inner_diam_pos,
        od_outliers=ODS_zscore,
        id_outliers=IDS_zscore,
        outer_diam=np.array(ODS),
        inner_diam=np.array(IDS),
    )




def process_ddts(
    ddts, thresh_factor, thresh, nx, scale, start_x, ID_mode, detection_mode, ultrasound_tracking
) -> DdtResult:
    outer_diameters1_pos = []  # array for diameter data
    outer_diameters2_pos = []
    inner_diameters1_pos = []  # array for diameter data
    inner_diameters2_pos = []
    ODS = []
    IDS = []
    scale = scale
    start_x = [int(x) for x in start_x]
    

    if ultrasound_tracking == 1:
        detection_algorithm = 2 # Normal
    elif detection_mode == 0:
        detection_algorithm = 0  # Fluorescence
    else:
        detection_algorithm = 1 # Ultrasound

    for j, ddt in enumerate(ddts):
        end_x = start_x[j] + len(ddts[0])

        # Get local extrema positions
        valley_indices = detect_peaks(ddt, mph=0.04, mpd=1, valley=True)
        peaks_indices = detect_peaks(ddt, mph=0.04, mpd=1, valley=False)
        # Get the local extrema values
        valleys = [ddt[indice] for indice in valley_indices]
        peaks = [ddt[indice] for indice in peaks_indices]
        try:
            # Get the value of the biggest nadir in the first half of the dataset
            if detection_algorithm == 0:

                

                args = [
                    i
                    for i, idx in enumerate(valley_indices)
                    if idx > thresh and idx < len(ddts[0]) / 2
                ]  # >170 to filter out tie #args = [i for i,idx in enumerate(valley_indices) if idx > thresh  and idx < (end_x)/2] # >170 to filter out tie

                length_to_add = len(
                    [i for i, idx in enumerate(valley_indices) if idx < thresh]
                )  # Add this to correct for above filter
                nadirs_firsthalf = [valleys[i] for i in args]
                arg1 = np.argmax(np.absolute(nadirs_firsthalf))
                arg1 = arg1 + length_to_add

                OD1 = valley_indices[arg1]
                OD1_ = valley_indices[arg1] + start_x[j]

                # Get the value of the biggest peak in the second half of the dataset

                args2 = [
                    i
                    for i, idx in enumerate(peaks_indices)
                    if idx > OD1 + (len(ddts[0]) - OD1) / 10
                ]  # if idx > (end_x)/2] #args2 = [i for i,idx in enumerate(peaks_indices) if idx > (end_x)/2]
                peaks_2ndhalf = [peaks[i] for i in args2]
                arg2 = np.argmax(np.absolute(peaks_2ndhalf))
                arg3 = np.where(peaks == peaks_2ndhalf[arg2])[0][0]

                OD2 = peaks_indices[arg3]
                OD2_ = peaks_indices[arg3] + start_x[j]

            elif detection_algorithm == 1:
                # Inverted dataset logic
                # Detecting the biggest peak (previously valley) in the first half of the dataset
                args = [
                    i
                    for i, idx in enumerate(peaks_indices)  # Changed from valley_indices to peaks_indices
                    if idx > thresh and idx < len(ddts[0]) / 2
                ]
                
                length_to_add = len(
                    [i for i, idx in enumerate(peaks_indices) if idx < thresh]  # Changed from valley_indices to peaks_indices
                )
                peaks_firsthalf = [peaks[i] for i in args]
                arg1 = np.argmax(np.absolute(peaks_firsthalf))
                arg1 += length_to_add

                OD1 = peaks_indices[arg1]  # Changed from valley_indices to peaks_indices
                OD1_ = peaks_indices[arg1] + start_x[j]  # Changed from valley_indices to peaks_indices

                # Detecting the biggest valley (previously peak) in the second half of the dataset
                args2 = [
                    i
                    for i, idx in enumerate(valley_indices)  # Changed from peaks_indices to valley_indices
                    if idx > OD1 + (len(ddts[0]) - OD1) / 10
                ]
                valleys_2ndhalf = [valleys[i] for i in args2]
                arg2 = np.argmax(np.absolute(valleys_2ndhalf))
                arg3 = np.where(valleys == valleys_2ndhalf[arg2])[0][0]

                OD2 = valley_indices[arg3]  # Changed from peaks_indices to valley_indices
                OD2_ = valley_indices[arg3] + start_x[j]  # Changed from peaks_indices to valley_indices

            # Algorithm 2: Modified for ultrasound tracking - Made consistent with algorithm 0
            elif detection_algorithm == 2:
                # Use the EXACT SAME CODE as algorithm 0 to ensure no pixel shifts
                # Get the value of the biggest nadir in the first half of the dataset
                args = [
                    i
                    for i, idx in enumerate(valley_indices)
                    if idx > thresh and idx < len(ddts[0]) / 2
                ]
                
                length_to_add = len(
                    [i for i, idx in enumerate(valley_indices) if idx < thresh]
                )
                
                nadirs_firsthalf = [valleys[i] for i in args]
                if nadirs_firsthalf:
                    arg1 = np.argmax(np.absolute(nadirs_firsthalf))
                    arg1 = arg1 + length_to_add

                    OD1 = valley_indices[arg1]
                    OD1_ = valley_indices[arg1] + start_x[j]

                    # Get the value of the biggest peak in the second half of the dataset
                    args2 = [
                        i
                        for i, idx in enumerate(peaks_indices)
                        if idx > OD1 + (len(ddts[0]) - OD1) / 10
                    ]
                    
                    if args2:
                        peaks_2ndhalf = [peaks[i] for i in args2]
                        arg2 = np.argmax(np.absolute(peaks_2ndhalf))
                        arg3 = np.where(peaks == peaks_2ndhalf[arg2])[0][0]

                        OD2 = peaks_indices[arg3]
                        OD2_ = peaks_indices[arg3] + start_x[j]
                        
                        print("Detected ultrasound artery - using standard detection")
                    else:
                        OD1_ = 0
                        OD2_ = nx
                else:
                    OD1_ = 0
                    OD2_ = nx

        



        except:
            OD1_ = 0
            OD2_ = nx

        


        # Inner diameter calculation
        try:
            if detection_algorithm == 0:
                # The first inner diameter point is the first big (or the biggest) positive peak after the initial negative peak
                # test = [item for item in peaks_indices if item > OD1_ and item < nx/2]
                test = [
                    item
                    for item in peaks_indices
                    if item > OD1 and item < (OD1 + (OD2 - OD1) / 2)
                ]  # nx/2]

                arg3 = 0  # This arg for the first!
                ID1_ = test[arg3] + start_x[j]

                # The second inner diameter point is the last big negative peak before the big positive
                # test2 = [item for item in valley_indices if item < OD2_ and item > nx/2]
                test2 = [
                    item
                    for item in valley_indices
                    if item < OD2 and item > (OD1 - (OD2 - OD1) / 2)
                ]  # nx/2]
                ID2_ = test2[-1] + start_x[j]

            elif detection_algorithm == 1:
                # Inverted data mode

                # First inner diameter point (ID1_)
                test = [
                    item
                    for item in valley_indices
                    if item > OD1 and item < (OD1 + (OD2 - OD1) / 2)
                ]

                ID1_ = test[0] + start_x[j]  # First significant valley after OD1


                # Second inner diameter point (ID2_)
                test2 = [
                    item
                    for item in peaks_indices
                    if item < OD2 and item > (OD1 + (OD2 - OD1) / 2)
                ]

                ID2_ = test2[-1] + start_x[j]  # Last significant peak before OD2

            elif detection_algorithm == 2:
                # In ultrasound mode, OD1 and OD2 already correspond to the inner diameter
                ID1 = OD1
                ID1_ = OD1_
                ID2 = OD2
                ID2_ = OD2_
                print("ID1, ID2: ",ID1, ID2)

        except:
            ID1_ = 0
            ID2_ = 0

            

        OD = scale * (OD2_ - OD1_)
        ID = scale * (ID2_ - ID1_)

        if ID_mode == 0:
            ID1_ = np.NaN
            ID2_ = np.NaN
            ID = np.NaN

        outer_diameters1_pos.append(
            OD1_,
        )  # Position 1 of outer diameter
        outer_diameters2_pos.append(
            OD2_,
        )  # Position 2 of outer diameter
        inner_diameters1_pos.append(
            ID1_,
        )  # Position 1 of inner diameter
        inner_diameters2_pos.append(
            ID2_,
        )  # Position 2 of inner diameter
        ODS.append(OD)
        IDS.append(ID)
    # ODlist = [el for el in ODS if el != 0]
    # IDlist = [el for el in IDS if el != 0]

    # OD = np.average(ODlist)
    # ID = np.average(IDlist)

    # STDEVOD = np.std(ODlist)
    # STDEVID = np.std(IDlist)

    # ODlist2 = [el for el in ODlist if (OD - STDEVOD) < el < (OD + STDEVOD)]
    # IDlist2 = [el for el in IDlist if (ID - STDEVID) < el < (ID + STDEVID)]

    # OD = np.average(ODlist2)
    # ID = np.average(IDlist2)

    # ODS_flag = [
    #     1 if (OD - 3 * STDEVOD) < el < (OD + 3 * STDEVOD) else 0
    #     for i, el in enumerate(ODS)
    # ]  # Flag indicating if OD measurements are good
    # IDS_flag = [
    #     1 if (ID - 3 * STDEVID) < el < (ID + 3 * STDEVID) else 0
    #     for i, el in enumerate(IDS)
    # ]  # Flag indicating if ID measurements are good

    ODS_zscore = is_outlier(np.asarray(ODS), thresh_factor)
    IDS_zscore = is_outlier(np.asarray(IDS), thresh_factor)

    # ODlist_inliers = [val for val, outlier in zip(ODlist, ODS_zscore) if not outlier]
    # IDlist_inliers = [val for val, outlier in zip(IDlist, IDS_zscore) if not outlier]

    # average_OD = np.mean(ODlist_inliers)
    # average_ID = np.mean(IDlist_inliers)

    outer_diam_pos = np.column_stack((outer_diameters1_pos, outer_diameters2_pos))
    inner_diam_pos = np.column_stack((inner_diameters1_pos, inner_diameters2_pos))

    return DdtResult(
        outer_diam_pos=outer_diam_pos,
        inner_diam_pos=inner_diam_pos,
        od_outliers=ODS_zscore,
        id_outliers=IDS_zscore,
        outer_diam=np.array(ODS),
        inner_diam=np.array(IDS),
    )



### Outlier function is from here:
### https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data


def is_outlier(points, thresh):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
