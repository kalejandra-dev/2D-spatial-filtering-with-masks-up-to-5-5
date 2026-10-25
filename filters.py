# filters.py

import numpy as np
...

# ========== PART A: filter (just A edits) ==========
"""
PART A: filter nucleus (secuential version).

B reuses this functions inside image_filter (parallel version).

Operation: CORRELATION by default (is_conv=0). With is_conv=1 the masks
turns 180º and CONVOLUTION is obtained.
"""

import numpy as np

# ======================================================================
# 1. Mask validation
# ======================================================================
def validate_mask(mask):
    """Checks that the mask is 2D, with odd dimensions from 1 to 5."""
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(f"The mask must be 2D and has {mask.ndim} dimensions.")

    m, n = mask.shape
    for size, name in ((m, "rows"), (n, "columns")):
        if size % 2 == 0:
            raise ValueError(f"The mask has {size} {name}: it must be an odd number (1, 3 or 5).")
        if size > 5:
            raise ValueError(f"The mask has {size} {name}: the maximum is 5.")

    if not np.issubdtype(mask.dtype, np.number):
        raise ValueError("The mask must contain numbers.")


# ======================================================================
# 2. Preparetion of the mask: convolution and normalization
# ======================================================================
def prepare_mask(mask, is_conv):
    """Returns the mask ready to use, in float64.

    - is_conv == 1: turns the mask 180º (convolution). If it is 0, correlation.
    - If the sum of coefficients is not 0 nor 1, it is divided by the sum
      to preserve the average brightness (e.g. Gaussian with integers / 256).
      If the sum is 0 (Sobel, Laplacian), it is not touched.
    """
    validate_mask(mask)
    mask = np.asarray(mask, dtype=np.float64)

    if is_conv == 1:
        mask = mask[::-1, ::-1]     # turn of 180º: rows and columns reversed

    total = mask.sum()
    if not np.isclose(total, 0) and not np.isclose(total, 1):
        mask = mask / total

    return mask


# ======================================================================
# 3. Boundary replication (replicate padding)
# ======================================================================
def pad_replicate(img, a, b):
    """Adds a rows above/below and b columns to the left/right
    copying the pixel from the nearest edge.

    img has shape (M, N, C). Returns (M + 2a, N + 2b, C).

    Rule from the statement, applied to each coordinate separately:
      row < 0 -> 0 ; row > M-1 -> M-1 (same with columns).
    """
    M, N = img.shape[0], img.shape[1]

    # Original index that corresponds to each row/column of the expanded image.
    # E.g. M=4, a=2: arange(-2, 6) = [-2,-1,0,1,2,3,4,5] -> clip -> [0,0,0,1,2,3,3,3]
    row_idx = np.clip(np.arange(-a, M + a), 0, M - 1)
    col_idx = np.clip(np.arange(-b, N + b), 0, N - 1)

    return img[row_idx][:, col_idx]


# ======================================================================
# 4. The filter over a range of rows (what each process executes)
# ======================================================================
def filter_rows(padded, mask, out, row_start, row_end):
    """Calculates the sums S of the rows [row_start, row_end) of the original image
    and writes them in out (shape (M, N, C), float64).

    - padded: image with padding (M + 2a, N + 2b, C), in float64
    - mask:   mask already prepared (m, n), in float64
    - Input is not rounded nor saturated: that is done at the end with finalize().
    - Never writes to padded (the input), only to out.
    """
    m, n = mask.shape
    a, b = (m - 1) // 2, (n - 1) // 2
    N = out.shape[1]
    C = out.shape[2]

    for x in range(row_start, row_end):          # rows from the original image
        for y in range(N):                        # cols
            for c in range(C):                    # channels (R, G, B)
                acc = 0.0                         # acumulator in float: without overflow
                for i in range(-a, a + 1):        # vetical offsets from the mask
                    for j in range(-b, b + 1):    # vertical offsets from the mask
                        # The original pixel (x+i, y+j) is at (x+i+a, y+j+b)
                        # within the image with padding.
                        acc += mask[i + a, j + b] * padded[x + i + a, y + j + b, c]
                out[x, y, c] = acc


# ======================================================================
# 5. Post-procesado: saturation o rescaling
# ======================================================================
def finalize(S, norm):
    """Transforms the sums S (float64, (M, N, C)) into a uint8 image.

    - norm == 0: round and saturate  -> min(255, max(0, round(S)))
    - norm == 1: rescaling lineal min-max of each channel to 0..255
                 (for masks that sum to 0: Sobel, Laplacian).
                 If Smax == Smin, that channel will be all 0.
    This is done at the end because the rescaling needs the min and max
    of the entire image.
    """
    if norm == 0:
        rounded = np.floor(S + 0.5)               #rounded to the nearest integer (0.5 towards the upper)
        return np.clip(rounded, 0, 255).astype(np.uint8)

    out = np.zeros(S.shape, dtype=np.uint8)
    for c in range(S.shape[2]):
        s_min, s_max = S[:, :, c].min(), S[:, :, c].max()
        if s_max == s_min:
            continue                              # img without variation -> 0
        scaled = 255 * (S[:, :, c] - s_min) / (s_max - s_min)
        out[:, :, c] = np.floor(scaled + 0.5).astype(np.uint8)
    return out


# ======================================================================
# 6. Utilities of Image Format
# ======================================================================
def to_3d(image):
    """Greyscale (M, N) -> (M, N, 1). Color (M, N, C) it keeps the same.
    This ensures that all code works with 3D images."""
    image = np.asarray(image)
    if image.ndim == 2:
        return image[:, :, np.newaxis]
    return image


def from_3d(result, original_ndim):
    """Undo to_3d: if the original image was 2D, return 2D."""
    if original_ndim == 2:
        return result[:, :, 0]
    return result


# ======================================================================
# 7. Complete sequential version (reference for B and C)
# ======================================================================
def filter_sequential(image, mask, is_conv=0, norm=0):
    """Applies the filter to the entire image with a single process."""
    original_ndim = np.asarray(image).ndim
    img = to_3d(image).astype(np.float64)        # turn to wide type BEFORE operating

    mask = prepare_mask(mask, is_conv)
    a, b = (mask.shape[0] - 1) // 2, (mask.shape[1] - 1) // 2

    padded = pad_replicate(img, a, b)
    S = np.zeros(img.shape, dtype=np.float64)    # output img, separated from the input

    filter_rows(padded, mask, S, 0, img.shape[0])

    return from_3d(finalize(S, norm), original_ndim)

# ========== PARTE B: parallelization (just B edits) ==========

"""
PART B: parallel version of the filter.

image_filter splits the rows of the image into numprocs contiguous strips
(medium-grained parallelism) and each process of a Pool runs A's filter_rows
on its strip. Input and output live in shared memory, so the processes do
not receive copies of the image.

Note: every function the workers run is defined at module level in this file.
With the "spawn" start method (Windows, Jupyter) the workers import filters.py
and can only find top-level functions.
"""

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

try:
    import multiprocess as mp          # dill-based, works better inside notebooks
except ImportError:
    import multiprocessing as mp


# ======================================================================
# 8. Worker side (runs inside each Pool process)
# ======================================================================
# Each worker fills these once, in _init_worker, so the big arrays are not
# sent again with every task.
_shared_padded = None
_shared_out = None
_shared_mask = None


def _init_worker(shared_padded, padded_shape, shared_out, out_shape, mask):
    """Pool initializer: stores numpy views of the shared blocks.

    np.frombuffer does not copy: the views point to the same shared memory
    that the main process created, so writes to _shared_out are seen by it.
    """
    global _shared_padded, _shared_out, _shared_mask
    _shared_padded = np.frombuffer(shared_padded, dtype=np.float64).reshape(padded_shape)
    _shared_out = np.frombuffer(shared_out, dtype=np.float64).reshape(out_shape)
    _shared_mask = mask


def _filter_strip(strip):
    """Task of one worker: filter the rows [start, end) with A's filter_rows."""
    start, end = strip
    filter_rows(_shared_padded, _shared_mask, _shared_out, start, end)


# ======================================================================
# 9. Helpers of the main process
# ======================================================================
def _load_image(image):
    """Accepts a numpy array or a path (str / PathLike) to an image file."""
    if isinstance(image, (str, os.PathLike)):
        return np.array(Image.open(image))
    return np.asarray(image)


def _check_numprocs(numprocs):
    """numprocs must be an integer >= 1 (bool is not accepted)."""
    if isinstance(numprocs, bool) or not isinstance(numprocs, (int, np.integer)):
        raise ValueError(f"numprocs must be an integer, got {type(numprocs).__name__}.")
    if numprocs < 1:
        raise ValueError(f"numprocs must be >= 1, got {numprocs}.")


def _split_rows(M, numprocs):
    """Splits the rows 0..M-1 into numprocs contiguous strips [start, end).

    Strip sizes differ by at most one row. If numprocs > M some strips would
    be empty; those are dropped.
    """
    bounds = np.linspace(0, M, numprocs + 1).astype(int)
    return [(int(bounds[k]), int(bounds[k + 1]))
            for k in range(numprocs) if bounds[k + 1] > bounds[k]]


# ======================================================================
# 10. Parallel filter
# ======================================================================
def image_filter(image: np.ndarray,
                 filter: np.ndarray,
                 numprocs: int,
                 is_conv: int,
                 norm: int) -> np.ndarray:
    """Applies the 2D filter to the image using numprocs processes.

    image can be a numpy array (greyscale (M, N) or colour (M, N, C)) or a
    path to an image file. Returns a uint8 image with the same shape.
    """
    _check_numprocs(numprocs)
    image = _load_image(image)
    original_ndim = image.ndim

    # Same preparation as filter_sequential (Part A), in the main process.
    img = to_3d(image).astype(np.float64)        # wide type BEFORE operating
    mask = prepare_mask(filter, is_conv)         # validates, rotates, normalises
    a, b = (mask.shape[0] - 1) // 2, (mask.shape[1] - 1) // 2
    padded = pad_replicate(img, a, b)
    M, N, C = img.shape

    # Shared memory: padded input (read only) and output sums S (M, N, C).
    #
    # Why no lock is needed (and so RawArray and not Array):
    #   - the padded input is only read by the workers, never written;
    #   - the strips never overlap, so every output pixel has exactly one
    #     writer and two processes never write to the same position;
    #   - the min/max for norm=1 is computed in finalize, after all the
    #     workers have finished.
    # Array would add a lock that is never needed and only slows things down.
    shared_padded = mp.RawArray(ctypes.c_double, padded.size)
    np.frombuffer(shared_padded, dtype=np.float64).reshape(padded.shape)[:] = padded
    shared_out = mp.RawArray(ctypes.c_double, M * N * C)

    # Medium-grained parallelism: one contiguous strip of rows per process.
    strips = _split_rows(M, numprocs)

    with mp.Pool(processes=numprocs,
                 initializer=_init_worker,
                 initargs=(shared_padded, padded.shape, shared_out, img.shape, mask)) as pool:
        pool.map(_filter_strip, strips)

    # Copy S out of shared memory before it is released.
    S = np.frombuffer(shared_out, dtype=np.float64).reshape(M, N, C).copy()

    # finalize runs here, in the main process: the min-max rescaling (norm=1)
    # needs the min and max of the WHOLE image, which no single worker has.
    return from_3d(finalize(S, norm), original_ndim)


# ======================================================================
# 11. Two filters at the same time
# ======================================================================
def parallel_filtering(image: np.ndarray,
                       filter1: np.ndarray,
                       filter2: np.ndarray,
                       numprocs: int,
                       is_conv: int,
                       norm: int):
    """Applies filter1 and filter2 to the same image at the same time.

    Each filter is an image_filter call with numprocs processes, so up to
    2*numprocs processes work at once. Returns (result1, result2).
    """
    _check_numprocs(numprocs)
    image = _load_image(image)                   # read the file only once

    # Why threads and not processes: the Pool workers are daemonic and a
    # daemonic process cannot create its own Pool, so image_filter cannot run
    # inside a Pool worker. Each thread only starts its Pool and waits for it;
    # the filtering itself runs in the worker processes, so the GIL is not a
    # problem.
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(image_filter, image, filter1, numprocs, is_conv, norm)
        future2 = executor.submit(image_filter, image, filter2, numprocs, is_conv, norm)
        return future1.result(), future2.result()
