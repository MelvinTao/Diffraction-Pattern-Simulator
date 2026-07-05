"""Convert a DXF mask drawing to a binary black/white PNG.

Black = solid material (blocks light), white = through holes (light passes).
Everything outside the outermost boundary is painted black as well.

Regions are classified by nesting depth (even-odd scanline fill):
depth 0 (outside the part) -> black, depth 1 (material) -> black,
depth 2 (holes) -> white, depth 3 (islands inside holes) -> black, ...
"""

import numpy as np
import ezdxf
from ezdxf import path as ezpath
from PIL import Image

dxfFileName = "JWST_Full.dxf"           ### Only edit this part
imageSize = 2000                        # output image is imageSize x imageSize pixels
marginFrac = 0.02                       # extra black border around the geometry


dxfFilePath = f"DXF_Mask/{dxfFileName}"
pngFilePath = f"Test_Images/{dxfFileName.replace('.dxf', '.png')}"


def collect_segments(msp, flatten_dist):
    """Flatten every curve entity in modelspace into straight line segments."""
    segments = []
    for entity in msp:
        try:
            p = ezpath.make_path(entity)
        except (TypeError, ValueError):
            continue  # entity type has no path representation (text, points, ...)
        pts = [(v.x, v.y) for v in p.flattening(flatten_dist)]
        segments.extend((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                        for i in range(len(pts) - 1))
    if not segments:
        raise ValueError("No drawable geometry found in the DXF modelspace.")
    return np.asarray(segments, dtype=float)


def rasterize(segments, size, margin_frac):
    """Even-odd scanline fill: returns a uint8 image (0 = black, 255 = white)."""
    xmin, ymin = segments[:, [0, 2]].min(), segments[:, [1, 3]].min()
    xmax, ymax = segments[:, [0, 2]].max(), segments[:, [1, 3]].max()
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2 * (1 + margin_frac)
    step = 2 * half / size

    # pixel-center sample coordinates (row 0 = top of image)
    px = cx - half + (np.arange(size) + 0.5) * step
    py = cy + half - (np.arange(size) + 0.5) * step

    x0, y0, x1, y1 = segments.T
    # orient every segment so y0 < y1; drop horizontal segments
    flip = y0 > y1
    x0[flip], x1[flip] = x1[flip], x0[flip].copy()
    y0[flip], y1[flip] = y1[flip], y0[flip].copy()
    keep = y1 > y0
    x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]

    row_crossings = [[] for _ in range(size)]
    for sx0, sy0, sx1, sy1 in zip(x0, y0, x1, y1):
        # rows whose sample y lies in [sy0, sy1)  (half-open: vertices count once)
        lo = np.searchsorted(-py, -sy1, side="right")   # first row with py < sy1
        hi = np.searchsorted(-py, -sy0, side="right")   # first row with py < sy0
        if lo >= hi:
            continue
        t = (py[lo:hi] - sy0) / (sy1 - sy0)
        xs = sx0 + t * (sx1 - sx0)
        for r, xc in zip(range(lo, hi), xs):
            row_crossings[r].append(xc)

    # Per row, the open (non-material) spans lie between even/odd crossing
    # pairs: (-inf, x1), (x2, x3), ..., (xn, +inf).  Open spans form either
    # through holes (white) or the outside region (black).  Group spans into
    # connected regions with a union-find; any region reaching +-inf touches
    # the image border and is therefore outside.
    parent = [0]  # node 0 = the outside region
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    rows_spans = []          # per row: list of (x_start, x_end, node_id)
    prev = []
    for xs in row_crossings:
        xs = np.sort(xs)
        bounds = np.concatenate([[-np.inf], xs, [np.inf]])
        spans = []
        for a, b in zip(bounds[0::2], bounds[1::2]):
            parent.append(len(parent))
            node = parent[-1]
            if np.isinf(a) or np.isinf(b):
                union(node, 0)
            for pa, pb, pnode in prev:      # connect to overlapping span above
                if a < pb and pa < b:
                    union(node, pnode)
            spans.append((a, b, node))
        rows_spans.append(spans)
        prev = spans

    img = np.zeros((size, size), dtype=np.uint8)
    outside = find(0)
    for r, spans in enumerate(rows_spans):
        for a, b, node in spans:
            if find(node) != outside:       # enclosed open span = through hole
                img[r, np.searchsorted(px, a, "right"):
                       np.searchsorted(px, b, "left")] = 255
    return img


def dxf_to_png(dxf_path, png_path):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # coarse pass to measure the drawing, fine pass at sub-pixel accuracy
    coarse = collect_segments(msp, flatten_dist=1.0)
    extent = max(np.ptp(coarse[:, [0, 2]]), np.ptp(coarse[:, [1, 3]]))
    segments = collect_segments(msp, flatten_dist=extent / imageSize / 8)

    img = rasterize(segments, imageSize, marginFrac)
    Image.fromarray(img, mode="L").save(png_path)
    open_frac = (img == 255).mean()
    print(f"Saved {png_path} ({imageSize}x{imageSize}, {open_frac:.1%} open area)")


if __name__ == "__main__":
    dxf_to_png(dxfFilePath, pngFilePath)
