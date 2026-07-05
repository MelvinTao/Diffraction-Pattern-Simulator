"""Animate the diffraction pattern of an aperture mask through a focus sweep.

Same simulation as test.py, but instead of a few static panels it renders one
animation sweeping the sensor from START_DEFOCUS to END_DEFOCUS waves of
defocus in SLICES steps, spread evenly over LENGTH_SECONDS. The frame rate is
therefore SLICES / LENGTH_SECONDS.

Edit the settings below, then run:  python focus_sweep.py
"""

import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# ------------------------- settings — edit these -------------------------
IMAGE = 'Test_Images/JWST_Full.png'  # aperture mask (png), brightness = transmission
START_DEFOCUS = -5.0                 # starting defocus, in waves of wavefront error at the pupil edge
END_DEFOCUS = 5.0                    # ending defocus, in waves
SLICES = 100                          # number of focus slices (= animation frames)
LENGTH_SECONDS = 6.0                 # total animation length in seconds
OUTPUT = None                        # output file, or None for Diffraction_Gifs/<image name>_sweep.gif
# --------------------------------------------------------------------------


def main():
    # Load the aperture mask as grayscale — pixel brightness stands in for transmission.
    img = cv2.imread(IMAGE, 0)
    assert img is not None, "check the file path"
    aperture = img.astype(float) / 255.0            # transmission 0..1

    # Normalized pupil coordinates, rho=1 at the image edge, for the defocus phase term.
    # float32 halves the cost of the per-frame complex exponential below.
    rows, cols = aperture.shape
    yy, xx = np.indices((rows, cols), dtype=np.float32)
    yy = (yy - rows / 2) / (rows / 2)
    xx = (xx - cols / 2) / (cols / 2)
    rho2 = xx**2 + yy**2
    aperture32 = aperture.astype(np.float32)

    # test.py computes the pattern as a 4x zero-padded FFT and then keeps only the
    # 600x600 center — i.e. it computes ~33M frequency samples and throws away 99%
    # of them. Here we instead evaluate just the frequencies we keep, as a direct
    # partial DFT: F = Wy @ pupil @ Wx. That's two BLAS matrix multiplies, ~20x
    # faster than the full FFT, and exactly equal to the fftshift-crop result
    # (to single precision, far below what's visible after the gamma stretch).
    crop_half, pad = 300, 4
    ky = np.arange(-crop_half, crop_half)          # frequency bins of the padded grid we keep
    Wy = np.exp(-2j * np.pi * np.outer(ky, np.arange(rows)) / (pad * rows)).astype(np.complex64)
    Wx = np.exp(-2j * np.pi * np.outer(np.arange(cols), ky) / (pad * cols)).astype(np.complex64)

    def diffraction_pattern(defocus_waves):
        """Fraunhofer intensity for a given defocus (see test.py for details)."""
        phase = np.exp(1j * 2 * np.float32(np.pi) * np.float32(defocus_waves) * rho2)
        pupil = aperture32 * phase
        F = Wy @ pupil @ Wx
        I = np.abs(F)**2
        I /= I.max()
        return I

    # Precompute every frame up front — each slice is a full padded FFT, which is far
    # too slow to do live inside the animation callback at the requested frame rate.
    defocus_values = np.linspace(START_DEFOCUS, END_DEFOCUS, SLICES)
    print(f"Computing {SLICES} slices from {START_DEFOCUS:+g} to {END_DEFOCUS:+g} waves...")
    frames = []
    for i, d in enumerate(defocus_values):
        # Gamma stretch, same as test.py, so faint spikes stay visible next to the core.
        frames.append(diffraction_pattern(d)**0.25)
        print(f"\r  slice {i + 1}/{SLICES}", end='', flush=True)
    print()

    # The in-focus pattern (zero defocus) as a static reference panel.
    in_focus = diffraction_pattern(0.0)**0.25

    # Frame rate comes entirely from the user's slices / length choice.
    fps = SLICES / LENGTH_SECONDS
    interval_ms = 1000.0 / fps

    # Three panels: the static input mask on the left, the animated pattern in the
    # middle, and the static in-focus pattern on the right for reference.
    # The suptitle shows the sweep range; the pattern title carries the per-frame readout.
    fig, (ax_mask, ax_pattern, ax_focus) = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(f"Focus sweep: {START_DEFOCUS:+g} → {END_DEFOCUS:+g} waves")
    ax_mask.imshow(aperture, cmap='gray')
    ax_mask.set_title(os.path.basename(IMAGE))
    ax_mask.axis('off')
    im = ax_pattern.imshow(frames[0], cmap='gray', vmin=0, vmax=1)
    title = ax_pattern.set_title(f"Defocus: {defocus_values[0]:+.2f} waves (frame 1/{SLICES})")
    ax_pattern.axis('off')
    ax_focus.imshow(in_focus, cmap='gray', vmin=0, vmax=1)
    ax_focus.set_title("In focus (0 waves)")
    ax_focus.axis('off')

    def update(i):
        im.set_data(frames[i])
        title.set_text(f"Defocus: {defocus_values[i]:+.2f} waves (frame {i + 1}/{SLICES})")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=SLICES,
                                   interval=interval_ms, blit=False)

    output = OUTPUT or os.path.join(
        'Diffraction_Gifs', os.path.splitext(os.path.basename(IMAGE))[0] + '_sweep.gif')
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    print(f"Saving {output} at {fps:.1f} fps...")
    anim.save(output, writer=animation.PillowWriter(fps=fps))
    print("Done.")
    plt.show()


if __name__ == '__main__':
    main()
