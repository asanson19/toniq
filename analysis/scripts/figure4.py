"""Make Figure 4 for paper.

"""
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sigpy as sp
import yaml

from os import path, makedirs

from toniq import sr
from toniq.config import parse_slice, load_volume
from toniq.plot_params import *
from toniq.plot import remove_ticks, color_panels, label_panels
from toniq.util import normalize

kwargs = {'vmin': 0, 'vmax': 1, 'cmap': CMAP['image']}

def plot_inputs(fig, target, reference, inset):
    axes = fig.subplots(2, 1, gridspec_kw={'left': 0.05, 'right': 0.95, 'bottom': 0.05, 'top': 0.92})
    axes[0].imshow(reference, **kwargs)
    axes[1].imshow(target, **kwargs)
    axes[0].set_title('Reference')
    axes[1].set_title('Target')
    remove_ticks(axes)
    inset_start = (inset[1].start, inset[0].start)
    inset_size = (inset[1].stop - inset[1].start, inset[0].stop - inset[0].start)
    for ax in axes.flat:
        rect = patches.Rectangle(inset_start, *inset_size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

def plot_model(fig, target, reference, psf, cell_size):
    axes = fig.subplots(2, 3, gridspec_kw={'left': 0.05, 'right': 0.95, 'bottom': 0.05, 'top': 0.92})
    titles = ('Target Patch', 'Reference Patch', 'Local PSF')
    images = (target, reference, psf)
    for i in range(3):
        axes[0, i].imshow(images[i], **kwargs)
        axes[0, i].set_title(titles[i])
        kspace = np.abs(sp.fft(sp.resize(images[i], (cell_size, cell_size))))
        kspace = np.log(kspace + 1) * 2
        axes[1, i].imshow(kspace, **kwargs)
    axes[1, 0].set_ylabel('2D DFT')
    symbols = ('=', r'$\circledast$', '=', r'$\odot$')
    # symbols = ('=', r'$*$', '=', r'$\odot$')
    for ax, symbol in zip(axes[:, :2].flat, symbols):
        ax.text(1.1, 0.49, symbol, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, size=20)
    remove_ticks(axes)

p = argparse.ArgumentParser(description='Make figure 4')
p.add_argument('save_dir', type=str, help='path where figure is saved')
p.add_argument('-c', '--config', type=str, default='fse125.yml', help='data config file')
p.add_argument('-x', type=int, default=105, help='left/right coordinate of inset location; default=105')
p.add_argument('-y', type=int, default=88, help='up/down coordinate of inset location; default=88')
p.add_argument('-z', type=int, default=18, help='relative position of FWHM z slice (after crop); default=18')
p.add_argument('--psf_window_size', type=int,  nargs=3, default=[14, 14, 4], help='psf window size in pixels; default=10')
p.add_argument('--unit_cell_size', type=int,  nargs=3, default=[10, 10, 8], help='unit cell size in pixels; default=10')
p.add_argument('-p', '--plot', action='store_true', help='show plots')
p.add_argument('-s', '--sigma', type=float, nargs='+', default=[1, 0.5], help='sigmas for gaussian PSF')
p.add_argument('--psf_size', type=int, default=5, help='psf size (std) in pixels')
p.add_argument('--psf_radius', type=int, default=20, help='psf extent in pixels')

if __name__ == '__main__':

    # process args
    args = p.parse_args()
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    # process config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    slc = parse_slice(config)
    slc = (slc[0], slc[1], slc[2].start + args.z)

    inset = (slice(args.y, args.y + args.psf_window_size),
             slice(args.x, args.x + args.psf_window_size))

    image = load_volume(config, 'structured-plastic-reference')
    resolution_mm = image.meta.resolution_mm
    reference = image.data
    reference = np.abs(sp.ifft(sp.resize(sp.fft(reference), (256, 256, 64))))
    reference = normalize(reference[slc], pct=100)
    target = sr.gaussian_blur(reference, args.sigma, args.psf_radius, axes=(0, 1))
    psf = sr.gaussian_psf(reference.shape, args.sigma, args.psf_radius, axes=(0, 1))
    psf = sp.resize(psf, (args.psf_window_size, args.psf_window_size))
    psf = psf / np.max(psf)

    fig = plt.figure(figsize=(FIG_WIDTH[2], FIG_WIDTH[2]*0.5))
    subfigs = fig.subfigures(1, 2, width_ratios=[1, 3], wspace=0.04)

    plot_inputs(subfigs[0], target, reference, inset)
    plot_model(subfigs[1], target[inset], reference[inset], psf, args.unit_cell_size)

    label_panels(subfigs)
    color_panels(subfigs)

    plt.savefig(path.join(args.save_dir, 'figure4.png'), dpi=DPI)
    plt.savefig(path.join(args.save_dir, 'figure4.pdf'), dpi=DPI)

    if args.plot:
        plt.show()