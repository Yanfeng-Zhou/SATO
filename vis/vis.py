import numpy as np
import networkx as nx
from mayavi import mlab
import os
import SimpleITK as sitk
import argparse
import skimage
from scipy import ndimage

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_path', default='D:/Desktop/ECCV2024/Code/demo_image/mask/synthetic_circular_helix.tif')
    parser.add_argument('--straighten_path', default='D:/Desktop/ECCV2024/Code/demo_image/straightened/synthetic_circular_helix.tif')
    parser.add_argument('--color', default=(1, 0, 0), help=
    'red: (1, 0, 0), '
    'blue: (0, 0, 1), '
    'lime: (0, 1, 0), '
    'green: (0, 0.5, 0), '
    'yellow: (1, 1, 0), '
    'orange: (1, 0.65, 0), '
    'DarkGoldenrod: (0.72, 0.53, 0.04)'
    'purple: (0.5, 0, 0.5)'
    'gray: (0.5, 0.5, 0.5)'
    'black: (0, 0, 0)'
    '(1, 0.87, 0.68)'
    '(0.49, 0.99, 0)'
    )
    args = parser.parse_args()

    seg = sitk.ReadImage(args.seg_path)
    seg = sitk.GetArrayFromImage(seg)

    straighten = sitk.ReadImage(args.straighten_path)
    straighten = sitk.GetArrayFromImage(straighten)


    figure1 = mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    mlab.contour3d(seg, contours=[1], color=args.color, opacity=0.9, figure=figure1)


    figure2 = mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    mlab.contour3d(straighten, contours=[1], color=args.color, opacity=0.9, figure=figure2)


    mlab.show()