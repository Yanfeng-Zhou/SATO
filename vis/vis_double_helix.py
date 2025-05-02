import numpy as np
import networkx as nx
from mayavi import mlab
import os
import SimpleITK as sitk
import argparse
import skimage

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_path', default='D:/Desktop/ECCV2024/Code/demo_image/mask/synthetic_double_helix.tif')
    parser.add_argument('--straighten_path', default='D:/Desktop/ECCV2024/Code/demo_image/straightened/synthetic_double_helix.tif')
    args = parser.parse_args()

    seg = sitk.ReadImage(args.seg_path)
    seg = sitk.GetArrayFromImage(seg)

    seg_1 = seg.copy()
    seg_1[seg != 1] = 0
    seg_2 = seg.copy()
    seg_2[seg != 2] = 0
    seg_key = seg.copy()
    seg_key[seg != 3] = 0

    straighten = sitk.ReadImage(args.straighten_path)
    straighten = sitk.GetArrayFromImage(straighten)

    straighten_1 = straighten.copy()
    straighten_1[straighten != 1] = 0
    straighten_2 = straighten.copy()
    straighten_2[straighten != 2] = 0
    straighten_key = straighten.copy()
    straighten_key[straighten != 3] = 0

    figure1 = mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    mlab.contour3d(seg_1, contours=[1], color=(0.49, 0.99, 0), opacity=0.9, figure=figure1)
    mlab.contour3d(seg_2, contours=[1], color=(0.49, 0.99, 0), opacity=0.9, figure=figure1)
    mlab.contour3d(seg_key, contours=[1], color=(0.25, 0.41, 0.88), opacity=0.9, figure=figure1)

    figure2 = mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    mlab.contour3d(straighten_1, contours=[1], color=(0.49, 0.99, 0), opacity=0.9, figure=figure2)
    mlab.contour3d(straighten_2, contours=[1], color=(0.49, 0.99, 0), opacity=0.9, figure=figure2)
    mlab.contour3d(straighten_key, contours=[1], color=(0.25, 0.41, 0.88), opacity=0.9, figure=figure2)

    mlab.show()