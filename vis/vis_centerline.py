import numpy as np
import networkx as nx
from mayavi import mlab
import os
import SimpleITK as sitk
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_path', default='D:/Desktop/ECCV2024/Code/demo_image/mask/stenosis.tif')
    parser.add_argument('--centerline_path', default='D:/Desktop/ECCV2024/Code/demo_image/centerline/stenosis.npy')
    parser.add_argument('--color', default=(0.5, 0, 0.5), help=
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

    image = sitk.ReadImage(args.seg_path)
    image = sitk.GetArrayFromImage(image)

    figure = mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    mlab.contour3d(image, contours=[1], color=args.color, opacity=0.5, figure=figure)

    centerline_information = np.load(args.centerline_path, allow_pickle=True)
    centerline_information = centerline_information.tolist()

    node = []
    node_number = []

    node_number.append(centerline_information['start_point'])
    node_number.append(centerline_information['end_point'])
    node.append(centerline_information['start_coordinate'])
    node.append(centerline_information['end_coordinate'])
    edge = []
    for j in centerline_information['point']:
        edge.append(j['coordinate'])
    edge = np.array(edge)
    edge_x = edge[:, 0]
    edge_y = edge[:, 1]
    edge_z = edge[:, 2]

    mlab.plot3d(edge_x, edge_y, edge_z, color=(0, 0, 1), opacity=0.65, tube_radius=1.3, tube_sides=20, figure=figure)

    node_repeat = [i for i, x in enumerate(node_number) if x in node_number[:i]]
    node_copy = node.copy()
    for k in reversed(node_repeat):
        del node_copy[k]

    node_copy = np.array(node_copy)
    node_copy_x = node_copy[:, 0]
    node_copy_y = node_copy[:, 1]
    node_copy_z = node_copy[:, 2]

    mlab.points3d(node_copy_x, node_copy_y, node_copy_z, color=(0, 1, 0), opacity=0.8, scale_factor=3, figure=figure)

    mlab.show()