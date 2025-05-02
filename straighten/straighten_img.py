import os
import numpy as np
import argparse
import scipy
from scipy import ndimage
from scipy import interpolate
import skimage
from skimage import measure
import SimpleITK as sitk
from scipy.spatial.transform import Rotation
from skimage.morphology import remove_small_holes, remove_small_objects
from collections import Counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_path', default='D:/Desktop/synthetic_circular_helix.tif')
    parser.add_argument('--centerline_path', default='D:/Desktop/synthetic_circular_helix_.npy')
    parser.add_argument('--save_straighten_path', default='D:/Desktop/synthetic_circular_helix_straighten_smooth.tif')
    parser.add_argument('--crop_radius_ratio', default=5)
    args = parser.parse_args()

    image = sitk.ReadImage(args.seg_path)
    image = sitk.GetArrayFromImage(image)

    centerline = np.load(args.centerline_path, allow_pickle=True)
    centerline = centerline.tolist()

    length = int(centerline['edge_length'])

    radius = []
    for j in range(len(centerline['point'])):

        x = centerline['point'][j]['coordinate'][0]
        y = centerline['point'][j]['coordinate'][1]
        z = centerline['point'][j]['coordinate'][2]
        if j == 0:
            coordinate = np.array([[x, y, z]])
        else:
            coordinate = np.concatenate((coordinate, np.array([[x, y, z]])), axis=0)

        radius.append(centerline['point'][j]['width'])
    max_radius = int(5 * np.array(radius).max())

    x_coordinate = coordinate[:, 0]
    y_coordinate = coordinate[:, 1]
    z_coordinate = coordinate[:, 2]

    tck, myu = interpolate.splprep([x_coordinate, y_coordinate, z_coordinate])

    myu = np.linspace(myu.min(), myu.max(), length)
    dx, dy, dz = interpolate.splev(myu, tck, der=1)
    x_coordinate, y_coordinate, z_coordinate = interpolate.splev(myu, tck)

    for m in range(len(dx)):
        uz = np.array([dx[m], dy[m], dz[m]])
        if m == 0:
            if dx[m] == 0:
                ux = np.array([0, -dz[m], dy[m]])
            elif dy[m] == 0:
                ux = np.array([-dz[m], 0, dx[m]])
            else:
                ux = np.array([-dy[m], dx[m], 0])
            uy = np.cross(uz, ux)

        else:
            if (uz == uz_before).all():
                ux = ux_before
                uy = uy_before
            else:
                intersect_vector = np.cross(uz_before, uz)
                intersect_vector_normal = intersect_vector / np.linalg.norm(intersect_vector, axis=0)

                cos_angle_flat = np.dot(uz, uz_before) / (np.sqrt(uz.dot(uz)) * np.sqrt(uz_before.dot(uz_before)))
                theta = np.arccos(cos_angle_flat)

                rot = Rotation.from_rotvec(theta * intersect_vector_normal)
                ux = rot.apply(ux_before)
                uy = rot.apply(uy_before)

        ux_normal = ux / np.linalg.norm(ux, axis=0)
        uy_normal = uy / np.linalg.norm(uy, axis=0)
        uz_normal = uz / np.linalg.norm(uz, axis=0)

        ux_before = ux_normal
        uy_before = uy_normal
        uz_before = uz_normal

        R = np.array([[ux_normal[0], uy_normal[0], uz_normal[0], x_coordinate[m]],
                      [ux_normal[1], uy_normal[1], uz_normal[1], y_coordinate[m]],
                      [ux_normal[2], uy_normal[2], uz_normal[2], z_coordinate[m]],
                      [0, 0, 0, 1]])

        # coordinate matrix in new coordinate system
        coordinate_matrix_one = np.linspace(-max_radius, max_radius, 2*max_radius+1)
        coordinate_matrix_one = np.tile(coordinate_matrix_one, 2*max_radius+1)
        coordinate_matrix_two = np.arange(max_radius, -max_radius-1, -1)
        coordinate_matrix_two = np.repeat(coordinate_matrix_two, 2*max_radius+1)
        coordinate_matrix_three = np.zeros((2*max_radius+1)**2)
        coordinate_matrix_four = np.ones((2*max_radius+1)**2)
        coordinate_matrix = np.stack((coordinate_matrix_one, coordinate_matrix_two, coordinate_matrix_three, coordinate_matrix_four), axis=0)

        coordinate_matrix_ori = np.dot(R, coordinate_matrix)
        coordinate_matrix_ori = coordinate_matrix_ori[:3, :]

        gray_value = ndimage.map_coordinates(image, coordinate_matrix_ori, order=1)
        gray_value = gray_value.reshape(2*max_radius+1, 2*max_radius+1)

        if m == 0:
            straighten_image = np.array([gray_value])
        else:
            straighten_image = np.concatenate((straighten_image, np.array([gray_value])), axis=0)

    straighten_image = sitk.GetImageFromArray(straighten_image.astype(np.uint8))
    sitk.WriteImage(straighten_image, args.save_straighten_path)