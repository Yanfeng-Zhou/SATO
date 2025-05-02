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

def save_center_objects(image, x_size, y_size):
    labeled_image = skimage.measure.label(image)
    labeled_list = skimage.measure.regionprops(labeled_image)
    for i in range(len(labeled_list)):

        coords = list(labeled_list[i].coords)
        if (coords == np.array([1, int((x_size-1)/2), int((y_size-1)/2)])).all(1).any():
            label_num = i + 1

    labeled_image[labeled_image != label_num] = 0
    labeled_image[labeled_image == label_num] = 1
    # labeled_image = labeled_image.astype(bool)

    return labeled_image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_1_path', default='D:/Desktop/synthetic_double_helix_1.tif')
    parser.add_argument('--seg_2_path', default='D:/Desktop/synthetic_double_helix_2.tif')
    parser.add_argument('--seg_key_path', default='D:/Desktop/synthetic_double_helix_key.tif')
    parser.add_argument('--centerline_1_path', default='D:/Desktop/synthetic_double_helix_1.npy')
    parser.add_argument('--centerline_2_path', default='D:/Desktop/synthetic_double_helix_2.npy')

    parser.add_argument('--save_straighten_1_path', default='D:/Desktop/synthetic_double_helix_straighten_1_smooth.tif')
    parser.add_argument('--save_straighten_2_path', default='D:/Desktop/synthetic_double_helix_straighten_2_smooth.tif')
    parser.add_argument('--save_straighten_key_path', default='D:/Desktop/synthetic_double_helix_straighten_key_smooth.tif')
    parser.add_argument('--save_straighten_path', default='D:/Desktop/synthetic_double_helix_straighten_smooth.tif')

    parser.add_argument('--radius', default=10, type=int)
    parser.add_argument('--rotation_radius', default=50, type=int)
    parser.add_argument('--crop_radius_ratio', default=3)
    parser.add_argument('--if_smooth', default=True)
    parser.add_argument('--remove_small_holes_thr', default=500)
    args = parser.parse_args()

    image_1 = sitk.ReadImage(args.seg_1_path)
    image_1 = sitk.GetArrayFromImage(image_1)

    image_2 = sitk.ReadImage(args.seg_2_path)
    image_2 = sitk.GetArrayFromImage(image_2)

    image_key = sitk.ReadImage(args.seg_key_path)
    image_key = sitk.GetArrayFromImage(image_key)

    centerline_1 = np.load(args.centerline_1_path, allow_pickle=True)
    centerline_1 = centerline_1.tolist()[0]

    centerline_2 = np.load(args.centerline_2_path, allow_pickle=True)
    centerline_2 = centerline_2.tolist()[0]

    size = int(args.crop_radius_ratio * args.radius)
    rotation_center = (args.rotation_radius + args.radius+10, args.rotation_radius + args.radius+10)
    key_size = args.rotation_radius + args.radius+10

    # straighten 1
    for j in range(len(centerline_1['point'])):

        x = centerline_1['point'][j]['coordinate'][0]
        y = centerline_1['point'][j]['coordinate'][1]
        z = centerline_1['point'][j]['coordinate'][2]
        if j == 0:
            coordinate = np.array([[x, y, z]])
        else:
            coordinate = np.concatenate((coordinate, np.array([[x, y, z]])), axis=0)

    x_coordinate_1 = coordinate[:, 0]
    y_coordinate_1 = coordinate[:, 1]
    z_coordinate_1 = coordinate[:, 2]

    tck, myu = interpolate.splprep([x_coordinate_1, y_coordinate_1, z_coordinate_1])
    dx, dy, dz = interpolate.splev(myu, tck, der=1)

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

        R = np.array([[ux_normal[0], uy_normal[0], uz_normal[0], x_coordinate_1[m]],
                      [ux_normal[1], uy_normal[1], uz_normal[1], y_coordinate_1[m]],
                      [ux_normal[2], uy_normal[2], uz_normal[2], z_coordinate_1[m]],
                      [0, 0, 0, 1]])

        # coordinate matrix in new coordinate system
        coordinate_matrix_one = np.linspace(-size, size, 2*size+1)
        coordinate_matrix_one = np.tile(coordinate_matrix_one, 2*size+1)
        coordinate_matrix_two = np.arange(size, -size-1, -1)
        coordinate_matrix_two = np.repeat(coordinate_matrix_two, 2*size+1)
        coordinate_matrix_three = np.zeros((2*size+1) ** 2)
        coordinate_matrix_four = np.ones((2*size+1) ** 2)
        coordinate_matrix = np.stack((coordinate_matrix_one, coordinate_matrix_two, coordinate_matrix_three, coordinate_matrix_four), axis=0)

        coordinate_matrix_ori = np.dot(R, coordinate_matrix)
        coordinate_matrix_ori = coordinate_matrix_ori[:3, :]

        gray_value = ndimage.map_coordinates(image_1, coordinate_matrix_ori, order=1)
        gray_value = gray_value.reshape(2*size+1, 2*size+1)

        if m == 0:
            straighten_image_1 = np.zeros([1, 2*size+1, 2*size+1])
            straighten_image_1 = np.concatenate((straighten_image_1, np.array([gray_value])), axis=0)
        elif m == len(dx) - 1:
            straighten_image_1 = np.concatenate((straighten_image_1, np.array([gray_value])), axis=0)
            straighten_image_1 = np.concatenate((straighten_image_1, np.zeros([1, 2*size+1, 2*size+1])), axis=0)
        else:
            straighten_image_1 = np.concatenate((straighten_image_1, np.array([gray_value])), axis=0)

    straighten_image_1 = straighten_image_1.astype(bool)
    # fill hole
    straighten_image_1 = remove_small_holes(straighten_image_1, args.remove_small_holes_thr)
    # remove small object
    straighten_image_1 = save_center_objects(straighten_image_1, 2*size+1, 2*size+1)
    # smooth
    if args.if_smooth:
        straighten_image_1 = ndimage.median_filter(straighten_image_1, size=3)
    sitk.WriteImage(sitk.GetImageFromArray(straighten_image_1.astype(np.uint8)), args.save_straighten_1_path)

    # straighten 2
    for j in range(len(centerline_2['point'])):

        x = centerline_2['point'][j]['coordinate'][0]
        y = centerline_2['point'][j]['coordinate'][1]
        z = centerline_2['point'][j]['coordinate'][2]
        if j == 0:
            coordinate = np.array([[x, y, z]])
        else:
            coordinate = np.concatenate((coordinate, np.array([[x, y, z]])), axis=0)

    x_coordinate_2 = coordinate[:, 0]
    y_coordinate_2 = coordinate[:, 1]
    z_coordinate_2 = coordinate[:, 2]

    tck, myu = interpolate.splprep([x_coordinate_2, y_coordinate_2, z_coordinate_2])
    dx, dy, dz = interpolate.splev(myu, tck, der=1)

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

        R = np.array([[ux_normal[0], uy_normal[0], uz_normal[0], x_coordinate_2[m]],
                      [ux_normal[1], uy_normal[1], uz_normal[1], y_coordinate_2[m]],
                      [ux_normal[2], uy_normal[2], uz_normal[2], z_coordinate_2[m]],
                      [0, 0, 0, 1]])

        # coordinate matrix in new coordinate system
        coordinate_matrix_one = np.linspace(-size, size, 2*size+1)
        coordinate_matrix_one = np.tile(coordinate_matrix_one, 2*size+1)
        coordinate_matrix_two = np.arange(size, -size - 1, -1)
        coordinate_matrix_two = np.repeat(coordinate_matrix_two, 2*size+1)
        coordinate_matrix_three = np.zeros((2*size+1) ** 2)
        coordinate_matrix_four = np.ones((2*size+1) ** 2)
        coordinate_matrix = np.stack((coordinate_matrix_one, coordinate_matrix_two, coordinate_matrix_three, coordinate_matrix_four), axis=0)

        coordinate_matrix_ori = np.dot(R, coordinate_matrix)
        coordinate_matrix_ori = coordinate_matrix_ori[:3, :]

        gray_value = ndimage.map_coordinates(image_2, coordinate_matrix_ori, order=1)
        gray_value = gray_value.reshape(2*size+1, 2*size+1)

        if m == 0:
            straighten_image_2 = np.zeros([1, 2*size+1, 2*size+1])
            straighten_image_2 = np.concatenate((straighten_image_2, np.array([gray_value])), axis=0)
        elif m == len(dx) - 1:
            straighten_image_2 = np.concatenate((straighten_image_2, np.array([gray_value])), axis=0)
            straighten_image_2 = np.concatenate((straighten_image_2, np.zeros([1, 2*size+1, 2*size+1])), axis=0)
        else:
            straighten_image_2 = np.concatenate((straighten_image_2, np.array([gray_value])), axis=0)

    straighten_image_2 = straighten_image_2.astype(bool)
    # fill hole
    straighten_image_2 = remove_small_holes(straighten_image_2, args.remove_small_holes_thr)
    # remove small object
    straighten_image_2 = save_center_objects(straighten_image_2, 2*size+1, 2*size+1)
    # smooth
    if args.if_smooth:
        straighten_image_2 = ndimage.median_filter(straighten_image_2, size=3)
    sitk.WriteImage(sitk.GetImageFromArray(straighten_image_2.astype(np.uint8)), args.save_straighten_2_path)

    # straighten key
    for k in range(len(centerline_1['point'])):
        ux = np.array([rotation_center[0] - x_coordinate_1[k], rotation_center[1] - y_coordinate_1[k], 0])
        uz_normal = np.array([0, 0, 1])
        uy = np.cross(uz_normal, ux)
        ux_normal = ux / np.linalg.norm(ux, axis=0)
        uy_normal = uy / np.linalg.norm(uy, axis=0)

        R = np.array([[ux_normal[0], uy_normal[0], uz_normal[0], rotation_center[0]],
                      [ux_normal[1], uy_normal[1], uz_normal[1], rotation_center[1]],
                      [ux_normal[2], uy_normal[2], uz_normal[2], z_coordinate_1[k]],
                      [0, 0, 0, 1]])

        coordinate_matrix_one = np.linspace(-key_size, key_size, 2*key_size+1)
        coordinate_matrix_one = np.tile(coordinate_matrix_one, 2*key_size+1)
        coordinate_matrix_two = np.arange(key_size, -key_size-1, -1)
        coordinate_matrix_two = np.repeat(coordinate_matrix_two, 2*key_size+1)
        coordinate_matrix_three = np.zeros((2*key_size+1) ** 2)
        coordinate_matrix_four = np.ones((2*key_size+1) ** 2)
        coordinate_matrix = np.stack((coordinate_matrix_one, coordinate_matrix_two, coordinate_matrix_three, coordinate_matrix_four), axis=0)

        coordinate_matrix_ori = np.dot(R, coordinate_matrix)
        coordinate_matrix_ori = coordinate_matrix_ori[:3, :]

        gray_value = ndimage.map_coordinates(image_key, coordinate_matrix_ori, order=1)
        gray_value = gray_value.reshape(2*key_size+1, 2*key_size+1)

        if k == 0:
            straighten_image_key = np.zeros([1, 2*key_size+1, 2*key_size+1])
            straighten_image_key = np.concatenate((straighten_image_key, np.array([gray_value])), axis=0)
        elif k == len(dx) - 1:
            straighten_image_key = np.concatenate((straighten_image_key, np.array([gray_value])), axis=0)
            straighten_image_key = np.concatenate((straighten_image_key, np.zeros([1, 2*key_size+1, 2*key_size+1])), axis=0)
        else:
            straighten_image_key = np.concatenate((straighten_image_key, np.array([gray_value])), axis=0)

    straighten_image_key = straighten_image_key.astype(bool)
    # fill hole
    straighten_image_key = remove_small_holes(straighten_image_key, args.remove_small_holes_thr)
    # smooth
    if args.if_smooth:
        straighten_image_key = ndimage.median_filter(straighten_image_key, size=3)
    sitk.WriteImage(sitk.GetImageFromArray(straighten_image_key.astype(np.uint8)), args.save_straighten_key_path)

    # draw straighted key, helix1, helix2
    straighten_image_1_index_z, straighten_image_1_index_x, straighten_image_1_index_y = np.nonzero(straighten_image_1)
    straighten_image_2_index_z, straighten_image_2_index_x, straighten_image_2_index_y = np.nonzero(straighten_image_2)
    straighten_image_key_index_z, straighten_image_key_index_x, straighten_image_key_index_y = np.nonzero(straighten_image_key)

    straighten_image = np.zeros([straighten_image_2.shape[0], 2*(size+args.rotation_radius), 2*(size+args.rotation_radius)]).astype(np.uint8)
    straighten_image[straighten_image_key_index_z, straighten_image_key_index_x-10+(args.crop_radius_ratio-1)*args.radius, straighten_image_key_index_y-10+(args.crop_radius_ratio-1)*args.radius] = 3
    straighten_image[straighten_image_1_index_z, straighten_image_1_index_x+args.rotation_radius, straighten_image_1_index_y+2*args.rotation_radius] = 1
    straighten_image[straighten_image_2_index_z, straighten_image_2_index_x+args.rotation_radius, straighten_image_2_index_y] = 2

    sitk.WriteImage(sitk.GetImageFromArray(straighten_image.astype(np.uint8)), args.save_straighten_path)
