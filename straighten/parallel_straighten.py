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
import tqdm_pathos

def save_center_objects(image, x_size, y_size, z_size):
    labeled_image = skimage.measure.label(image)
    labeled_list = skimage.measure.regionprops(labeled_image)
    for i in range(len(labeled_list)):

        coords = list(labeled_list[i].coords)
        if (coords == np.array([int(z_size/2), int((x_size-1)/2), int((y_size-1)/2)])).all(1).any():
            label_num = i + 1

    labeled_image[labeled_image != label_num] = 0
    labeled_image[labeled_image == label_num] = 1
    # labeled_image = labeled_image.astype(bool)

    return labeled_image

def straighten_vessel(centerline_,
                      centerline_path,
                      image_path,
                      seg_path,
                      remove_duplicate_point,
                      crop_radius,
                      remove_small_holes_thr,
                      if_smooth,
                      if_remove_start_end,
                      save_straighten_seg_path,
                      save_straighten_img_path
                      ):
    centerline = np.load(os.path.join(centerline_path, centerline_), allow_pickle=True).item()
    name = os.path.splitext(centerline_)[0] + '.gz'

    image = sitk.ReadImage(os.path.join(image_path, name))
    image = sitk.GetArrayFromImage(image)

    seg = sitk.ReadImage(os.path.join(seg_path, name))
    seg = sitk.GetArrayFromImage(seg)

    for j in centerline:
        point_list = centerline[j]['point']
        save_straighten_name = os.path.splitext(centerline_)[0] + '_' + str(j) + '.tif'
        # print(i, j, centerline[j]['edge_width'], centerline[j]['edge_length'])

        for k in range(len(point_list)):

            x = point_list[k]['coordinate'][0]
            y = point_list[k]['coordinate'][1]
            z = point_list[k]['coordinate'][2]
            if k == 0:
                coordinate = np.array([[x, y, z]])
            else:
                coordinate = np.concatenate((coordinate, np.array([[x, y, z]])), axis=0)

        if remove_duplicate_point:
            unique_mask = np.any(coordinate[1:] != coordinate[:-1], axis=1)
            unique_mask = np.concatenate([[True], unique_mask])
            coordinate = coordinate[unique_mask]

        x_coordinate = coordinate[:, 0]
        y_coordinate = coordinate[:, 1]
        z_coordinate = coordinate[:, 2]
        tck, myu = interpolate.splprep([x_coordinate, y_coordinate, z_coordinate])
        # myu = np.linspace(myu.min(), myu.max(), length)
        dx, dy, dz = interpolate.splev(myu, tck, der=1)
        x_coordinate, y_coordinate, z_coordinate = interpolate.splev(myu, tck)

        # remove_start_end
        if if_remove_start_end:
            x_coordinate = x_coordinate[min(10, int(0.10 * len(x_coordinate))): max(len(x_coordinate) - 10, int(0.9 * len(x_coordinate)))]
            y_coordinate = y_coordinate[min(10, int(0.10 * len(y_coordinate))): max(len(y_coordinate) - 10, int(0.9 * len(y_coordinate)))]
            z_coordinate = z_coordinate[min(10, int(0.10 * len(z_coordinate))): max(len(z_coordinate) - 10, int(0.9 * len(z_coordinate)))]

            dx = dx[min(10, int(0.10 * len(dx))): max(len(dx) - 10, int(0.9 * len(dx)))]
            dy = dy[min(10, int(0.10 * len(dy))): max(len(dy) - 10, int(0.9 * len(dy)))]
            dz = dz[min(10, int(0.10 * len(dz))): max(len(dz) - 10, int(0.9 * len(dz)))]

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
            coordinate_matrix_one = np.linspace(-crop_radius, crop_radius, 2 * crop_radius + 1)
            coordinate_matrix_one = np.tile(coordinate_matrix_one, 2 * crop_radius + 1)
            coordinate_matrix_two = np.arange(crop_radius, -crop_radius - 1, -1)
            coordinate_matrix_two = np.repeat(coordinate_matrix_two, 2 * crop_radius + 1)
            coordinate_matrix_three = np.zeros((2 * crop_radius + 1) ** 2)
            coordinate_matrix_four = np.ones((2 * crop_radius + 1) ** 2)
            coordinate_matrix = np.stack((coordinate_matrix_one, coordinate_matrix_two, coordinate_matrix_three, coordinate_matrix_four), axis=0)

            coordinate_matrix_ori = np.dot(R, coordinate_matrix)
            coordinate_matrix_ori = coordinate_matrix_ori[:3, :]

            gray_value_img = ndimage.map_coordinates(image, coordinate_matrix_ori, order=3)
            gray_value_img = gray_value_img.reshape(2 * crop_radius + 1, 2 * crop_radius + 1)

            gray_value = ndimage.map_coordinates(seg, coordinate_matrix_ori, order=1)
            gray_value = gray_value.reshape(2 * crop_radius + 1, 2 * crop_radius + 1)

            if m == 0:
                straighten_seg = np.array([gray_value])
                straighten_img = np.array([gray_value_img])
            else:
                straighten_seg = np.concatenate((straighten_seg, np.array([gray_value])), axis=0)
                straighten_img = np.concatenate((straighten_img, np.array([gray_value_img])), axis=0)

        straighten_seg[straighten_seg >= 0.5] = 1
        straighten_seg[straighten_seg < 0.5] = 0
        straighten_seg = straighten_seg.astype(bool)
        # fill hole
        straighten_seg = remove_small_holes(straighten_seg, remove_small_holes_thr)
        # remove small object
        straighten_seg = save_center_objects(straighten_seg, 2 * crop_radius + 1, 2 * crop_radius + 1, int(centerline[j]['edge_length']))
        # smooths
        if if_smooth:
            straighten_seg = ndimage.median_filter(straighten_seg, size=3)

        straighten_seg = sitk.GetImageFromArray(straighten_seg.astype(np.uint8))
        sitk.WriteImage(straighten_seg, os.path.join(save_straighten_seg_path, save_straighten_name))

        straighten_img = sitk.GetImageFromArray(straighten_img.astype(np.int16))
        sitk.WriteImage(straighten_img, os.path.join(save_straighten_img_path, save_straighten_name))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='/...')
    parser.add_argument('--seg_path', default='/...')
    parser.add_argument('--centerline_path', default='/...')
    parser.add_argument('--save_straighten_img_path', default='/...')
    parser.add_argument('--save_straighten_seg_path', default='/...')

    parser.add_argument('--crop_radius', default=32)
    parser.add_argument('--remove_duplicate_point', default=True)
    parser.add_argument('--if_smooth', default=True)
    parser.add_argument('--if_remove_start_end', default=False)
    parser.add_argument('--remove_small_holes_thr', default=300)
    args = parser.parse_args()

    if not os.path.exists(args.save_straighten_img_path):
        os.makedirs(args.save_straighten_img_path)
    if not os.path.exists(args.save_straighten_seg_path):
        os.makedirs(args.save_straighten_seg_path)

    tqdm_pathos.map(straighten_vessel,
                    os.listdir(args.centerline_path),
                    args.centerline_path,
                    args.image_path,
                    args.seg_path,
                    args.remove_duplicate_point,
                    args.crop_radius,
                    args.remove_small_holes_thr,
                    args.if_smooth,
                    args.if_remove_start_end,
                    args.save_straighten_seg_path,
                    args.save_straighten_img_path
                    )



