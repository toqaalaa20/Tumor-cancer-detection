import json
import os
import numpy as np
import cv2


def convert_to_masks(frames_dir, json_file_dir, masks_save_dir):
    with open(json_file_dir, "r") as read_file:
        data = json.load(read_file)

    all_file_names = list(data.keys())

    Files_in_directory = []
    for root, dirs, files in os.walk(frames_dir):
        for filename in files:
            Files_in_directory.append(filename)

    for j in range(len(all_file_names)):
        image_name = data[all_file_names[j]]['filename']
        if image_name in Files_in_directory:
            img = np.asarray(cv2.imread(os.path.join(frames_dir, image_name)))
        else:
            continue

        if data[all_file_names[j]]['regions']:
            try:
                region = data[all_file_names[j]]['regions'][0]
                shape_attributes = region['shape_attributes']
                if 'all_points_x' in shape_attributes and 'all_points_y' in shape_attributes:
                    shape_x = shape_attributes['all_points_x']
                    shape_y = shape_attributes['all_points_y']
                elif region['shape_attributes']['name'] == 'ellipse':
                    shape_x, shape_y = [], []
                    ellipse = shape_attributes
                    center_x, center_y = ellipse['cx'], ellipse['cy']
                    major_axis, minor_axis = ellipse['rx'], ellipse['ry']
                    for angle in range(0, 360, 5):
                        angle_rad = np.deg2rad(angle)
                        x = center_x + major_axis * np.cos(angle_rad)
                        y = center_y + minor_axis * np.sin(angle_rad)
                        shape_x.append(x)
                        shape_y.append(y)
                else:
                    continue
            except Exception as e:
                print(f"Error accessing shape points: {e}")
                continue

            ab = np.stack((shape_x, shape_y), axis=1)

            img_copy = img.copy()

            cv2.drawContours(img_copy, [ab.astype(int)], -1, (255, 255, 255), -1)

            mask = np.zeros((img.shape[0], img.shape[1]))
            img3 = cv2.drawContours(mask, [ab.astype(int)], -1, 255, -1)

            try:
                cv2.imwrite(os.path.join(masks_save_dir, f'{j + 1}.png'), mask.astype(np.uint8))
                print(f"Image {j + 1} saved successfully")
            except Exception as e:
                print(f"Error saving image: {e}")


if __name__ == "__main__":
    frames_dir = "C:\\Users\\Toqaa\\Downloads\\archive (4)\\Br35H-Mask-RCNN\\TRAIN"
    json_file_dir = 'annotations_train.json'
    masks_save_dir = 'C:\\Users\\Toqaa\\Downloads\\archive (4)\\Br35H-Mask-RCNN\\train_masks'

    convert_to_masks(frames_dir, json_file_dir, masks_save_dir)
