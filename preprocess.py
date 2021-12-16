import cv2
import os
from skimage import io
import dlib


# get the bounding box of the face
def crop_face(pic):
    detector = dlib.get_frontal_face_detector()
    pic = io.imread(pic)
    face = detector(pic, 0)
    x1, y1, x2, y2 = face[0].left(), face[0].top(), face[0].right(), face[0].bottom()

    return (x1, y1, x2, y2)


# input params:
# @video_path: path to diretory containing frame sequence of a video
# 	e.g. ./data/your_dataset_rawdata/train/006_1_2
# @s_path: path to the diretory you would like to save(without video name)
#	e.g. ./data/your_dataset/train
def crop_and_save(video_path, s_path):
	i_names = os.listdir(video_path)
    s_path = os.path.join(s_path, os.path.basename(video_path))
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    p0 = os.path.join(video_path, i_names[0])

    # get the face bounding box of the onset frame
    box = crop_face(p0)
    for i in i_names:
        i_path = os.path.join(video_path, i)
        img = cv2.imread(i_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)# get 3-channel grayscale (optional)

        # crop and resize to 256 x 256
        cropped_image = img[box[1]:box[3], box[0]:box[2]]
        cropped_image = cv2.resize(cropped_image, (256, 256))

        cropped_image = cv2.merge([cropped_image, cropped_image, cropped_image])# get 3-channel grayscale (optional)
        cv2.imwrite(os.path.join(s_path, i[:-3]+"png"), cropped_image)


if __name__ == '__main__':
	path = r"./source_samples/Smic_challenge"
	s_path = r"./data/your_dataset/test"

	cnt = 0
	for video_name in os.listdir(path):
		video_path = os.path.join(path, video_name)
		crop_and_save(video_path, s_path)
		cnt += 1
		print(str(cnt)+" completed: " + video_path)