from skimage import io
import dlib
import numpy as np
import torch

# 如果对网络输入一张图片，请从此处开始使用代码
def feature_generator(pic, model):
    au_num = np.empty([18, 2], dtype = np.float32)
    pic =pic[:,:,0:3]
    # >>>>>>>> 人脸模型预测区 >>>>>>>>>>
    predictor = model
    detector = dlib.get_frontal_face_detector()
    dets = detector(pic, 1)
    if len(dets) == 0:
        au_num = torch.load(r'C:\Users\唐梓轩\Desktop\微表情识别\新建文件夹\first-order-model-master\kp.pt').to("cpu")
        au_num = au_num[0]
    else:
        for k, d in enumerate(dets):
            shape = predictor(pic, d)
        i = 0
        vec = np.empty([68, 2], dtype=int)
        for b in range(68):
            vec[b][0] = shape.part(b).x
            vec[b][1] = shape.part(b).y
            if b in [19, 24, 27, 38, 41, 43, 46, 48, 54, 55, 57, 59]:
                au_num[i] = vec[b]
                i += 1
        for c in range(68):
            if c in [31, 35, 41, 46, 48, 54]:
                au_num[i] = vec[c]
                i += 1
        eye_distance = vec[42][0] - vec[39][0]

        au_num[12][0] = au_num[12][0] - eye_distance / 2
        au_num[13][0] = au_num[13][0] + eye_distance / 2
        au_num[14][1] = au_num[14][1] + eye_distance / 2
        au_num[15][1] = au_num[15][1] + eye_distance / 2
        au_num[16][0] = au_num[16][0] - eye_distance / 2
        au_num[17][0] = au_num[17][0] + eye_distance / 2
        # >>>>>>>>   关键点区 >>>>>>>>>
        # a = make_masks(np.array(au_num), (pic.shape[0],pic.shape[1]), (64,64), 1)
        au_num = torch.tensor(au_num)
    a = 0
    return [a, au_num.requires_grad_(True)]



