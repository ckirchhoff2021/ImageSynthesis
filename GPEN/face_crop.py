import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

import cv2
import __init_paths
from face_detect.retinaface_detection import RetinaFaceDetection
from align_faces import warp_and_crop_face, get_reference_facial_points
import numpy as np
from tqdm import tqdm


def detect(file_dir, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    reference_5pts = get_reference_facial_points(
        (512, 512), inner_padding_factor, outer_padding, default_square)

    files = os.listdir(file_dir)

    for name in tqdm(files):
        file = os.path.join(file_dir, name)
        try:
            data = cv2.imread(file, cv2.IMREAD_COLOR)
            print(data.shape)
            facedetector = RetinaFaceDetection('./')
            facebs, landms = facedetector.detect(data)
        except:
            continue

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4] < 0.9: continue
            fh, fw = (faceb[3] - faceb[1]), (faceb[2] - faceb[0])
            x1, y1, x2, y2 = int(faceb[0]), int(faceb[1]), int(faceb[2]), int(faceb[3])
            print(x1, y1, x2, y2)
            crop = data[y1:y2, x1:x2, :]
            cv2.imwrite('examples/crop.jpg', crop)

            print('height: %d, width : %d' % (fw, fw))
            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(data, facial5points, reference_pts=reference_5pts, crop_size=(512, 512))
            # print(of.shape)
            cv2.imwrite(os.path.join(save_folder, name), of)
            break

if __name__ == '__main__':
    file_dir = '/data/juicefs_hz_cv_v3/11145199/datas/samples/20220217/samples'
    save_dir = '/data/juicefs_hz_cv_v3/11145199/datas/samples/20220217/faces'

    file_dir = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/test_score/deblur_test_dataset_all/medium_light'
    save_dir = '/data/juicefs_hz_cv_v3/11145199/datas/test_data/20220218/medium_faces_all'

    # file_dir = '/data/juicefs_hz_cv_v3/public_data/motion_deblur/test_score/20210119_noeyeglass/medium_light'
    # save_dir = '/data/juicefs_hz_cv_v3/11145199/datas/test_data/20220218/medium_faces_noglass'
    detect(file_dir, save_dir)


