import math
import cv2
import json
import numpy as np
import PIL.Image

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))

    # # save image
    # out_img_pil.save('output.jpg')
    return out_img_pil

if __name__ == '__main__':
    image = PIL.Image.open('data/CelebA_sm/imgs_sm/001677.jpg')
    json_data = json.load(open('data/CelebA_sm/data.json', 'r'))
    kps = [a for a in json_data if a['image_file'] == 'data/CelebA_sm/imgs_sm/001677.jpg'][0]['landmarks']
    kps = [kps['left_eye'], kps['right_eye'], kps['nose'], kps['mouth_left'], kps['mouth_right']]

    draw_kps(image, kps)

