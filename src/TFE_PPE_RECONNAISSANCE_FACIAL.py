import sys
sys.path.insert(0,'./codes')
sys.path.insert(0,'./codes/yolov5/')

import cv2
import numpy as np
import torch
import pafy
from torchvision import transforms

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, check_img_size
from utils.augmentations import letterbox


def get_id_from_youtube(url):
    video = pafy.new(url)
    best  = video.getbest()
    return best.url

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[[0, 2]] -= pad[0]  # x padding
    coords[[1, 3]] -= pad[1]  # y padding
    coords[:4] /= gain
    return coords

config = './codes/yolov5/config.yaml'
half = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = './codes/yolov5/runs/train/ppe_y5m_30epc_v5_chv+/weights/best.pt'
model = DetectMultiBackend(model_path, device=device, data=config, fp16=half)

# Définir les classes à détecter
classes = ["personne", "casque", "veste"]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# Initialiser la capture vidéo
# video_path = 1
video_path = './data/videos/demo0.2.mp4'
# video_path = get_id_from_youtube('https://www.youtube.com/watch?v=bKBlwgxk-rk')
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

face_reco = True
face_blur = False

vide_output_name = 'video_face_noblur.avi'

writer = cv2.VideoWriter(vide_output_name, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, size)

# Initialiser l'état de la vidéo
video_status = "playing"



if face_reco:
    import face_recognition

    # FACE stuff
    sohaib_image = face_recognition.load_image_file("./codes/target/sohaib.jpg")
    imran_image = face_recognition.load_image_file("./codes/target/imran.jpg")

    sohaib_encoding = face_recognition.face_encodings(sohaib_image)[0]
    imran_encoding = face_recognition.face_encodings(imran_image)[0]


def process(frame):
    h, w = frame.shape[:2]
    img = letterbox(frame, [640]*2, stride=model.stride, auto=True)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Passer l'image dans le modèle pour faire la détection
    output = model(img)
    output = non_max_suppression(output, conf_thres=0.2, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)[0]

    preds = []
    for det in output:
        det[:4] = scale_coords(img.shape[2:], det[:4], frame.shape).round()
        preds.append(det)

    return preds

def intersect(rect1, rect2):
    x0 = max(rect1[0], rect2[0])
    y0 = max(rect1[1], rect2[1])
    x1 = min(rect1[2], rect2[2])
    y1 = min(rect1[3], rect2[3])
    if x0 < x1 and y0 < y1:
        return (x1 - x0) * (y1 - y0)/((rect2[2]-rect2[0])*(rect2[3]-rect2[1]))
    else:
        return 0


def draw_rect_beautify(frame, f_i, x, y, w, h, alpha=0.3, zoom=0.5, color=(0, 0, 255), anim_frames=10, fill=False):
    # Create a copy of the frame with the same size

    if fill:
        fill = -1

    overlay = frame.copy()

    # Define the current coordinates of the rectangle
    x_current = x - int((zoom * (anim_frames - f_i) / anim_frames)*w/4)
    y_current = y - int((zoom * (anim_frames - f_i) / anim_frames)*h/4)
    w_current = w + int((zoom * (anim_frames - f_i) / anim_frames)*w/2)
    h_current = h + int((zoom * (anim_frames - f_i) / anim_frames)*h/2)

    # Draw the rectangle on the overlay
    cv2.rectangle(overlay, (x_current, y_current), (x_current + w_current, y_current + h_current), color, fill)

    # Apply the overlay on the frame with transparency
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.line(frame, (x_current, y_current), (int(x_current+w_current/10), y_current), color, 5)
    cv2.line(frame, (x_current, y_current), (x_current, int(y_current+w_current/10)), color, 5)

    cv2.line(frame, (x_current+w_current, y_current), (int(x_current+w_current-w_current/10), y_current), color, 5)
    cv2.line(frame, (x_current+w_current, y_current), (x_current+w_current, int(y_current+w_current/10)), color, 5)

    cv2.line(frame, (x_current, y_current+h_current), (int(x_current+w_current/10), y_current+h_current), color, 5)
    cv2.line(frame, (x_current, y_current+h_current), (x_current, int(y_current+h_current-w_current/10)), color, 5)

    cv2.line(frame, (x_current+w_current, y_current+h_current), (int(x_current+w_current-w_current/10), y_current+h_current), color, 5)
    cv2.line(frame, (x_current+w_current, y_current+h_current), (x_current+w_current, int(y_current+h_current-w_current/10)), color, 5)


ret, frame = cap.read()

cv2.namedWindow("Detections et suivis", cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Detections et suivis', 1080, 720)

f_i = 0
anim_frames = 10
ksize = (30, 30)

while ret:
    if face_reco:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)


    # objects = dict.fromkeys(classes, [])
    objects = {'personne': [], 'casque': [], 'veste': []}


    if video_status == "paused":
        cv2.putText(frame, 'PAUSE', (10, 720 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Detections et suivis", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key != -1:
            video_status = "playing"
        continue

    preds = process(frame)

    for i, p in enumerate(preds):
        p = p.cpu().detach().numpy()
        x0, y0, x1, y1, c = int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[5])
        objects[classes[c]].append([x0, y0, x1, y1])

    
    for p in objects['personne']:
        x0, y0, x1, y1 = p[0], p[1], p[2], p[3]
        if face_reco:
            for i, face_encoding in enumerate(face_encodings):
                sohaib_result = face_recognition.compare_faces([sohaib_encoding], face_encoding)
                imran_result = face_recognition.compare_faces([imran_encoding], face_encoding)
                top, right, bottom, left = face_locations[i]
                # (left, top), (right, bottom)
                if face_blur:
                    frame[top:bottom, left:right, :] = cv2.blur(frame[top:bottom, left:right, :], ksize, cv2.BORDER_DEFAULT)

                if sohaib_result[0]:
                    name = 'Worker1'
                elif imran_result[0]:
                    name = 'Worker2'
                else: name = 'Unknown'

                if intersect(p, [left, top, right, bottom]) >= 0.9:
                    cv2.putText(frame, name, (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 0, 0), 2, cv2.LINE_AA)


        vest_ok = False
        casque_ok = False
        for v in objects['veste']:
            if intersect(p, v) >= 0.75:
                w = v[2]-v[0]
                h = v[3]-v[1]

                draw_rect_beautify(frame, 0, v[0], v[1], w, h, alpha=0.3, zoom=0.1, color=(0, 255, 0), anim_frames=3, fill=1)




                vest_ok = True
                break
        for q in objects['casque']:
            if intersect(p, q) >= 0.75:
                w = q[2]-q[0]
                h = q[3]-q[1]
                # cv2.rectangle(frame, (q[0], q[1]), (q[2], q[3]), (0, 255, 0), 2)
                draw_rect_beautify(frame, 0, q[0], q[1], w, h, alpha=0.3, zoom=0.1, color=(0, 255, 0), anim_frames=3, fill=1)
                casque_ok = True
                break

        if vest_ok:
            if casque_ok:
                # cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                draw_rect_beautify(frame, 0, x0, y0, x1-x0+1, y1-y0+1, alpha=0.3, zoom=0.1, color=(0, 255, 0), anim_frames=3, fill=True)

            else:
                # cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 165, 255), 2)
                draw_rect_beautify(frame, f_i, x0, y0, x1-x0+1, y1-y0+1, alpha=0.3, zoom=0.1, color=(0, 165, 255), anim_frames=3, fill=True)
        else:
            if casque_ok:
                # cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 165, 255), 2)
                draw_rect_beautify(frame, f_i, x0, y0, x1-x0+1, y1-y0+1, alpha=0.3, zoom=0.1, color=(0, 165, 255), anim_frames=3, fill=True)
            else:
                # cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                draw_rect_beautify(frame, f_i, x0, y0, x1-x0+1, y1-y0+1, alpha=0.3, zoom=0.1, color=(0, 0, 255), anim_frames=3, fill=True)


    f_i += 1
    if f_i == anim_frames:
        f_i = 0

    # Afficher l'image avec les détections et les suivis
    cv2.imshow("Detections et suivis", frame)
    writer.write(frame)

    # Quitter si l'utilisateur appuie sur la touche 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    key = cv2.waitKey(1)
    if key == ord('p'):
        video_status = "paused"
    elif key == ord('q'):
        break
    elif key != -1:
        video_status = "playing"

    # Lire une frame de la webcam
    if video_status == "playing":
        # Lire le frame suivant
        ret, frame = cap.read()



# Libérer la webcam et fermer la fenêtre
cap.release()
writer.release()
cv2.destroyAllWindows()