import cv2
import pandas as pd
import torch
from ultralytics import YOLO
import math
import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate('firebase-credentials.json')

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://temp-fbb14-default-rtdb.asia-southeast1.firebasedatabase.app/' # <-- IMPORTANT: REPLACE WITH YOUR URL
})

root_ref = db.reference()

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.id_to_class = {}
        self.id_to_confidence = {}

    def update(self, objects_rect, class_names, confidences):
        objects_bbs_ids = []
        for rect, class_name, confidence in zip(objects_rect, class_names, confidences):
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 50:
                    if confidence > self.id_to_confidence.get(id, 0):
                        self.center_points[id] = (cx, cy)
                        self.id_to_class[id] = class_name
                        self.id_to_confidence[id] = confidence
                    objects_bbs_ids.append([x, y, w, h, id, class_name])
                    same_object_detected = True
                    break
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                self.id_to_class[self.id_count] = class_name
                self.id_to_confidence[self.id_count] = confidence
                objects_bbs_ids.append([x, y, w, h, self.id_count, class_name])
                self.id_count += 1
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, _ = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

# Check if CUDA is available and set PyTorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('yolov8x.pt')
# nsmlx small to big
model.to(device)

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        #print(colorsBGR)

def wrap_text(text, max_width, font, font_scale, thickness):
    words = text.split(' ')
    lines = []
    current_line = words[0]

    for word in words[1:]:
        if cv2.getTextSize(current_line + ' ' + word, font, font_scale, thickness)[0][0] <= max_width:
            current_line += ' ' + word
        else:   
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(0)

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

print(class_list)

frame_count = 0

tracker = Tracker()

cy1 = 200
cy2 = 320
offset = 60

vh_down = {}
vh_up = {}
incounter = {}
outcounter = {}
objects_inside_names = []

while True:
    ret, frame = cap.read()
    frame_count += 1
    frame = cv2.resize(frame, (800, 500))

    results = model.predict(frame, show=True)
    a = results[0].boxes.data
    px = pd.DataFrame(a.cpu()).astype("float")
    bbox_list = []
    class_names = []
    confidences = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        confidence = float(row[4])
        d = int(row[5])
        c = class_list[d]
        if c == 'person':
            continue
        bbox_list.append([x1, y1, x2, y2])
        class_names.append(c)
        confidences.append(confidence)

    bbox_id = tracker.update(bbox_list, class_names, confidences)

    for bbox in bbox_id:
        x3, y3, x4, y4, id, class_name = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        print(f"Tracking ID: {id}, Class: {class_name}, Coordinates: ({cx}, {cy})")  # Debugging statement
        if cy1 < (cy + offset) and cy1 > (cy - offset):  # going in
            vh_down[id] = (cy, class_name, False)  # Add a flag for counting
        if id in vh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset) and not vh_down[id][2]:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"{id} ({class_name})", (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                incounter[id] = class_name
                objects_inside_names.append(class_name)
                vh_down[id] = (cy, class_name, True)  # Update the flag to True

        if cy2 < (cy + offset) and cy2 > (cy - offset):  # going out
            vh_up[id] = (cy, class_name, False)  # Add a flag for counting
        if id in vh_up:
            if cy1 < (cy + offset) and cy1 > (cy - offset) and not vh_up[id][2]:
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"{id} ({class_name})", (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                outcounter[id] = class_name
                if class_name in objects_inside_names:
                    objects_inside_names.remove(class_name)
                vh_up[id] = (cy, class_name, True)  # Update the flag to True

    cv2.line(frame, (0, cy1), (800, cy1), (255, 255, 255), 1)
    cv2.line(frame, (0, cy2), (800, cy2), (255, 255, 255), 1)
    cv2.putText(frame, f"going in: {list(incounter.items())}", (60, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"going out: {list(outcounter.items())}", (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Objects inside: {objects_inside_names}", (60, 360), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Number of objects inside: {len(incounter) - len(outcounter)}", (60, 450), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("RGB", frame)
    if frame_count % 30 == 0:
        data_ref = root_ref.child('Data')
        data = {
            'going in': str(list(incounter.items())),
            'going out': str(list(outcounter.items())),
            'Number of objects inside': str(len(incounter) - len(outcounter)),
            'Objects inside': str(objects_inside_names)
        }
        data_ref.set(data)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()