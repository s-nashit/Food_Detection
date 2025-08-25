from ultralytics import YOLO
import cv2
from collections import defaultdict
import pandas as pd
from datetime import datetime

def count_unique_objects(video_path, output_path, model_path):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracked_ids = set()
    object_counts = defaultdict(int)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=0.3, imgsz=640)[0]


        if results is not None and results.boxes.id is not None:
            boxes = results.boxes
            ids = boxes.id.cpu().tolist()
            
            cls = boxes.cls.cpu().tolist()

            for obj_id, class_id in zip(ids, cls):
                if obj_id not in tracked_ids:
                    tracked_ids.add(obj_id)
                    class_name = model.names[int(class_id)]
                    object_counts[class_name] += 1

        annotated_frame = results.plot()
        y0 = 30
        for i, (item, count) in enumerate(object_counts.items()):
            cv2.putText(annotated_frame, f"{item}: {count}", (10, y0 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(annotated_frame)

    cap.release()
    out.release()

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df = pd.DataFrame(object_counts.items(), columns=["Class", "Count"])
    csv_path = f"counts_{date_str}.csv"
    df.to_csv(csv_path, index=False)

    return output_path, csv_path, df
