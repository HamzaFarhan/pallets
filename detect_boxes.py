from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

SAME_COL_THRESH = 0.1


def drop_column(
    detections,
    image_width: int,
    which_col: str = "left",
    same_col_thresh: float = SAME_COL_THRESH,
):
    y_sorted_detections = detections[np.argsort(detections.xyxy[:, 1])]
    total_height = y_sorted_detections.xyxy[-1][3] - y_sorted_detections.xyxy[0][1]
    avg_height = np.mean(
        y_sorted_detections.xyxy[:, 3] - y_sorted_detections.xyxy[:, 1]
    )
    num_boxes = int((total_height / avg_height) * 0.6)
    print(f"NUM BOXES: {num_boxes}")
    if which_col == "left":
        sort_on = 0
        sorted_detections = detections[detections.xyxy[:, sort_on].argsort()]
    else:
        sort_on = 2
        sorted_detections = detections[(-detections.xyxy)[:, sort_on].argsort()]

    print(f"Sorted Detections: {sorted_detections}")
    col = {
        "xyxy": sorted_detections[:num_boxes].xyxy.tolist(),
        "mask": None,
        "confidence": sorted_detections[:num_boxes].confidence.tolist(),
        "class_id": sorted_detections[:num_boxes].class_id.tolist(),
        "tracker_id": None,
        "data": {
            "class_name": sorted_detections[:num_boxes].data["class_name"].tolist()
        },
    }
    res = {
        "xyxy": [],
        "mask": None,
        "confidence": [],
        "class_id": [],
        "tracker_id": None,
        "data": {"class_name": []},
    }
    print(image_width * same_col_thresh, image_width)
    print()
    for detection in sorted_detections[num_boxes:]:
        xyxy = detection[0].tolist()
        confidence = detection[2].tolist()
        class_id = detection[3].tolist()
        class_name = detection[5]["class_name"].tolist()
        curr_xyxy = col["xyxy"][-1]
        diff = abs(xyxy[sort_on] - curr_xyxy[sort_on])
        if diff < image_width * same_col_thresh:
            print(diff)
            col["xyxy"].append(xyxy)
            col["confidence"].append(confidence)
            col["class_id"].append(class_id)
            col["data"]["class_name"].append(class_name)
        else:
            res["xyxy"].append(xyxy)
            res["confidence"].append(confidence)
            res["class_id"].append(class_id)
            res["data"]["class_name"].append(class_name)
    print(f"Num Boxes in Column: {len(col['xyxy'])}")
    print(res)
    return (
        sv.Detections(
            xyxy=np.array(res["xyxy"]),
            confidence=np.array(res["confidence"]),
            class_id=np.array(res["class_id"]),
            data={"class_name": np.array(res["data"]["class_name"])},
        )
        if res
        else detections
    )


model = YOLO("cartonLat.pt")
image_paths = sorted(Path("testImgs/Pallet1").glob("*.jpg"), key=lambda x: int(x.stem))
num_boxes = 0

for i, image_path in enumerate(image_paths):
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    area = height * width
    results = model(image, conf=0.5)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[
        (detections.class_id == 0) & (detections.area / area > 0.011)
    ].with_nms(threshold=0.022)
    print(f"Num Detections: {len(detections)}")
    # print(f"{detections[:4]}")
    if detections:
        if i > 0 and i < 3:
            detections = drop_column(
                detections=detections, image_width=width, which_col="left"
            )
        elif i == 3:
            detections = drop_column(
                detections=detections, image_width=width, which_col="left"
            )
            detections = drop_column(
                detections=detections, image_width=width, which_col="right"
            )
        print(f"Num Detections Left: {len(detections)}")
        num_boxes += len(detections)
        print(f"Boxes So Far: {num_boxes}")
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections
        )
        cv2.imwrite(
            f"annotated_{SAME_COL_THRESH}_{image_path.stem}_{i}.jpg", annotated_image
        )
