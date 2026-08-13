import time

import numpy as np
from PIL import Image

from deployment.model_registry import registry


def preprocess_image(
    image,
    size
):
    image = Image.open(
        image
    ).convert("RGB")

    original_width, original_height = (
        image.size
    )

    image = image.resize(
        (size, size)
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array /= 255.0

    array = np.transpose(
        array,
        (2, 0, 1)
    )

    array = np.expand_dims(
        array,
        axis=0
    )

    return (
        array,
        original_width,
        original_height
    )


def softmax(
    x
):
    x = x - np.max(
        x,
        axis=1,
        keepdims=True
    )

    exp_x = np.exp(x)

    return (
        exp_x
        / np.sum(
            exp_x,
            axis=1,
            keepdims=True
        )
    )


def box_iou(
    box,
    boxes
):
    x1 = np.maximum(
        box[0],
        boxes[:, 0]
    )

    y1 = np.maximum(
        box[1],
        boxes[:, 1]
    )

    x2 = np.minimum(
        box[2],
        boxes[:, 2]
    )

    y2 = np.minimum(
        box[3],
        boxes[:, 3]
    )

    intersection = (
        np.maximum(
            0,
            x2 - x1
        )
        * np.maximum(
            0,
            y2 - y1
        )
    )

    area_box = (
        box[2] - box[0]
    ) * (
        box[3] - box[1]
    )

    area_boxes = (
        boxes[:, 2]
        - boxes[:, 0]
    ) * (
        boxes[:, 3]
        - boxes[:, 1]
    )

    union = (
        area_box
        + area_boxes
        - intersection
    )

    return intersection / (
        union + 1e-7
    )


def nms(
    boxes,
    scores,
    iou_threshold=0.45
):
    if len(boxes) == 0:
        return []

    boxes = np.asarray(
        boxes,
        dtype=np.float32
    )

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:

        index = order[0]

        keep.append(
            index
        )

        if len(order) == 1:
            break

        overlaps = box_iou(
            boxes[index],
            boxes[order[1:]]
        )

        order = order[
            1:
        ][
            overlaps < iou_threshold
        ]

    return keep


def decode_yolo_output(
    output,
    classes,
    image_width,
    image_height,
    input_size,
    confidence_threshold=0.25,
    iou_threshold=0.45
):
    predictions = output[0]

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    num_classes = len(
        classes
    )

    boxes = []

    scores = []

    class_ids = []

    for prediction in predictions:

        class_scores = prediction[
            4:
            4 + num_classes
        ]

        class_id = int(
            np.argmax(
                class_scores
            )
        )

        confidence = float(
            class_scores[class_id]
        )

        if confidence < confidence_threshold:
            continue

        cx, cy, width, height = (
            prediction[:4]
        )

        x1 = (
            cx - width / 2
        )

        y1 = (
            cy - height / 2
        )

        x2 = (
            cx + width / 2
        )

        y2 = (
            cy + height / 2
        )

        scale_x = (
            image_width
            / input_size
        )

        scale_y = (
            image_height
            / input_size
        )

        x1 *= scale_x
        x2 *= scale_x

        y1 *= scale_y
        y2 *= scale_y

        x1 = max(
            0,
            min(
                x1,
                image_width
            )
        )

        y1 = max(
            0,
            min(
                y1,
                image_height
            )
        )

        x2 = max(
            0,
            min(
                x2,
                image_width
            )
        )

        y2 = max(
            0,
            min(
                y2,
                image_height
            )
        )

        boxes.append(
            [
                x1,
                y1,
                x2,
                y2
            ]
        )

        scores.append(
            confidence
        )

        class_ids.append(
            class_id
        )

    final_indices = []

    for class_id in sorted(
        set(class_ids)
    ):

        class_indices = [
            index
            for index, value
            in enumerate(class_ids)
            if value == class_id
        ]

        class_boxes = [
            boxes[index]
            for index in class_indices
        ]

        class_scores = [
            scores[index]
            for index in class_indices
        ]

        keep = nms(
            class_boxes,
            class_scores,
            iou_threshold
        )

        final_indices.extend(
            [
                class_indices[index]
                for index in keep
            ]
        )

    detections = []

    for index in final_indices:

        detections.append(
            {
                "class_id": class_ids[index],
                "class": classes[
                    class_ids[index]
                ],
                "confidence": scores[index],
                "box": {
                    "x1": round(
                        boxes[index][0],
                        2
                    ),
                    "y1": round(
                        boxes[index][1],
                        2
                    ),
                    "x2": round(
                        boxes[index][2],
                        2
                    ),
                    "y2": round(
                        boxes[index][3],
                        2
                    )
                }
            }
        )

    detections.sort(
        key=lambda item: item[
            "confidence"
        ],
        reverse=True
    )

    return detections


def predict_detection(
    model_name,
    image_path,
    confidence_threshold=0.25,
    iou_threshold=0.45
):
    metadata = registry.metadata(
        model_name
    )

    session = registry.load(
        model_name
    )

    (
        tensor,
        image_width,
        image_height
    ) = preprocess_image(
        image_path,
        metadata["input_size"]
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    start = time.perf_counter()

    outputs = session.run(
        None,
        {
            input_name: tensor
        }
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    detections = decode_yolo_output(
        outputs[0],
        metadata["classes"],
        image_width,
        image_height,
        metadata["input_size"],
        confidence_threshold,
        iou_threshold
    )

    return {
        "model": model_name,
        "task": "detection",
        "image_width": image_width,
        "image_height": image_height,
        "detections": detections,
        "count": len(
            detections
        ),
        "latency_ms": round(
            latency_ms,
            3
        )
    }


def predict_classification(
    model_name,
    image_path
):
    metadata = registry.metadata(
        model_name
    )

    session = registry.load(
        model_name
    )

    (
        tensor,
        image_width,
        image_height
    ) = preprocess_image(
        image_path,
        metadata["input_size"]
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    start = time.perf_counter()

    outputs = session.run(
        None,
        {
            input_name: tensor
        }
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    probabilities = softmax(
        outputs[0]
    )[0]

    class_id = int(
        np.argmax(
            probabilities
        )
    )

    return {
        "model": model_name,
        "task": "classification",
        "class_id": class_id,
        "class": metadata[
            "classes"
        ][class_id],
        "confidence": float(
            probabilities[class_id]
        ),
        "probabilities": {
            name: float(
                probabilities[index]
            )
            for index, name
            in enumerate(
                metadata["classes"]
            )
        },
        "latency_ms": round(
            latency_ms,
            3
        )
    }


def predict(
    model_name,
    image_path,
    confidence_threshold=0.25,
    iou_threshold=0.45
):
    metadata = registry.metadata(
        model_name
    )

    if metadata["task"] == "detection":

        return predict_detection(
            model_name,
            image_path,
            confidence_threshold,
            iou_threshold
        )

    if metadata["task"] == "classification":

        return predict_classification(
            model_name,
            image_path
        )

    raise ValueError(
        f"Unsupported task: "
        f"{metadata['task']}"
    )