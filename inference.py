from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import torch
from ultralytics import YOLO
import os


def inference_sahi(filepath_src, file_dir_out, filename_out, device = "cuda:0"):
    """
    :param filepath_src: строка содержащая путь до файла
    :param file_dir_out: строка содержащая путь до папки, в которую положится картинка с детекцией
    :param filename_out: строка с названием выходного файла (наша картинка с детекцией)
    :param device: "cuda:0" (в дефолте) или 'cpu'
    :param save: сохранять ли картинку в file_dir_out/filename_out
    :param hide_labels: скрывать название класса на изображении
    :param hide_conf: скрывать confidence на изображении
    """
    category_mapping = {"0": "person"}
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="best.pt",
        confidence_threshold=0.25,
        category_mapping=category_mapping,
        device=device,
    )
    result = get_sliced_prediction(
        filepath_src,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_match_metric='IOU',
        postprocess_type="NMS",
        postprocess_match_threshold=0.6,
    )

    txt_path = os.path.join(file_dir_out, filename_out + '.txt')

    img = cv2.imread(filepath_src)
    img_height, img_width = img.shape[:2]

    with open(txt_path, 'w') as f:
        for prediction in result.object_prediction_list:
            bbox = prediction.bbox
            x_center = (bbox.minx + bbox.maxx) / 2 / img_width
            y_center = (bbox.miny + bbox.maxy) / 2 / img_height
            width = (bbox.maxx - bbox.minx) / img_width
            height = (bbox.maxy - bbox.miny) / img_height

            class_id = 0

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
