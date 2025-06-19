from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import torch
from ultralytics import YOLO


def inference_sahi(filepath_src, file_dir_out, filename_out, device = "cuda:0", save=False):
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

    if save:
        result.export_visuals(
            export_dir=file_dir_out,
            file_name=filename_out,
            hide_labels=True,
            hide_conf=True,
            rect_th=2,
        )
