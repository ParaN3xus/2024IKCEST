import os

class VideoProcessConfig:
    # 视频帧截取间隔（秒）
    CAPTURE_EVERY = 1/30

class PaddleDetectionConfig:
    MODEL_DIR = os.path.join(os.getcwd(), "video_process", "paddle_detection", "model")
    THRESHOLD = 0.5  # 预测阈值

class YOLOXConfig:
    DATASET_DIR = os.path.join(os.getcwd(), "YOLOX", "datasets", "MOT20", "test")
    SEQUENCE_PREFIX = "SNMOT-"
