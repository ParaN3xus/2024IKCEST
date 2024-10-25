import os
import cv2
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_tracking_info(txt_file):
    """读取跟踪信息"""
    tracking_data = []
    logging.info(f'Reading tracking info from: {txt_file}')
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            frame, track_id, left, top, width, height, *_ = map(float, line.strip().split(','))
            tracking_data.append((int(frame), int(track_id), int(left), int(top), int(width), int(height)))
    return tracking_data

def draw_tracks_on_image(image, tracks, frame_id, scale_x, scale_y):
    """在图像上绘制跟踪框和ID，并应用缩放系数"""
    for track in tracks:
        if track[0] == frame_id:
            frame, track_id, left, top, width, height = track
            # 应用缩放系数
            left, top = int(left * scale_x), int(top * scale_y)
            width, height = int(width * scale_x), int(height * scale_y)
            cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)
            cv2.putText(image, f'ID: {track_id}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            logging.debug(f'Drew track ID {track_id} on frame {frame_id}')
    return image

def create_videos(tracking_dir, images_base_dir, output_dir, scale_x, scale_y):
    """读取跟踪信息并生成视频"""
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f'Created output directory: {output_dir}')
    
    folders = sorted(os.listdir(images_base_dir))
    
    for folder in folders:
        if not folder.startswith("SNMOT-"):
            continue
        
        tracking_file = os.path.join(tracking_dir, folder + ".txt" ) #+ ".txt"
        if not os.path.exists(tracking_file):
            logging.warning(f'Tracking file does not exist: {tracking_file}')
            continue
        
        logging.info(f'Processing folder: {folder}')
        tracks = get_tracking_info(tracking_file)
        images_dir = os.path.join(images_base_dir, folder, "img1")
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        if not image_files:
            logging.warning(f'No image files found in {images_dir}')
            continue
        
        # 获取视频尺寸
        first_image = cv2.imread(os.path.join(images_dir, image_files[0]))
        height, width, _ = first_image.shape
        
        video_path = os.path.join(output_dir, f'{folder}.mp4')
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        
        for image_file in image_files:
            frame_id = int(image_file.split('.')[0])
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path)
            image = draw_tracks_on_image(image, tracks, frame_id, scale_x, scale_y)
            video_writer.write(image)
            logging.debug(f'Processed frame {frame_id} for folder {folder}')
        
        video_writer.release()
        logging.info(f'Generated video: {video_path}')

# 参数设定
tracking_dir = 'GHOST/out/yolox_dets_OnTheFly:0_each_sample2:0.8:LastFrame:0.7LenThresh:0RemUnconf:0.0LastNFrames:10MM:1sum_0.8InactPat:50DetConf:0.45NewTrackConf:0.6'
images_base_dir = 'YOLOX/datasets/MOT20/test'
output_dir = 'output_videos'

# 你可以调整这些缩放参数
scale_x = 6.0  # 水平方向缩放系数
scale_y = 6.0  # 垂直方向缩放系数

create_videos(tracking_dir, images_base_dir, output_dir, scale_x, scale_y)
