import cv2
import os
from config import VideoProcessConfig, YOLOXConfig

def capture_frames(video_path, output_base_dir=YOLOXConfig.DATASET_DIR):
    sequence_id = generate_sequence_id(output_base_dir)
    print(f"开始抓取视频帧: {video_path}，序列 ID: {sequence_id}")

    sequence_dir = os.path.join(output_base_dir, sequence_id)
    img_dir = os.path.join(sequence_dir, "img1")
    os.makedirs(img_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("打开视频文件失败。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    seqinfo_content = f"""[Sequence]
name={sequence_id}
imDir=img1
frameRate={fps}
seqLength={total_frames}
imWidth={width}
imHeight={height}
imExt=.jpg
"""

    seqinfo_path = os.path.join(sequence_dir, "seqinfo.ini")
    with open(seqinfo_path, 'w') as configfile:
        configfile.write(seqinfo_content)
    print(f"生成的seqinfo信息位于: {seqinfo_path}")

    # 开始抓取帧
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("打开视频文件失败。")

    frame_interval = max(int(fps * VideoProcessConfig.CAPTURE_EVERY), 1)
    frame_count = 0
    saved_frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"{saved_frame_count:06d}.jpg"
            output_file = os.path.join(img_dir, frame_name)
            cv2.imwrite(output_file, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"总共抓取到 {saved_frame_count - 1} 帧.")

def generate_sequence_id(output_base_dir):
    test_dir = os.path.join(output_base_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    existing_ids = [
        int(d.split('-')[-1])
        for d in os.listdir(test_dir)
        if d.startswith('SNMOT-') and os.path.isdir(os.path.join(test_dir, d))
    ]
    new_id = max(existing_ids, default=0) + 1
    return f"SNMOT-{new_id:03d}"
