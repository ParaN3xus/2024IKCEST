import cv2
import os
import json
import uuid
from paddleocr import PaddleOCR
from config import VideoProcessConfig, YOLOXConfig

MAX_FRAMES_PER_SEQUENCE = 750

def capture_frames(video_path, task_hash, output_base_dir=YOLOXConfig.DATASET_DIR):
    sequence_id = generate_sequence_id(output_base_dir)
    print(f"开始抓取视频帧: {video_path}，序列 ID: {sequence_id}")

    sequence_dir = create_new_sequence_directory(sequence_id, output_base_dir)
    img_dir = os.path.join(sequence_dir, "img1")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("打开视频文件失败。")

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 抓取第一帧进行OCR
    ret, first_frame = cap.read()
    if not ret:
        raise Exception("无法读取视频的第一帧。")
    
    # 提取比赛信息并保存到JSON
    extract_and_save_match_info(first_frame, task_hash)

    # 生成seqinfo.ini内容
    create_seqinfo_content(sequence_id, fps, total_frames, width, height, sequence_dir)

    frame_interval = max(int(fps * VideoProcessConfig.CAPTURE_EVERY), 1)
    frame_count = 1
    saved_frame_count = 1

    # 重置到第一帧开始抓取帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            if saved_frame_count > MAX_FRAMES_PER_SEQUENCE:
                sequence_id = generate_sequence_id(output_base_dir)
                sequence_dir = create_new_sequence_directory(sequence_id, output_base_dir)
                img_dir = os.path.join(sequence_dir, "img1")
                create_seqinfo_content(sequence_id, fps, total_frames, width, height, sequence_dir)
                saved_frame_count = 1

            frame_name = f"{saved_frame_count:06d}.jpg"
            output_file = os.path.join(img_dir, frame_name)
            cv2.imwrite(output_file, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"总共抓取到 {saved_frame_count - 1} 帧.")
    update_splits_file(output_base_dir)
    print("======清理视频======")
    os.remove(video_path)


def extract_and_save_match_info(frame, task_hash):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    time_left, time_top, time_width, time_height = 3352, 121, 293, 61
    team1_left, team1_top, team1_width, team1_height = 2450, 118, 336, 74
    team2_left, team2_top, team2_width, team2_height = 2896, 119, 330, 69

    time_roi = frame[time_top:time_top + time_height, time_left:time_left + time_width]
    team1_roi = frame[team1_top:team1_top + team1_height, team1_left:team1_left + team1_width]
    team2_roi = frame[team2_top:team2_top + team2_height, team2_left:team2_left + team2_width]

    def extract_text(roi_image):
        result = ocr.ocr(roi_image, cls=True)
        texts = []
        for line in result:
            for text_line in line:
                texts.append(text_line[1][0])
        return texts

    time_text = extract_text(time_roi)
    team1_text = extract_text(team1_roi)
    team2_text = extract_text(team2_roi)

    match_info = {
        "matchInfo": {
            "id": str(uuid.uuid4()),
            "start_time_stamp": time_text[0] if len(time_text) > 0 else "",
            "first_team": team1_text[0] if len(team1_text) > 0 else "",
            "first_team_score": team1_text[1] if len(team1_text) > 1 else "",
            "second_team": team2_text[1] if len(team2_text) > 1 else "",
            "second_team_score": team2_text[0] if len(team2_text) > 0 else "",
        }
    }

    output_dir = os.path.join('out', 'match')
    os.makedirs(output_dir, exist_ok=True)
    filename = task_hash + '.json'
    output_file = os.path.join(output_dir, filename)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(match_info, f, ensure_ascii=False, indent=4)

    print(f'比赛信息获取成功: {output_file}')
    print(json.dumps(match_info, ensure_ascii=False, indent=4))


def generate_sequence_id(output_base_dir):
    existing_ids = [
        int(d.split('-')[-1])
        for d in os.listdir(output_base_dir)
        if d.startswith('SNMOT-') and os.path.isdir(os.path.join(output_base_dir, d))
    ]
    new_id = max(existing_ids, default=0) + 1
    return f"SNMOT-{new_id:03d}"


def create_new_sequence_directory(sequence_id, base_dir):
    sequence_dir = os.path.join(base_dir, sequence_id)
    img_dir = os.path.join(sequence_dir, "img1")
    os.makedirs(img_dir, exist_ok=True)
    return sequence_dir


def create_seqinfo_content(sequence_id, fps, total_frames, width, height, sequence_dir):
    seqinfo_content = f"""[Sequence]
name={sequence_id}
imDir=img1
frameRate={fps}
seqLength={MAX_FRAMES_PER_SEQUENCE}
imWidth={width}
imHeight={height}
imExt=.jpg
"""
    seqinfo_path = os.path.join(sequence_dir, "seqinfo.ini")
    with open(seqinfo_path, 'w') as configfile:
        configfile.write(seqinfo_content)
    print(f"生成的seqinfo信息位于: {seqinfo_path}")


def update_splits_file(output_base_dir):
    test_seq_dirs = [
        d for d in os.listdir(output_base_dir)
        if d.startswith('SNMOT-') and os.path.isdir(os.path.join(output_base_dir, d))
    ]
    sequence_numbers = [
        int(d.split('-')[-1]) for d in test_seq_dirs
    ]
    if not sequence_numbers:
        print("未找到任何序列目录以更新 splits 文件。")
        return
    start = min(sequence_numbers)
    end = max(sequence_numbers) + 1
    splits_file_path = os.path.join("/home/pod/shared-nvme/2024IKCEST/GHOST/data/splits.py")

    if not os.path.exists(splits_file_path):
        print(f"未找到 splits 文件: {splits_file_path}")
        return

    with open(splits_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        if lines[i].startswith("test_seq_nums = "):
            lines[i] = f"test_seq_nums = list(range({start}, {end}))\n"
            break

    with open(splits_file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

    print(f"已更新 {splits_file_path}")
    