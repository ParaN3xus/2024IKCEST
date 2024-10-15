import cv2
import os
from pathlib import Path


def save_video_frames(video_stream):
    # 获取视频属性
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(video_stream.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

    # 生成递增的 SNMOT-xxx 目录名称
    base_path = Path("YOLOX", "datasets", "MOT20", "test")
    base_path.mkdir(parents=True, exist_ok=True)
    existing_dirs = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.startswith("SNMOT-")]
    next_num = max([int(d.split('-')[-1]) for d in existing_dirs], default=-1) + 1
    snmot_dir = base_path / f"SNMOT-{next_num:03d}"
    img1_dir = snmot_dir / "img1"

    # 创建目录
    img1_dir.mkdir(parents=True, exist_ok=True)

    # 初始化帧计数
    frame_count = 0

    # 处理视频帧并存储为图片
    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        # 保存当前帧为 jpg 文件
        img_path = img1_dir / f"{frame_count:06d}.jpg"
        cv2.imwrite(str(img_path), frame)

        frame_count += 1

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频资源
    video_stream.release()

    # 生成 seqinfo.ini 文件
    seqinfo_content = f"""[Sequence]
name=SNMOT-{next_num:03d}
imDir=img1
frameRate={frame_rate}
seqLength={min(frame_count, total_frames)}
imWidth={frame_width}
imHeight={frame_height}
imExt=.jpg"""

    # 写入 seqinfo.ini 文件
    seqinfo_path = snmot_dir / "seqinfo.ini"
    with open(seqinfo_path, "w") as f:
        f.write(seqinfo_content)

    print(f"Finished: {snmot_dir}")