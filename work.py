from video_download.get_web_video import *
from video_process.video_capture import *

import os
import shutil
import asyncio
import csv
from PIL import Image

links = [
        ("YOLOX/datasets/MOT20_dets1/test/", "GHOST/datasets/detections_GHOST/MOT20/test/"),
        ("YOLOX/datasets/MOT20/test/", "GHOST/datasets/MOT20/test/")
    ]

async def create_symlink(source_path, base_target_path, link_name):
    target_path = os.path.join(base_target_path, link_name)
    
    os.makedirs(base_target_path, exist_ok=True)
    
    if os.path.exists(target_path) or os.path.islink(target_path):
        print(f"Link or file {target_path} already exists.")
    else:
        os.symlink(source_path, target_path)
        print(f"Link created from {source_path} to {target_path}")

async def run_command(command, cwd=None):
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"{command}\nerr: {stderr.decode()}")
    return stdout.decode()

async def run_yolox_detection():
    command = f"python get_dets.py image -f my.py -c best_ckpt.pth --conf 0.25 --nms 0.45 --device gpu"
    await run_command(command, cwd="YOLOX")

async def run_ghost():
    command = "bash scripts/main_20.sh"
    await run_command(command, cwd="GHOST")
    command = "python rename.py"
    await run_command(command, cwd="GHOST")
    command = "python rescale.py"
    await run_command(command, cwd="GHOST")
    
async def extract_images(csv_folder, images_base_folder, videos_base_folder, csv_file):
    video_name = csv_file.split('.')[0]
    csv_path = os.path.join(csv_folder, csv_file)
    
    # Read CSV file and create tasks for cropping and saving images
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        tasks = [
            save_image(videos_base_folder, images_base_folder, video_name, frame, track_id, left, top, width, height)
            for frame, track_id, left, top, width, height, _, _, _, _ in reader
        ]
        
    await asyncio.gather(*tasks)

async def save_image(videos_base_folder, images_base_folder, video_name, frame, track_id, left, top, width, height):
    frame = int(frame)
    track_id = int(track_id)
    left = float(left)
    top = float(top)
    width = float(width)
    height = float(height)
    
    img_folder = f"{images_base_folder}/{video_name}_{track_id}"
    os.makedirs(img_folder, exist_ok=True)

    image_path = f"{videos_base_folder}/{video_name}/img1/{frame:06d}.jpg"
    save_path = f"{img_folder}/{video_name}_{track_id}_{frame:06d}.jpg"

    if os.path.exists(image_path):
        image = Image.open(image_path)
        cropped_image = image.crop((left, top, left + width, top + height))
        cropped_image.save(save_path)
    else:
        print(f"Image {image_path} does not exist.")

async def extract_csv():
    csv_folder = "GHOST/out/scaled"
    images_base_folder = "jersey-number-pipeline/data/SoccerNet/test/images"
    videos_base_folder = "YOLOX/datasets/MOT20/test"

    tasks = [
        extract_images(csv_folder, images_base_folder, videos_base_folder, csv_file)
        for csv_file in os.listdir(csv_folder) if csv_file.endswith('.txt')
    ]
    
    await asyncio.gather(*tasks)

async def run(url: str):
    video_path = await download_video(url)
    capture_frames(video_path)
    await run_yolox_detection()

    for _, link_name in links:
        os.makedirs(os.path.dirname(link_name), exist_ok=True)
    await asyncio.gather(*(create_symlink(target, link_name, "test") for target, link_name in links))

    await run_ghost()
    await extract_csv()
    


