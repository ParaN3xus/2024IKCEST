from video_process.video_capture import capture_frames
from gen_event_flow import *
from team_identifier import *

import os
import sys
import shutil
import csv
import threading
import subprocess
import concurrent.futures
from PIL import Image

# 前面一个需要绝对路径，后面那个用相对路径就行
links = [
        ("/path/to/YOLOX/datasets/MOT20_dets1/test/", "../GHOST/datasets/detections_GHOST/MOT20/test"),
        ("/path/to/YOLOX/datasets/MOT20/test/", "../GHOST/datasets/MOT20/test")
    ]
csv_folder = "../GHOST/out/scaled" # GHOST的放大结果
images_base_folder = "../jersey-number-pipeline/data/SoccerNet/test/images"
videos_base_folder = "../YOLOX/datasets/MOT20/test"
task_file = 'tasks/tasks.json'


def create_symlink(source_path, target_path):
    if os.path.islink(target_path):
        print("文件已存在，跳过")
    else:
        os.symlink(source_path, target_path)
        print(f"符号链接已创建，从 {source_path} 到 {target_path}")

def run_command(command, cwd=None):
    def read_output(pipe, write_func):
        try:
            for line in iter(pipe.readline, ''):
                write_func(line)
        finally:
            pipe.close()

    try:
        process = subprocess.Popen(
            f"eval \"$(conda shell.bash hook)\" && {command}",
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, sys.stdout.write))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, sys.stderr.write))

        stdout_thread.start()
        stderr_thread.start()

        stdout_thread.join()
        stderr_thread.join()

        process.wait()

        if process.returncode != 0:
            sys.stderr.write(f"Command '{command}' failed with return code {process.returncode}\n")

    except Exception as e:
        sys.stderr.write(f"An error occurred while running the command '{command}': {str(e)}\n")

def run_yolox_detection():
    command = f"conda activate YOLOX && python get_dets.py image -f my.py -c best_ckpt.pth --conf 0.25 --nms 0.45 --device gpu"
    run_command(command, cwd="../YOLOX")

def run_ghost():
    command = "conda activate GHOST && bash scripts/main_20.sh"
    run_command(command, cwd="../GHOST")
    command = "conda activate GHOST && python rename.py"
    run_command(command, cwd="../GHOST")
    command = "conda activate GHOST && python rescale.py"
    run_command(command, cwd="../GHOST")

def run_jersey_number_pipeline():
    command = "conda activate YOLOX && python3 main.py SoccerNet test"
    run_command(command, cwd="../jersey-number-pipeline")


def extract_images(csv_folder, images_base_folder, videos_base_folder, csv_file):
    video_name = csv_file.split('.')[0]
    csv_path = os.path.join(csv_folder, csv_file)
    
    tasks = []
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            frame, track_id, left, top, width, height, _, _, _, _ = row
            tasks.append((videos_base_folder, images_base_folder, video_name, frame, track_id, left, top, width, height))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda args: crop_image(*args), tasks)

def crop_image(videos_base_folder, images_base_folder, video_name, frame, track_id, left, top, width, height):
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

def extract_csv():
    tasks = []
    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith('.txt'):
            tasks.append((csv_folder, images_base_folder, videos_base_folder, csv_file))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda args: extract_images(*args), tasks)

def run(video_path, task_hash):
    print("======开始抽帧======")
    capture_frames(video_path, task_hash)
    print("======开始执行YOLOX检测======")
    run_yolox_detection()
    print("======创建符号链接======")
    for source_path, target_path in links:
        create_symlink(source_path, target_path)
    print("======开始执行GHOST追踪======")
    run_ghost()
    print("======开始裁切图片======")
    extract_csv()
    print("======开始执行球衣号码识别======")
    run_jersey_number_pipeline()
    print("======开始分类球员信息======")
    identify_teams()
    print("======开始生成事件流======")
    gen_event_seq(task_hash)
    print("======任务完成，开始清理文件======")
    with open('tasks_file.json', 'r') as f:
        task_data = json.load(f)
    for task in task_data['tasks']:
        if task['id'] == task_hash:
            task['status'] = 'completed'
            break
    with open('tasks_file.json', 'w') as f:
        json.dump(task_data, f, indent=4)
    shutil.rmtree('../YOLOX/datasets/MOT20/test/')
    shutil.rmtree('../YOLOX/datasets/MOT20_dets1/test/')
    shutil.rmtree('../GHOST/out/')
    shutil.rmtree('../GHOST/datasets/detections_GHOST/MOT20/test/')
    shutil.rmtree('../jersey-number-pipeline/data/SoccerNet/test/images/')
    shutil.rmtree('../jersey-number-pipeline/out/SoccerNetResults/')
    print("======清理完成======")
    

    