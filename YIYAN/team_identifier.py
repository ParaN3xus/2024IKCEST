import os
import re
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

IMAGES_DIR = "../jersey-number-pipeline/out/SoccerNetResults/crops/imgs/"
ORIGINAL_IMAGES_DIR = "../jersey-number-pipeline/data/SoccerNet/test/images/"
FINAL_RESULTS_FILE = "../jersey-number-pipeline/out/SoccerNetResults/final_results.json"
OUTPUT_FILE = "../jersey-number-pipeline/out/SoccerNetResults/player_team_mapping.json"
SOCCER_BALL_FILE = "../jersey-number-pipeline/out/SoccerNetResults/soccer_ball.json"
ghost_out_path = "../GHOST/out/yolox_dets_OnTheFly:0_each_sample2:0.8:LastFrame:0.7LenThresh:0RemUnconf:0.0LastNFrames:10MM:1sum_0.8InactPat:50DetConf:0.45NewTrackConf:0.6"
yolox_mot20_test_path = "../YOLOX/MOT20_dets1/test/"

MAX_WORKERS = 8  # MAX WORKERS for ThreadPoolExecutor
FILENAME_PATTERN = re.compile(r"^SNMOT-(\d+_\d+)_(\d{6})\.jpg$")
THRESHOLD = 100  # 区分类别的阈值


def get_first_file(directory):
    # 获取目录中的所有文件列表
    files = os.listdir(directory)
    # 过滤掉子目录，仅保留文件
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    # 检查是否存在文件
    if files:
        # 返回第一个文件的路径
        return os.path.join(directory, files[0])
    else:
        print("目录中没有文件")
        return None


def parse_filename(filename):
    match = FILENAME_PATTERN.match(filename)
    if match:
        track_id, frame_id = match.groups()
        return f"SNMOT-{track_id}", frame_id
    return None, None


def load_and_filter_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: {image_path} not found")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    white_threshold = 200
    white_mask = np.all(image_rgb >= white_threshold, axis=2)
    green_lower = np.array([0, 100, 0])
    green_upper = np.array([100, 255, 100])
    grass_mask = cv2.inRange(image_rgb, green_lower, green_upper) > 0
    combined_mask = white_mask | grass_mask
    filtered_pixels = image_rgb[~combined_mask]

    return filtered_pixels


# 本来想着是直接另开个函数的但是看不懂你们下面 filtered_pixels 是个什么东西而且我发现我这个函数里的东西跟你们下面实现的好像差不多所以就直接把 lower 和 upper 复制到你们代码里了
# def crop_green_background(image):
#     """Remove green areas from the image and save as PNG with transparency."""
#     # Convert the image from BGR to RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Define a range for green color in RGB (adjust as needed)
#     lower_green = np.array([0, 100, 0], dtype=np.uint8)
#     upper_green = np.array([150, 255, 150], dtype=np.uint8)

#     # Create a mask for green areas
#     mask = cv2.inRange(image_rgb, lower_green, upper_green)

#     image_rgb[mask != 0] = [0, 0, 0]

#     # Create an alpha channel (transparent where green is detected)
#     alpha_channel = np.ones(mask.shape, dtype=mask.dtype) * 255  # Full opacity
#     alpha_channel[mask != 0] = 0  # Make green areas transparent

#     # Create the 4-channel image (RGB + Alpha)
#     image_rgba = np.dstack((image_rgb, alpha_channel))

#     return cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA)


import matplotlib.pyplot as plt


def extract_center_average_colors(image_path):
    res = []
    for path in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, path))
        if image is None:
            print(f"Error: {image_path} not found")
            return np.array([0, 0, 0])
        
        # image = crop_green_background(image)

        # 裁剪中间部分：1/4 高度和 1/2 宽度
        h, w, _ = image.shape
        cropped_image = image[h // 4 : 2 * h // 4, w // 4 : 3 * w // 4]

        # 显示裁剪后的图像
        # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        # plt.title("Cropped Center Image")
        # plt.axis("off")
        # plt.show()

        # 去除草地颜色
        image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        green_lower = np.array([0, 100, 0], dtype=np.uint8)
        green_upper = np.array([150, 255, 150], dtype=np.uint8)
        grass_mask = cv2.inRange(image_rgb, green_lower, green_upper) > 0
        filtered_pixels = image_rgb[~grass_mask]

        # 显示去除绿色后的图像
        filtered_image = image_rgb.copy()
        filtered_image[grass_mask] = [0, 0, 0]  # 将草地像素设为黑色以便观察
        # plt.imshow(filtered_image)
        # plt.title("Image After Removing Green")
        # plt.axis("off")
        # plt.show()

        if filtered_pixels.size == 0:
            continue

        res.append(np.mean(filtered_pixels, axis=0))

    return res


def compute_average_color(pixels):
    if pixels is None or pixels.size == 0:
        return np.array([0, 0, 0])
    return np.mean(pixels, axis=0)


def load_csv_files_as_dict(directory_path, file_extension=".txt"):
    data_dict = {}
    # Loop through files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith(file_extension):
            # Extract file name without extension
            key = file_name.replace(file_extension, "")
            # Load CSV data
            file_path = os.path.join(directory_path, file_name)
            data_dict[key] = pd.read_csv(file_path, header=None)
    return data_dict


# Function to traverse subdirectories and load specific txt file in each 'det' folder
def load_nested_csv_files_as_dict(
    parent_directory, subfolder_name="det", target_file="yolox_dets.txt"
):
    data_dict = {}
    # Loop through each subdirectory in the parent directory
    for folder_name in os.listdir(parent_directory):
        folder_path = os.path.join(parent_directory, folder_name)
        det_folder_path = os.path.join(folder_path, subfolder_name)
        target_file_path = os.path.join(det_folder_path, target_file)

        # Check if the det folder and target file exist
        if os.path.isdir(det_folder_path) and os.path.isfile(target_file_path):
            # Load CSV data
            data_dict[folder_name] = pd.read_csv(target_file_path, header=None)
    return data_dict


def identify_teams(average_colors, track_ids_list):
    valid_data = [
        (track_id, color)
        for track_id, color in zip(track_ids_list, average_colors)
        if not np.array_equal(color, [0, 0, 0])
    ]

    if len(valid_data) < 2:
        print("Clustering failed: At least two valid player color data are required")
        return {}

    valid_track_ids, valid_colors = zip(*valid_data)
    valid_colors = np.array(valid_colors)
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(valid_colors)
    team_mapping = {}
    for track_id, label in zip(valid_track_ids, labels):
        team_mapping[track_id] = "A" if label == 0 else "B"

    return team_mapping


def get_class(track_id, ghost_dict, yolox_dict):
    dict_key, local_id = track_id.split("_")
    local_id = int(local_id)

    # 获取 GHOST 中指定键的 DataFrame
    ghost_df = ghost_dict[dict_key]
    # 在 GHOST DataFrame 中查找第二列等于 local_id 的第一行
    ghost_row = ghost_df[ghost_df.iloc[:, 1] == local_id].iloc[0]
    # 获取 GHOST 中该行的第 3, 4, 5, 6 列的值
    ghost_values = ghost_row.iloc[2:6].values

    # 获取 YOLOX 中指定键的 DataFrame
    yolox_df = yolox_dict[dict_key]
    # 在 YOLOX DataFrame 中查找第 3, 4, 5, 6 列完全匹配的行
    matching_yolox_row = yolox_df[
        (yolox_df.iloc[:, 2] - ghost_values[0] < 1e-3)
        & (yolox_df.iloc[:, 3] - ghost_values[1] < 1e-3)
        & (yolox_df.iloc[:, 4] - ghost_values[2] < 1e-3)
        & (yolox_df.iloc[:, 5] - ghost_values[3] < 1e-3)
    ].iloc[0]
    # 获取匹配行的第 9 列的值
    matching_value = matching_yolox_row.iloc[8]

    return matching_value


def classify_new_players(average_colors, new_colors, track_ids):
    ghost_data_dict = load_csv_files_as_dict(ghost_out_path)
    yolox_mot20_data_dict = load_nested_csv_files_as_dict(yolox_mot20_test_path)

    # 使用 identify_teams 构建 team_mapping
    team_mapping = identify_teams(
        average_colors, track_ids
    )  # 使用已有的identify_teams方法

    given_up = []

    # 初始化队伍A和队伍B的颜色列表
    team_a_colors = [
        average_colors[track_ids.index(tid)]
        for tid, team in team_mapping.items()
        if team == "A"
    ]
    team_b_colors = [
        average_colors[track_ids.index(tid)]
        for tid, team in team_mapping.items()
        if team == "B"
    ]

    avg_team_a_color = np.mean(team_a_colors, axis=0)
    avg_team_b_color = np.mean(team_b_colors, axis=0)

    notperson = 0

    valid_teams = {"A": [], "B": []}
    for track_id, colors in new_colors.items():
        # according to yolox, this is not a person
        classid = get_class(track_id, ghost_data_dict, yolox_mot20_data_dict)
        if int(classid) != 2:
            notperson = notperson + 1
            continue

        img_count = len(os.listdir(os.path.join(ORIGINAL_IMAGES_DIR, track_id)))
        if img_count <= 10:
            print(f"{track_id} skipped for too less img: {img_count}")
            continue

        vote_a = 0
        vote_b = 0

        for color in colors:
            closest_team = None
            diff_a = None
            diff_b = None
            min_diff = float("inf")

            # 计算新球员颜色与队伍A的平均颜色距离
            if team_a_colors:
                diff_a = np.linalg.norm(color - avg_team_a_color)
                if diff_a < THRESHOLD and diff_a < min_diff:
                    min_diff = diff_a
                    closest_team = "A"

            # 计算新球员颜色与队伍B的平均颜色距离
            if team_b_colors:
                diff_b = np.linalg.norm(color - avg_team_b_color)
                if diff_b < THRESHOLD and diff_b < min_diff:
                    min_diff = diff_b
                    closest_team = "B"

            # 根据距离最小的队伍对队伍进行投票
            if not closest_team:
                continue

            if closest_team == "A":
                vote_a = vote_a + 1
            else:
                vote_b = vote_b + 1

        closest_team = None
        if vote_a > vote_b and vote_a > img_count / 2:
            closest_team = "A"
        elif vote_a < vote_b and vote_b > img_count / 2:
            closest_team = "B"
        else:
            if vote_b != 0:
                print("WTF THEY ARE EQUAL")

        print(f"{ORIGINAL_IMAGES_DIR}{track_id}")
        print(f"vote: A: {vote_a}, B: {vote_b}, total: {img_count}")
        if closest_team:
            valid_teams[closest_team].append(track_id)

    for track, team in team_mapping.items():
        valid_teams[team].append(track)

    print(f"not person: {notperson}")

    return valid_teams, given_up


def main():
    if not os.path.exists(FINAL_RESULTS_FILE):
        print(f"Error: {FINAL_RESULTS_FILE} not found")
        return
    with open(FINAL_RESULTS_FILE, "r", encoding="utf-8") as f:
        jersey_data = json.load(f)

    with open(SOCCER_BALL_FILE, "r", encoding="utf-8") as f:
        ball_data = json.load(f)

    trackid_to_images = defaultdict(list)
    for filename in os.listdir(IMAGES_DIR):
        track_id, frame_id = parse_filename(filename)
        if track_id:
            filepath = os.path.join(IMAGES_DIR, filename)
            trackid_to_images[track_id].append(filepath)

    if not trackid_to_images:
        print(
            f"Error: Can not find any image files matching the pattern at {IMAGES_DIR} "
        )
        return

    track_ids = list(trackid_to_images.keys())
    average_colors = [np.array([0, 0, 0])] * len(track_ids)

    def process_track_images(track_id, image_paths):
        all_filtered_pixels = []
        for image_path in image_paths:
            pixels = load_and_filter_image(image_path)
            if pixels is not None and pixels.size > 0:
                all_filtered_pixels.append(pixels)

        if not all_filtered_pixels:
            print(f"Error: Track ID {track_id} doesn't have any valid pixels")
            return track_id, np.array([0, 0, 0])
        all_pixels = np.concatenate(all_filtered_pixels, axis=0)
        average_color = compute_average_color(all_pixels)
        return track_id, average_color

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_track = {
            executor.submit(
                process_track_images, track_id, trackid_to_images[track_id]
            ): track_id
            for track_id in track_ids
        }
        for future in as_completed(future_to_track):
            track_id = future_to_track[future]
            try:
                result_track_id, avg_color = future.result()
                avg_color = (
                    avg_color if not np.isnan(avg_color).any() else np.array([0, 0, 0])
                )
                average_colors[track_ids.index(result_track_id)] = avg_color
                print(f"Track ID: {result_track_id}, average color: {avg_color}")
            except Exception as e:
                print(f"An error occurred while processing Track ID {track_id}: {e}")

    existing_track_ids = set(track_ids)
    all_track_ids = set(jersey_data.keys())
    ball_tracks = set(ball_data["ball_tracks"])
    missing_track_ids = all_track_ids - existing_track_ids - ball_tracks

    new_average_colors = {}

    for track_id in missing_track_ids:
        if not track_id.startswith("SNMOT"):
            continue
        image_path = os.path.join(ORIGINAL_IMAGES_DIR, track_id)
        avg_colors = extract_center_average_colors(image_path)
        new_average_colors[track_id] = avg_colors

    final_team_mapping, unknown_team_player = classify_new_players(
        average_colors, new_average_colors, track_ids
    )

    final_output = {}
    for team in final_team_mapping:
        for track_id in final_team_mapping[team]:
            jersey_number = jersey_data.get(track_id, "Unknown")
            final_output[track_id] = {"jersey_number": str(jersey_number), "team": team}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    print(
        f"identified {len(final_output)}, total {len(missing_track_ids) + len(track_ids)}, unknown team {len(unknown_team_player)}"
    )
    print(f"All tasks done! See output at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
