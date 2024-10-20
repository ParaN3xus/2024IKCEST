import os
import re
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

IMAGES_DIR = '../jersey-number-pipeline/out/SoccerNetResults/crops/imgs/'
FINAL_RESULTS_FILE = '../jersey-number-pipeline/out/SoccerNetResults/final_results.json'
OUTPUT_FILE = '../jersey-number-pipeline/out/SoccerNetResults/player_team_mapping.json'
MAX_WORKERS = 8  # MAX WORKERS for ThreadPoolExecutor
FILENAME_PATTERN = re.compile(r'^SNMOT-(\d+_\d+)_(\d{6})\.jpg$')

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

def compute_average_color(pixels):
    if pixels is None or pixels.size == 0:
        return np.array([0, 0, 0])
    return np.mean(pixels, axis=0)

def identify_teams(average_colors, track_ids_list):
    valid_data = [(track_id, color) for track_id, color in zip(track_ids_list, average_colors) if not np.array_equal(color, [0,0,0])]
    
    if len(valid_data) < 2:
        print("Clustering failed: At least two valid player color data are required")
        return {}
    
    valid_track_ids, valid_colors = zip(*valid_data)
    valid_colors = np.array(valid_colors)
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(valid_colors)
    team_mapping = {}
    for track_id, label in zip(valid_track_ids, labels):
        team_mapping[track_id] = 'A' if label == 0 else 'B'
    
    return team_mapping

def main():
    if not os.path.exists(FINAL_RESULTS_FILE):
        print(f"Error: {FINAL_RESULTS_FILE} not found")
        return
    with open(FINAL_RESULTS_FILE, 'r', encoding='utf-8') as f:
        jersey_data = json.load(f)
    trackid_to_images = defaultdict(list)
    for filename in os.listdir(IMAGES_DIR):
        track_id, frame_id = parse_filename(filename)
        if track_id:
            filepath = os.path.join(IMAGES_DIR, filename)
            trackid_to_images[track_id].append(filepath)

    if not trackid_to_images:
        print(f"Error: Can not find any image files matching the pattern at {IMAGES_DIR} ")
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
            executor.submit(process_track_images, track_id, trackid_to_images[track_id]): track_id 
            for track_id in track_ids
        }
        for future in as_completed(future_to_track):
            track_id = future_to_track[future]
            try:
                result_track_id, avg_color = future.result()
                avg_color = avg_color if not np.isnan(avg_color).any() else np.array([0, 0, 0])
                average_colors[track_ids.index(result_track_id)] = avg_color
                print(f"Track ID: {result_track_id}, average color: {avg_color}")
            except Exception as e:
                print(f"An error occurred while processing Track ID {track_id}: {e}")

    team_mapping = identify_teams(average_colors, track_ids)

    final_output = {}
    for track_id, avg_color in zip(track_ids, average_colors):
        jersey_number = jersey_data.get(track_id, "Unknown")
        team = team_mapping.get(track_id, "Unknown")
        final_output[track_id] = {
            "jersey_number": jersey_number,
            "team": team
        }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    
    print(f"All tasks done! See output at: {OUTPUT_FILE}")