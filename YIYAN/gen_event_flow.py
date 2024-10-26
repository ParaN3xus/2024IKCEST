import pandas as pd
import json
import numpy as np
import math
import os
import re
import glob
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("event_reconstruction_debug.log"),
        logging.StreamHandler()
    ]
)

# 参数
MAX_DISTANCE = 100
MAX_GAP = 2
SMALL_WINDOW = 5
ANGLE_THRESHOLD = 45
POSITION_THRESHOLD = 30
POST_HEADER_FRAMES = 5
MIN_VISIBLE_FRAMES = 2
DISTANCE_THRESHOLD = 200
MIN_FALL_FRAMES = 30
CARRY_DISTANCE = 100
CARRY_CONSISTENCY = 15
MAX_ID_CHANGES = 2
CARRY_DISTANCE_PASS = 200
CONFIDENCE_FRAMES = 5
DURATION_SECONDS = 3
FRAME_RATE = 30
MISSING_THRESHOLD = 0.9
ANGLE_THRESHOLD_VERTICAL = 30
X_THRESHOLD_VERTICAL = 20
MIN_SHOT_FRAMES_AROUND = 15
TRACKING_FILES_PATH = '/home/pod/shared-nvme/2024IKCEST/GHOST/out/scaled/'
BALL_TRACKS_PATH = '/home/pod/shared-nvme/2024IKCEST/jersey-number-pipeline/out/SoccerNetResults/soccer_ball.json'
PLAYER_INFO_PATH = '/home/pod/shared-nvme/2024IKCEST/jersey-number-pipeline/out/SoccerNetResults/player_team_mapping.json'
IMAGE_FOLDER = '/home/pod/shared-nvme/2024IKCEST/GHOST/datasets/MOT20/test/'

def convert_time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(":"))
    return minutes * 60 + seconds

def calculate_timestamp(frame_number, start_time_seconds, fps=30):
    total_seconds = start_time_seconds + frame_number / fps
    minutes, seconds = divmod(total_seconds, 60)
    milliseconds = (total_seconds - int(total_seconds)) * 1000
    return f"{int(minutes)}:{int(seconds):02d}.{int(milliseconds):03d}"

def merge_events(events):
    def parse_timestamp(ts):
        mins, secs = map(float, ts.split(':'))
        return int(mins * 60 * 1000 + secs * 1000)
    
    def merge_similar_events(events, time_threshold_ms):
        merged_events = []
        last_event = None
        
        for event in events:
            if last_event and event['type'] == last_event['type'] and event['team'] == last_event['team'] and event['player'] == last_event['player']:
                current_time = parse_timestamp(event['timestamp'])
                last_time = parse_timestamp(last_event['timestamp'])
                if current_time - last_time <= time_threshold_ms:
                    continue
            
            if last_event:
                merged_events.append(last_event)
            
            last_event = event
        if last_event:
            merged_events.append(last_event)

        return merged_events

    events.sort(key=lambda x: parse_timestamp(x['timestamp']))
    merged_events = merge_similar_events(events, time_threshold_ms=333)
    return merged_events

def load_csv_multiple(tracking_files_path):
    all_files = sorted(glob.glob(os.path.join(tracking_files_path, 'SNMOT-*.txt')))
    combined_data = []
    global_frame_offset = 0
    for file in all_files:
        filename = os.path.basename(file)
        file_id = filename.split('.')[0].split('-')[-1]
        df = pd.read_csv(file, header=None, names=['frame', 'track_id', 'left', 'top', 'width', 'height', 'ignore1', 'ignore2', 'ignore3', 'ignore4'])
        df['frame'] = df['frame'] + global_frame_offset
        df['track_id'] = df['track_id'].apply(lambda x: f"{file_id}_{x}")
        combined_data.append(df)
        global_frame_offset += df['frame'].max()
    combined_df = pd.concat(combined_data, ignore_index=True)
    return combined_df

def load_ball_tracks(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    ball_tracks = [track.split('-')[1] for track in data.get('ball_tracks', [])]
    return [f"{track_id}" for track_id in ball_tracks]

def load_player_info(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    player_info = {}
    for key, value in data.items():
        parts = key.split('_')
        if len(parts) < 2:
            continue
        file_id = parts[0].split('-')[-1]
        track_id = parts[1]
        combined_track_id = f"{file_id}_{track_id}"
        player_info[combined_track_id] = value
    return player_info

def build_frame_dict(df):
    frame_dict = {}
    for _, row in df.iterrows():
        frame = int(row['frame'])
        track_id = row['track_id']
        bbox = (float(row['left']), float(row['top']),
                float(row['width']), float(row['height']))
        if frame not in frame_dict:
            frame_dict[frame] = []
        frame_dict[frame].append({'track_id': track_id, 'bbox': bbox})
    return frame_dict

def calculate_center(bbox):
    left, top, width, height = bbox
    center_x = left + width / 2
    center_y = top + height / 2
    return np.array([center_x, center_y])

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def angle_between_vectors(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    unit_v1 = v1 / norm_v1
    unit_v2 = v2 / norm_v2
    dot_product = np.dot(unit_v1, unit_v2)
    dot_product = max(min(dot_product, 1.0), -1.0)
    angle_rad = math.acos(dot_product)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.x = np.array([0, 0, 0, 0])
    kf.P *= 1000.
    kf.Q = Q_discrete_white_noise(dim=4, dt=1, var=0.1)
    kf.R = np.array([[5, 0],
                     [0, 5]])
    return kf

def confirm_ball_tracks_with_kalman(frame_dict, ball_track_ids, max_distance=MAX_DISTANCE, max_gap=MAX_GAP):
    all_frames = sorted(frame_dict.keys())
    ball_tracks_confirmed = []
    last_position = None
    last_frame = None
    gaps = 0
    kf = initialize_kalman_filter()
    is_kalman_initialized = False

    for frame in all_frames:
        detections = frame_dict[frame]
        current_balls = [det for det in detections if det['track_id'] in ball_track_ids]

        if current_balls:
            if last_position is not None:
                distances = [distance(calculate_center(det['bbox']), last_position) for det in current_balls]
                min_dist = min(distances)
                min_idx = distances.index(min_dist)
                if min_dist <= max_distance:
                    confirmed_track_id = current_balls[min_idx]['track_id']
                    confirmed_position = calculate_center(current_balls[min_idx]['bbox'])
                    ball_tracks_confirmed.append({'frame': frame, 'track_id': confirmed_track_id, 'position': confirmed_position, 'predicted': False})

                    if not is_kalman_initialized:
                        kf.x[:2] = confirmed_position
                        is_kalman_initialized = True
                    kf.predict()
                    kf.update(confirmed_position)
                    last_position = confirmed_position
                    last_frame = frame
                    gaps = 0
                else:
                    gaps += 1
            else:
                confirmed_track_id = current_balls[0]['track_id']
                confirmed_position = calculate_center(current_balls[0]['bbox'])
                ball_tracks_confirmed.append({'frame': frame, 'track_id': confirmed_track_id, 'position': confirmed_position, 'predicted': False})

                kf.x[:2] = confirmed_position
                kf.predict()
                kf.update(confirmed_position)
                last_position = confirmed_position
                last_frame = frame
                gaps = 0
                is_kalman_initialized = True
        else:
            gaps += 1
            if gaps <= max_gap and is_kalman_initialized:
                kf.predict()
                predicted_position = kf.x[:2]
                ball_tracks_confirmed.append({'frame': frame, 'track_id': None, 'position': predicted_position, 'predicted': True})
                last_position = predicted_position
                last_frame = frame
            else:
                kf = initialize_kalman_filter()
                last_position = None
                last_frame = None
                gaps = 0
                is_kalman_initialized = False

    return ball_tracks_confirmed

def detect_headers(ball_tracks_confirmed, player_info, frame_dict, small_window=SMALL_WINDOW, angle_threshold=ANGLE_THRESHOLD, position_threshold=POSITION_THRESHOLD, post_header_frames=POST_HEADER_FRAMES, min_visible_frames=MIN_VISIBLE_FRAMES):
    headers = []
    ball_tracks_sorted = sorted(ball_tracks_confirmed, key=lambda x: x['frame'])
    frames = [bt['frame'] for bt in ball_tracks_sorted]
    positions = [bt['position'] for bt in ball_tracks_sorted]

    for i, bt in enumerate(ball_tracks_sorted):
        current_frame = bt['frame']
        current_position = bt['position']
        predicted = bt['predicted']

        if not predicted:
            detections = frame_dict.get(current_frame, [])
            ball_detections_real = [det for det in detections if det['track_id'] == bt['track_id']]
            if ball_detections_real:
                ball_bbox = ball_detections_real[0]['bbox']
                ball_top = ball_bbox[1]
                ball_left = ball_bbox[0]
                ball_right = ball_bbox[0] + ball_bbox[2]
            else:
                continue
        else:
            continue

        nearest_player_id = None
        min_distance = float('inf')
        for player_id, info in player_info.items():
            player_detections = [det for det in frame_dict.get(current_frame, []) if det['track_id'] == player_id]
            if not player_detections:
                continue
            player_bbox = player_detections[0]['bbox']
            player_center = calculate_center(player_bbox)
            player_top = player_bbox[1]
            player_left_player = player_bbox[0]
            player_right_player = player_bbox[0] + player_bbox[2]

            condition1 = (ball_top <= player_top and player_top - ball_top <= position_threshold) or (ball_top >= player_top and ball_top - player_top <= position_threshold)
            condition2 = (ball_left >= player_left_player) and (ball_right <= player_right_player)

            if condition1 and condition2:
                dist = distance(current_position, player_center)
                if dist < min_distance:
                    min_distance = dist
                    nearest_player_id = player_id

        if nearest_player_id is not None:
            window_start = max(0, i - small_window)
            window_end = min(len(ball_tracks_sorted), i + small_window + 1)
            window_positions = positions[window_start:window_end]

            if len(window_positions) >= 2:
                movement_vector = window_positions[-1] - window_positions[0]
                if np.linalg.norm(movement_vector) == 0:
                    continue
                prev_window_start = max(0, window_start - small_window)
                prev_window_end = window_start
                prev_window_positions = positions[prev_window_start:prev_window_end]
                if len(prev_window_positions) >= 2:
                    prev_movement_vector = prev_window_positions[-1] - prev_window_positions[0]
                    angle = angle_between_vectors(prev_movement_vector, movement_vector)

                    if angle >= angle_threshold:
                        post_frames_start = i + 1
                        post_frames_end = min(len(ball_tracks_sorted), i + post_header_frames + 1)
                        visible_count = sum([not ball_tracks_sorted[j]['predicted'] for j in range(post_frames_start, post_frames_end)])

                        if visible_count >= min_visible_frames:
                            if movement_vector[0] > 0:
                                direction = 'right'
                            else:
                                direction = 'left'

                            already_recorded = any(header['frame'] == current_frame and header['player_track_id'] == nearest_player_id for header in headers)
                            if not already_recorded:
                                headers.append({
                                    'frame': current_frame,
                                    'player_track_id': nearest_player_id,
                                    'direction': direction
                                })
    return headers

def detect_falls(frame_dict, player_info, distance_threshold=DISTANCE_THRESHOLD, min_fall_frames=MIN_FALL_FRAMES):
    falls = []
    all_frames = sorted(frame_dict.keys())
    player_states = {player_id: {'falling': False, 'fall_frames': 0} for player_id in player_info.keys()}

    for frame in all_frames:
        detections = frame_dict[frame]
        players = [det for det in detections if det['track_id'] in player_info]
        player_centers = {det['track_id']: calculate_center(det['bbox']) for det in players}
        player_bboxes = {det['track_id']: det['bbox'] for det in players}

        for player_id, center in player_centers.items():
            bbox = player_bboxes[player_id]
            left, top, width, height = bbox

            if width > height:
                player_states[player_id]['fall_frames'] += 1
                if player_states[player_id]['fall_frames'] >= min_fall_frames and not player_states[player_id]['falling']:
                    close_players = []
                    for other_id, other_center in player_centers.items():
                        if other_id == player_id:
                            continue
                        dist = distance(center, other_center)
                        if dist <= distance_threshold:
                            close_players.append(other_id)

                    if len(close_players) >= 1:
                        player = player_info[player_id]
                        jersey_number = player.get('jersey_number', 'Unknown')
                        team = player.get('team', 'Unknown')

                        contest_players = []
                        for cp_id in close_players:
                            cp_info = player_info.get(cp_id, {})
                            contest_players.append({
                                'player_track_id': cp_id,
                                'jersey_number': cp_info.get('jersey_number', 'Unknown'),
                                'team': cp_info.get('team', 'Unknown')
                            })

                        falls.append({
                            'frame': frame,
                            'player_track_id': player_id,
                            'jersey_number': jersey_number,
                            'team': team,
                            'contest_players': contest_players
                        })

                        player_states[player_id]['falling'] = True
            else:
                if player_states[player_id]['falling']:
                    player_states[player_id]['falling'] = False
                player_states[player_id]['fall_frames'] = 0

    return falls

def detect_ball_carries(ball_tracks_confirmed, player_info, frame_dict, carry_distance=CARRY_DISTANCE, carry_consistency=CARRY_CONSISTENCY, max_id_changes=MAX_ID_CHANGES):
    carries = []
    carry_tracker = {}

    for bt in ball_tracks_confirmed:
        frame = bt['frame']
        ball_pos = bt['position']
        track_id = bt['track_id']

        if track_id is None:
            continue

        for player_id, info in player_info.items():
            player_detections = frame_dict.get(frame, [])
            player_det = next((det for det in player_detections if det['track_id'] == player_id), None)
            if not player_det:
                continue
            player_bbox = player_det['bbox']
            left, top, width, height = player_bbox
            left_bottom = np.array([left, top + height])
            right_bottom = np.array([left + width, top + height])
            dist_left = distance(ball_pos, left_bottom)
            dist_right = distance(ball_pos, right_bottom)

            if dist_left <= carry_distance or dist_right <= carry_distance:
                if player_id not in carry_tracker:
                    carry_tracker[player_id] = {'count': 1, 'last_frame': frame, 'id_changes': 0, 'active': True}
                else:
                    if frame == carry_tracker[player_id]['last_frame'] + 1:
                        carry_tracker[player_id]['count'] += 1
                        carry_tracker[player_id]['last_frame'] = frame
                    else:
                        carry_tracker[player_id]['id_changes'] += 1
                        if carry_tracker[player_id]['id_changes'] <= max_id_changes:
                            carry_tracker[player_id]['count'] += 1
                            carry_tracker[player_id]['last_frame'] = frame
                        else:
                            carry_tracker[player_id] = {'count': 1, 'last_frame': frame, 'id_changes': 0, 'active': True}

                if carry_tracker[player_id]['count'] >= carry_consistency:
                    player = player_info[player_id]
                    jersey_number = player.get('jersey_number', 'Unknown')
                    team = player.get('team', 'Unknown')

                    overlapping = False
                    for existing in carries:
                        if existing['player_track_id'] == player_id and \
                           not (frame - carry_tracker[player_id]['count'] + 1 > existing['end_frame'] or
                                frame < existing['start_frame']):
                            existing['end_frame'] = frame
                            overlapping = True
                            break

                    if not overlapping:
                        carries.append({
                            'start_frame': frame - carry_tracker[player_id]['count'] + 1,
                            'end_frame': frame,
                            'player_track_id': player_id,
                            'jersey_number': jersey_number,
                            'team': team
                        })

                    carry_tracker[player_id]['count'] = 0
                    carry_tracker[player_id]['active'] = False

    merged_carries = []
    carries_sorted = sorted(carries, key=lambda x: (x['player_track_id'], x['start_frame']))
    for carry in carries_sorted:
        if not merged_carries:
            merged_carries.append(carry)
            continue
        last = merged_carries[-1]
        if carry['player_track_id'] == last['player_track_id'] and carry['start_frame'] <= last['end_frame'] + 1:
            last['end_frame'] = max(last['end_frame'], carry['end_frame'])
        else:
            merged_carries.append(carry)

    final_carries = []
    for carry in merged_carries:
        overlap = False
        for final in final_carries:
            if carry['player_track_id'] != final['player_track_id']:
                if not (carry['end_frame'] < final['start_frame'] or carry['start_frame'] > final['end_frame']):
                    overlap = True
                    break
        if not overlap:
            final_carries.append(carry)

    return final_carries

def detect_passes(ball_tracks_confirmed, player_info, frame_dict, carry_distance=CARRY_DISTANCE_PASS, confidence_frames=CONFIDENCE_FRAMES):
    passes = []
    current_possessor = None
    possessor_start_frame = None
    new_possessor_candidate = None
    candidate_count = 0

    for bt in ball_tracks_confirmed:
        frame = bt['frame']
        ball_pos = bt['position']
        track_id = bt['track_id']

        if track_id is None:
            possessor = None
        else:
            possessor = None
            min_distance = float('inf')
            for player_id, info in player_info.items():
                player_detections = frame_dict.get(frame, [])
                player_det = next((det for det in player_detections if det['track_id'] == player_id), None)
                if not player_det:
                    continue
                player_bbox = player_det['bbox']
                left, top, width, height = player_bbox
                left_bottom = np.array([left, top + height])
                right_bottom = np.array([left + width, top + height])
                dist_left = distance(ball_pos, left_bottom)
                dist_right = distance(ball_pos, right_bottom)

                if dist_left <= carry_distance or dist_right <= carry_distance:
                    if dist_left < min_distance or dist_right < min_distance:
                        min_distance = min(dist_left, dist_right)
                        possessor = player_id

            if possessor is None:
                possessor = None

        if possessor != current_possessor:
            if possessor != new_possessor_candidate:
                new_possessor_candidate = possessor
                candidate_count = 1
            else:
                candidate_count += 1

            if new_possessor_candidate is not None and candidate_count >= confidence_frames:
                if current_possessor is not None and new_possessor_candidate != current_possessor:
                    team_current = player_info[current_possessor]['team']
                    team_new = player_info[new_possessor_candidate]['team']
                    if team_current == team_new:
                        passes.append({
                            'from_player_id': current_possessor,
                            'to_player_id': new_possessor_candidate,
                            'from_jersey_number': player_info[current_possessor].get('jersey_number', 'Unknown'),
                            'to_jersey_number': player_info[new_possessor_candidate].get('jersey_number', 'Unknown'),
                            'team': team_current,
                            'start_frame': possessor_start_frame if possessor_start_frame else frame - confidence_frames + 1,
                            'end_frame': frame
                        })
                current_possessor = new_possessor_candidate
                possessor_start_frame = frame - confidence_frames + 1
                new_possessor_candidate = None
                candidate_count = 0
        else:
            new_possessor_candidate = None
            candidate_count = 0

    merged_passes = []
    passes_sorted = sorted(passes, key=lambda x: x['start_frame'])
    for p in passes_sorted:
        if not merged_passes:
            merged_passes.append(p)
            continue
        last = merged_passes[-1]
        if (p['from_player_id'] == last['from_player_id'] and
            p['to_player_id'] == last['to_player_id'] and
            p['start_frame'] <= last['end_frame'] + 1):
            last['end_frame'] = max(last['end_frame'], p['end_frame'])
        else:
            merged_passes.append(p)

    return merged_passes

def detect_intense_periods(ball_tracks_confirmed, frame_dict, duration_seconds=DURATION_SECONDS, frame_rate=FRAME_RATE, missing_threshold=MISSING_THRESHOLD):
    intense_periods = []
    required_frames = duration_seconds * frame_rate
    missing_required = int(required_frames * missing_threshold)
    frame_presence = {}

    for frame in frame_dict.keys():
        frame_presence[frame] = False

    for bt in ball_tracks_confirmed:
        frame = bt['frame']
        track_id = bt['track_id']
        predicted = bt.get('predicted', False)
        if track_id is not None and not predicted:
            frame_presence[frame] = True

    all_frames = sorted(frame_presence.keys())
    total_frames = len(all_frames)

    for start in range(total_frames - required_frames + 1):
        window_frames = all_frames[start:start + required_frames]
        missing_frames = sum(not frame_presence[f] for f in window_frames)

        if missing_frames >= missing_required:
            start_frame = window_frames[0]
            end_frame = window_frames[-1]

            if intense_periods and start_frame <= intense_periods[-1]['end_frame'] + 1:
                intense_periods[-1]['end_frame'] = end_frame
            else:
                intense_periods.append({'start_frame': start_frame, 'end_frame': end_frame})

    merged_periods = []
    for period in intense_periods:
        if not merged_periods:
            merged_periods.append(period)
        else:
            last = merged_periods[-1]
            if period['start_frame'] <= last['end_frame'] + 1:
                last['end_frame'] = max(last['end_frame'], period['end_frame'])
            else:
                merged_periods.append(period)

    return merged_periods

# 射门检测部分
def is_approximately_vertical(line, angle_threshold=ANGLE_THRESHOLD_VERTICAL):
    x1, y1, x2, y2 = line[0]
    if x2 - x1 == 0:
        angle = 90
    else:
        angle = np.degrees(np.arctan(abs((y2 - y1) / (x2 - x1))))
    is_vert = (90 - angle_threshold) <= angle <= (90 + angle_threshold)
    return is_vert

def extract_white_regions(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask

def get_bounding_box_from_line(line):
    x1, y1, x2, y2 = line[0]
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return (left, top, width, height)

def process_image(image_path, angle_threshold=ANGLE_THRESHOLD_VERTICAL, x_threshold=X_THRESHOLD_VERTICAL):
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"无法读取图像: {image_path}")
        return None, []
    
    white_mask = extract_white_regions(image)
    blurred = cv2.GaussianBlur(white_mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 
                            rho=1, 
                            theta=np.pi / 180, 
                            threshold=100, 
                            minLineLength=50, 
                            maxLineGap=10)

    vertical_lines = []
    if lines is not None:
        for line in lines:
            if is_approximately_vertical(line, angle_threshold):
                vertical_lines.append(line)
            
    unique_vertical_lines = []
    unique_x = []
    for line in vertical_lines:
        x = (line[0][0] + line[0][2]) / 2
        if not any(abs(x - existing_x) < x_threshold for existing_x in unique_x):
            unique_vertical_lines.append(line)
            unique_x.append(x)

    if len(unique_vertical_lines) >= 2:
        post_bboxes = [get_bounding_box_from_line(line) for line in unique_vertical_lines[:2]]
        match = re.search(r"SNMOT-(\d{3})/img1/(\d{6})\.jpg", image_path)
        sequence_number = match.group(1)
        img_number = match.group(2)
        frame_number = (int(sequence_number) - 1) * 750 + int(img_number)
        return frame_number, post_bboxes
    else:
        return None, []

def compute_average_color(image, bbox):
    left, top, width, height = map(int, bbox)
    roi = image[top:top+height, left:left+width]
    if roi.size == 0:
        logging.warning(f"Empty ROI for bbox: {bbox}")
        return (0, 0, 0)
    avg_color = cv2.mean(roi)[:3]
    return avg_color

def color_distance(color1, color2):
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))
    return distance

def is_field_color(avg_color, white_threshold=50, green_threshold=50):
    white = (255, 255, 255)
    green = (0, 128, 0)
    distance_to_white = color_distance(avg_color, white)
    distance_to_green = color_distance(avg_color, green)
    is_field = distance_to_white < white_threshold or distance_to_green < green_threshold
    return is_field

def get_image_path_from_frame(folder_path, frame_number):
    sequence_count = int(frame_number) // 750 + 1
    flodername = 'SNMOT-' + str(int(sequence_count)).zfill(3)
    frame_number = int(frame_number)
    image_filename = f"{((frame_number % 750) + 1):06d}.jpg"
    image_path = os.path.join(folder_path, flodername, 'img1', image_filename)
    return image_path

def identify_goalkeeper(frame_number, post_bboxes, tracking_detections, image_folder):
    image_path = get_image_path_from_frame(image_folder, frame_number)
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"无法读取图像: {image_path}")
        return None, None

    track_id_colors = []
    player_detections = []
    for det in tracking_detections:
        avg_color = compute_average_color(image, det['bbox'])
        if not is_field_color(avg_color):
            player_detections.append({
                'track_id': det['track_id'],
                'bbox': det['bbox'],
                'avg_color': avg_color
            })
            track_id_colors.append({
                'track_id': det['track_id'],
                'avg_color': avg_color
            })

    if not player_detections:
        return None, None

    players_with_distance = []
    for player in player_detections:
        avg_color = player['avg_color']
        white = (255, 255, 255)
        green = (0, 128, 0)
        distance_to_white = color_distance(avg_color, white)
        distance_to_green = color_distance(avg_color, green)
        max_distance = max(distance_to_white, distance_to_green)
        players_with_distance.append((player, max_distance))

    players_with_distance.sort(key=lambda x: x[1], reverse=True)
    if not players_with_distance:
        logging.info(f"Frame {frame_number}: No goalkeeper candidate found")
        return None, None
    goalkeeper_candidates = [p for p, dist in players_with_distance]

    def bbox_center(bbox):
        left, top, width, height = bbox
        return (left + width / 2, top + height / 2)

    for candidate in goalkeeper_candidates:
        player_center = bbox_center(candidate['bbox'])
        post_centers = [bbox_center(bbox) for bbox in post_bboxes]
        min_distance = min(
            [math.hypot(player_center[0] - post_center[0], player_center[1] - post_center[1]) 
             for post_center in post_centers]
        )
        candidate['min_distance'] = min_distance

    goalkeeper_candidates.sort(key=lambda x: x['min_distance'])
    goalkeeper_candidate = goalkeeper_candidates[0]

    for bbox in post_bboxes:
        bbox = goalkeeper_candidate['bbox']
        left, top, width, height = bbox
        if width > height*1.4:
            shot_frame = int(frame_number)
            logging.info(f"识别射门帧：{shot_frame}")
            return goalkeeper_candidate['track_id'], shot_frame
        else:
            return goalkeeper_candidate['track_id'], None
    
def detect_shots(goalkeeper_results, ball_tracks_confirmed, frame_dict, player_info, image_folder):
    """
    检测射门事件。

    参数:
        goalkeeper_results (dict): 每帧的守门员 track_id。
        ball_tracks_confirmed (list): 确认的球轨迹列表。
        frame_dict (dict): 每帧的检测数据。
        player_info (dict): 球员信息。
        image_folder (str): 图像文件夹路径。

    返回:
        shots (list): 检测到的射门事件列表，包含射门帧、守门员ID和射门球员ID。
    """
    shots = []
    shot_frames = [frame for frame, shot in goalkeeper_results.items() if shot is not None]

    for shot_frame in shot_frames:
        # 定义搜索范围前后各15帧
        search_start = max(1, int(shot_frame) - MIN_SHOT_FRAMES_AROUND)
        search_end = int(shot_frame) + MIN_SHOT_FRAMES_AROUND

        # 获取最后出现球的位置
        relevant_ball_tracks = [bt for bt in ball_tracks_confirmed if search_start <= bt['frame'] <= search_end and bt['track_id'] is not None]
        if not relevant_ball_tracks:
            logging.warning(f"No relevant ball tracks found around shot frame {shot_frame}")
            continue

        # 找到球最后出现的位置
        last_ball_track = max(relevant_ball_tracks, key=lambda x: x['frame'])
        ball_position = last_ball_track['position']
        ball_track_id = last_ball_track['track_id']

        # 找到离球最近的球员
        min_dist = float('inf')
        shooter_id = None
        for bt in relevant_ball_tracks:
            frame = bt['frame']
            detections = frame_dict.get(frame, [])
            for det in detections:
                if det['track_id'] in player_info:
                    bbox = det['bbox']
                    left, top, width, height = bbox
                    left_bottom = np.array([left, top + height])
                    right_bottom = np.array([left + width, top + height])
                    dist_left = distance(ball_position, left_bottom)
                    dist_right = distance(ball_position, right_bottom)
                    dist = min(dist_left, dist_right)
                    if dist < min_dist:
                        min_dist = dist
                        shooter_id = det['track_id']
                        shooter_jersey = player_info[shooter_id]['jersey_number']
                        shooter_team = player_info[shooter_id]['team']

        if shooter_id is not None:
            shots.append({
                'shot_frame': shot_frame,
                'goalkeeper_id': goalkeeper_results[shot_frame],
                'shooter_id': shooter_id,
                'shooter_jersey': shooter_jersey,
                'shooter_team': shooter_team
            })
            logging.info(f"Shot detected at frame {shot_frame}: Shooter Track ID {shooter_id}, Goalkeeper Track ID {goalkeeper_results[shot_frame]}")
        else:
            logging.warning(f"No shooter identified for shot frame {shot_frame}")

    return shots

def process_images(frame_dict, folder_path, tracking_files_path=None):
    max_frame = max(frame_dict.keys())
    use_frame_start = max_frame - 100
    use_frame_end = max_frame-1
    image_files = [get_image_path_from_frame(folder_path, frame) for frame in range(use_frame_start, use_frame_end + 1)]  
    frames_with_two_posts = {}
    with ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(process_image, img_path, ANGLE_THRESHOLD_VERTICAL, X_THRESHOLD_VERTICAL): img_path for img_path in image_files}
        for future in as_completed(future_to_image):
            img_path = future_to_image[future]
            try:
                frame_number, post_bboxes = future.result()
                if frame_number and len(post_bboxes) == 2:
                    frames_with_two_posts[frame_number] = post_bboxes
            except Exception as exc:
                logging.error(f"图像处理异常 {img_path}: {exc}")

    if not frames_with_two_posts:
        logging.warning("没有检测到包含两个立柱的帧。")
        return

    goalkeeper_results = {}
    with ThreadPoolExecutor() as executor:
        future_to_frame = {}
        for frame_number, post_bboxes in frames_with_two_posts.items():
            detections = frame_dict.get(int(frame_number), [])
            future = executor.submit(identify_goalkeeper, frame_number, post_bboxes, detections, folder_path)
            future_to_frame[future] = frame_number
        for future in as_completed(future_to_frame):
            frame_number = future_to_frame[future]
            try:
                goalkeeper_id, shot_frame = future.result()
                if goalkeeper_id is not None:
                    if shot_frame:
                        goalkeeper_results[frame_number] = shot_frame
                        logging.info(f"Frame {frame_number}: Goalkeeper Track ID {goalkeeper_id} detected a shot.")
                    else:
                        logging.info(f"Frame {frame_number}: Goalkeeper Track ID {goalkeeper_id} identified.")
                else:
                    logging.warning(f"Frame {frame_number}: 未能识别守门员。")
            except Exception as exc:
                logging.error(f"守门员识别生成异常 帧 {frame_number}: {exc}")

    logging.info("\n识别到的守门员:")
    for frame, result in goalkeeper_results.items():
        if isinstance(result, int):
            logging.info(f"帧 {frame}: 守门员 Track ID {goalkeeper_results[frame]}")
        else:
            logging.info(f"帧 {frame}: 守门员 Track ID {result}")

    return goalkeeper_results

def generate_seq(task_hash, tracking_files_path = TRACKING_FILES_PATH, ball_tracks_path = BALL_TRACKS_PATH , player_info_path = PLAYER_INFO_PATH, image_folder = IMAGE_FOLDER):
    df = load_csv_multiple(tracking_files_path)
    ball_track_ids = load_ball_tracks(ball_tracks_path)
    player_info = load_player_info(player_info_path)
    frame_dict = build_frame_dict(df)
    ball_tracks_confirmed = confirm_ball_tracks_with_kalman(frame_dict, ball_track_ids)
    match_info_filename = task_hash + '.json'
    match_info_path = os.path.join('out', 'match', match_info_filename)

    with open(match_info_path) as f:
        match_info_data = json.load(f)

    start_time_stamp = match_info_data['matchInfo']['start_time_stamp']
    start_time_seconds = convert_time_to_seconds(start_time_stamp)

    events_sequence = []

    headers = detect_headers(ball_tracks_confirmed, player_info, frame_dict)
    for header in headers:
        frame_number = header['frame']
        timestamp = calculate_timestamp(frame_number, start_time_seconds)
        event = {
            "timestamp": timestamp,
            "type": "header",
            "team": player_info[header['player_track_id']]['team'],
            "player": player_info[header['player_track_id']]['jersey_number'],
            "description": f"Player {player_info[header['player_track_id']]['jersey_number']} performed a header."
        }
        events_sequence.append(event)

    falls = detect_falls(frame_dict, player_info)
    for fall in falls:
        frame_number = fall['frame']
        timestamp = calculate_timestamp(frame_number, start_time_seconds)
        contest_players_info = [
            {
                "team": cp['team'],
                "jersey": cp['jersey_number']
            }
            for cp in fall['contest_players']
        ]
        event = {
            "timestamp": timestamp,
            "type": "fall",
            "team": fall['team'],
            "player": fall['jersey_number'],
            "description": f"Player {fall['jersey_number']} fell.",
            "with_player": contest_players_info
        }
        events_sequence.append(event)

    carries = detect_ball_carries(ball_tracks_confirmed, player_info, frame_dict)
    for carry in carries:
        start_frame_number = carry['start_frame']
        timestamp = calculate_timestamp(start_frame_number, start_time_seconds)
        event = {
            "timestamp": timestamp,
            "type": "carry",
            "team": carry['team'],
            "player": carry['jersey_number'],
            "description": f"Player {carry['jersey_number']} is carrying the ball.",
            "period": carry['end_frame'] - carry['start_frame']
        }
        events_sequence.append(event)

    passes = detect_passes(ball_tracks_confirmed, player_info, frame_dict)
    for p in passes:
        start_frame_number = p['start_frame']
        timestamp = calculate_timestamp(start_frame_number, start_time_seconds)
        event = {
            "timestamp": timestamp,
            "type": "pass",
            "team": p['team'],
            "player": p['from_jersey_number'],
            "with_player": {
                "team": p['team'],
                "jersey": p['to_jersey_number']
            },
            "description": f"Player {p['from_jersey_number']} passed to Player {p['to_jersey_number']}.",
            "period": p['end_frame'] - p['start_frame']
        }
        events_sequence.append(event)

    goalkeeper_results = process_images(
        frame_dict=frame_dict,
        folder_path=image_folder,
        tracking_files_path=tracking_files_path,
    )
    shots = detect_shots(goalkeeper_results, ball_tracks_confirmed, frame_dict, player_info, image_folder)
    for shot in shots:
        frame_number = shot['shot_frame']
        timestamp = calculate_timestamp(frame_number, start_time_seconds)
        event = {
            "timestamp": timestamp,
            "type": "shot",
            "team": shot['shooter_team'],
            "player": shot['shooter_jersey'],
            "description": f"Player {shot['shooter_jersey']} took a shot."
        }
        events_sequence.append(event)

    intense_periods = detect_intense_periods(ball_tracks_confirmed, frame_dict)
    for period in intense_periods:
        start_frame_number = period['start_frame']
        timestamp = calculate_timestamp(start_frame_number, start_time_seconds)
        event = {
            "timestamp": timestamp,
            "type": "intense",
            "description": "Intense period of play.",
            "period": period['end_frame'] - period['start_frame']
        }
        events_sequence.append(event)

    cleaned_events = merge_events(events_sequence)
    match_info_data['events_sequence'] = cleaned_events

    with open(match_info_path, "w") as f:
        json.dump(match_info_data, f, ensure_ascii=False, indent=4)