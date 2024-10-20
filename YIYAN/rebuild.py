import os
import json
import pandas as pd
from math import sqrt, floor
import logging
import numpy as np

# 配置参数
SCALED_PATH = '../GHOST/out/scaled/'
PLAYER_TEAM_MAPPING_PATH = '../jersey-number-pipeline/out/SoccerNetResults/player_team_mapping.json'
SOCCER_BALL_PATH = '../jersey-number-pipeline/out/SoccerNetResults/soccer_ball.json'
MATCH_INFO_PATH = '../out/match/match_info.json'
FPS = 30  # 视频帧率
DISTANCE_THRESHOLD = 50  # 拥有球距离阈值（像素）
DRIBBLE_MIN_FRAMES = 10  # 带球最少连续帧数
SPEED_THRESHOLD = 15  # 射门速度阈值（像素/帧）
FIELD_MARGIN_RATIO = 0.05  # 球场边界边距比例

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("event_reconstruction_debug.log"),
        logging.StreamHandler()
    ]
)

# 辅助函数
def get_timestamp(frame, start_time, fps=25):
    """
    将帧号转换为时间戳。
    start_time格式为"MM:SS"
    """
    try:
        start_minutes, start_seconds = map(float, start_time.split(':'))
    except ValueError:
        logging.error(f"无效的 start_time 格式: {start_time}")
        return "00:00.000"
    total_start_seconds = start_minutes * 60 + start_seconds
    event_time_seconds = total_start_seconds + frame / fps
    minutes = int(event_time_seconds // 60)
    seconds = int(event_time_seconds % 60)
    milliseconds = int((event_time_seconds - floor(event_time_seconds)) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def calculate_distance(pos1, pos2):
    """计算两个位置之间的欧氏距离。"""
    return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def get_team(track_id, player_team_mapping):
    """根据 track_id 返回所属球队。"""
    if track_id in player_team_mapping:
        return player_team_mapping[track_id]['team']
    return "Unknown"

def get_player(track_id, player_team_mapping):
    """根据 track_id 返回球员的球衣号码。"""
    if track_id in player_team_mapping:
        jersey_number = player_team_mapping[track_id]['jersey_number']
        if jersey_number == "-1":
            return "Unknown"
        return jersey_number
    return "Unknown"

def teams_match(track_id1, track_id2, player_team_mapping):
    """判断两个 track_id 是否属于同一球队。"""
    return get_team(track_id1, player_team_mapping) == get_team(track_id2, player_team_mapping)

# 数据加载 test OK
def load_player_team_mapping(path):
    logging.info(f"加载球员和球队映射数据，从 {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        logging.error(f"加载球员和球队映射数据失败: {e}")
        return {}

def load_soccer_ball_tracks(path):
    logging.info(f"加载足球追踪数据，从 {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            ball_tracks = data.get('ball_tracks', [])
            return ball_tracks
    except Exception as e:
        logging.error(f"加载足球追踪数据失败: {e}")
        return []

def load_match_info(path):
    logging.info(f"加载比赛信息，从 {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        logging.error(f"加载比赛信息失败: {e}")
        return {}

def load_tracking_data(scaled_path):
    logging.info(f"加载所有追踪数据，从 {scaled_path}...")
    all_tracks = {}
    current_frame_offset = 0
    
    if not os.path.exists(scaled_path):
        logging.error(f"路径不存在: {scaled_path}")
        return all_tracks
    
    scaled_files = sorted([f for f in os.listdir(scaled_path) if f.endswith('.txt')])

    for filename in scaled_files:
        filepath = os.path.join(scaled_path, filename)
        base_name = os.path.basename(filepath).split('.')[0]

        try:
            df = pd.read_csv(filepath, header=None,
                             names=['frame', 'track_id', 'left', 'top', 'width', 'height', 'f7', 'f8', 'f9', 'f10'])

            max_frame_in_file = df['frame'].max()

            for _, row in df.iterrows():
                global_frame = row['frame'] + current_frame_offset
                track_id = int(row['track_id'])
                track = f"{base_name}_{track_id}"
                if track not in all_tracks:
                    all_tracks[track] = []
                all_tracks[track].append({'frame': global_frame, 'left': row['left'], 'top': row['top']})

            # 更新帧偏移
            current_frame_offset += max_frame_in_file

        except Exception as e:
            logging.error(f"处理文件 {filename} 失败: {e}")
    
    # 转换为 DataFrame
    tracks_df = {}
    for track, data in all_tracks.items():
        try:
            tracks_df[track] = pd.DataFrame(data).sort_values('frame').reset_index(drop=True)
        except Exception as e:
            logging.error(f"转换 track {track} 为 DataFrame 失败: {e}")
    
    logging.info(f"总共加载了 {len(tracks_df)} 个 track。")
    return tracks_df

# 动态推断球门位置
def infer_goal_lines(soccer_ball_tracks, tracks_df):
    logging.info("推断球门位置...")
    ball_positions = []
    for ball_track in soccer_ball_tracks:
        if ball_track in tracks_df:
            positions = tracks_df[ball_track][['left', 'top']].values.tolist()
            ball_positions.extend(positions)
        else:
            logging.warning(f"足球 track_id {ball_track} 不在追踪数据中。")
    if not ball_positions:
        # 默认值，如果没有足球位置数据
        left_most = 0
        right_most = 1000
        logging.warning("没有找到足球位置数据，使用默认的球门位置。")
    else:
        left_most = min([pos[0] for pos in ball_positions])
        right_most = max([pos[0] for pos in ball_positions])
    # 添加边距
    field_margin = (right_most - left_most) * FIELD_MARGIN_RATIO
    goal_line_left = left_most - field_margin
    goal_line_right = right_most + field_margin
    return goal_line_left, goal_line_right

# 事件检测
def detect_events(tracks_df, soccer_ball_tracks, player_team_mapping, match_info):
    goal_line_left, goal_line_right = infer_goal_lines(soccer_ball_tracks, tracks_df)
    start_time_stamp = match_info['matchInfo'].get('start_time_stamp', '00:00')
    logging.info(f"比赛开始时间戳: {start_time_stamp}")
    
    events_sequence = []
    previous_owner = None
    dribble_start_frame = None
    previous_ball_pos = None
    previous_frame = None
    
    all_frames = sorted({row.frame for df in tracks_df.values() for row in df.itertuples(index=False)})
    
    for frame in all_frames:
        # 获取当前帧中所有足球的位置
        current_ball_positions = []
        for ball_track in soccer_ball_tracks:
            if ball_track in tracks_df:
                ball_df = tracks_df[ball_track]
                ball_pos_row = ball_df[ball_df['frame'] == frame]
                if not ball_pos_row.empty:
                    pos = ball_pos_row[['left', 'top']].values[0]
                    current_ball_positions.append(pos)
            else:
                logging.warning(f"足球 track_id {ball_track} 不在追踪数据中。")
        
        if not current_ball_positions:
            continue  # 没有足球位置，跳过此帧
        
        # 主要足球（如有多个）
        ball_pos = current_ball_positions[0]
        
        # 找到当前帧拥有球的球员
        owners = []
        for player_track in player_team_mapping:
            if player_track not in tracks_df:
                continue
            player_df = tracks_df[player_track]
            player_pos_rows = player_df[player_df['frame'] == frame]
            if player_pos_rows.empty:
                continue
            player_pos = player_pos_rows[['left', 'top']].values[0]
            distance = calculate_distance(player_pos, ball_pos)
            if distance < DISTANCE_THRESHOLD:
                owners.append(player_track)
        
        # 判断拥有者
        if len(owners) > 1:
            current_owner = None  # 多个拥有者，无法确定
        elif len(owners) == 1:
            current_owner = owners[0]
        else:
            current_owner = None  # 无人拥有
        
        # 事件检测：传球
        if previous_owner and current_owner and previous_owner != current_owner:
            if teams_match(previous_owner, current_owner, player_team_mapping):
                timestamp = get_timestamp(frame, start_time_stamp, FPS)
                event = {
                    "timestamp": timestamp,
                    "type": "pass",
                    "team": get_team(previous_owner, player_team_mapping),
                    "player": get_player(previous_owner, player_team_mapping),
                    "target_player": get_player(current_owner, player_team_mapping),
                    "description": f"Player {get_player(previous_owner, player_team_mapping)} passed to Player {get_player(current_owner, player_team_mapping)}."
                }
                events_sequence.append(event)
                logging.info(f"检测到传球事件: {event}")
        
        # 事件检测：带球
        if current_owner:
            if current_owner == previous_owner:
                if dribble_start_frame is None:
                    dribble_start_frame = frame
                    logging.debug(f"球员 {current_owner} 开始带球，帧 {frame}。")
                else:
                    duration = frame - dribble_start_frame
                    if duration >= DRIBBLE_MIN_FRAMES:
                        timestamp = get_timestamp(frame, start_time_stamp, FPS)
                        event = {
                            "timestamp": timestamp,
                            "type": "dribbling",
                            "team": get_team(current_owner, player_team_mapping),
                            "player": get_player(current_owner, player_team_mapping),
                            "description": f"Player {get_player(current_owner, player_team_mapping)} is dribbling the ball."
                        }
                        events_sequence.append(event)
                        logging.info(f"检测到带球事件: {event}")
                        dribble_start_frame = None  # 重置
            else:
                dribble_start_frame = frame  # 新的拥有者，重置带球开始帧
        else:
            dribble_start_frame = None  # 无人拥有，重置带球开始帧
        
        # 事件检测：射门与进球
        if np.any(previous_ball_pos) and previous_frame:
            dx = ball_pos[0] - previous_ball_pos[0]
            dy = ball_pos[1] - previous_ball_pos[1]
            distance = calculate_distance(ball_pos, previous_ball_pos)
            frame_diff = frame - previous_frame
            if frame_diff == 0:
                frame_diff = 1  # 避免除以零
            speed = distance / frame_diff  # 像素/帧
            
            # 射门检测
            if speed > SPEED_THRESHOLD:
                # 判断射门方向是否朝向球门
                if ball_pos[0] < goal_line_left + (goal_line_right - goal_line_left)*0.1 or \
                   ball_pos[0] > goal_line_right - (goal_line_right - goal_line_left)*0.1:
                    # 添加射门事件
                    owner = previous_owner
                    if owner:
                        team = get_team(owner, player_team_mapping)
                        player = get_player(owner, player_team_mapping)
                        timestamp = get_timestamp(frame, start_time_stamp, FPS)
                        shot_event = {
                            "timestamp": timestamp,
                            "type": "shot",
                            "team": team,
                            "player": player,
                            "status": "fail"  # 默认未进球
                        }
                        events_sequence.append(shot_event)
                        logging.info(f"检测到射门事件: {shot_event}")
        
        # 事件检测：进球
        goal_detected = False
        team_scored = None
        player_scored = None
        if ball_pos[0] < goal_line_left:
            goal_detected = True
            # 进球方向为左侧球门
            team_scored = match_info['matchInfo'].get('second_team', "Unknown")
        elif ball_pos[0] > goal_line_right:
            goal_detected = True
            # 进球方向为右侧球门
            team_scored = match_info['matchInfo'].get('first_team', "Unknown")
        
        if goal_detected and previous_owner:
            player_scored = get_player(previous_owner, player_team_mapping)
            timestamp = get_timestamp(frame, start_time_stamp, FPS)
            goal_event = {
                "timestamp": timestamp,
                "type": "shot",
                "team": team_scored,
                "player": player_scored,
                "status": "success"
            }
            # 更新最新的未完成的射门事件
            for event in reversed(events_sequence):
                if event["type"] == "shot" and event["status"] == "fail":
                    event["status"] = "success"
                    break
            # 添加进球事件
            events_sequence.append(goal_event)
            logging.info(f"检测到进球事件: {goal_event}")
        
        # 更新前一帧信息
        previous_ball_pos = ball_pos
        previous_frame = frame
        previous_owner = current_owner
    
    return events_sequence

def rebuild():
    logging.info("开始加载数据...")
    player_team_mapping = load_player_team_mapping(PLAYER_TEAM_MAPPING_PATH)
    soccer_ball_tracks = load_soccer_ball_tracks(SOCCER_BALL_PATH)
    match_info = load_match_info(MATCH_INFO_PATH)
    tracks_df = load_tracking_data(SCALED_PATH)

    if not tracks_df:
        logging.error("没有有效的追踪数据")
        return
    if not soccer_ball_tracks:
        logging.error("没有有效的足球 track_id")
        return
    if not player_team_mapping:
        logging.error("没有有效的球员和球队映射数据")
        return
    if not match_info:
        logging.error("没有有效的比赛信息")
        return
    
    logging.info("开始事件检测...")
    events_sequence = detect_events(tracks_df, soccer_ball_tracks, player_team_mapping, match_info)
    logging.info(f"检测到 {len(events_sequence)} 个事件。")
    match_info['events_sequence'] = events_sequence
    updated_match_info_path = 'out/match/match_info.json'
    try:
        with open(updated_match_info_path, 'w', encoding='utf-8') as f:
            json.dump(match_info, f, ensure_ascii=False, indent=4)
        logging.info(f"保存成功: {updated_match_info_path}")
    except Exception as e:
        logging.error(f"保存失败: {e}")