import os

def get_tracking_info(txt_file):
    """读取跟踪信息"""
    tracking_data = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            frame, track_id, left, top, width, height, *others = map(float, line.strip().split(','))
            tracking_data.append((int(frame), int(track_id), left, top, width, height, *others))
    return tracking_data

def scale_tracking_data(tracking_data, scale_x, scale_y):
    """按缩放系数调整跟踪数据"""
    scaled_data = []
    for data in tracking_data:
        frame, track_id, left, top, width, height, *others = data
        left *= scale_x
        top *= scale_y
        width *= scale_x
        height *= scale_y
        scaled_data.append((frame, track_id, left, top, width, height, *others))
    return scaled_data

def save_scaled_data(scaled_data, output_file):
    """保存缩放后的数据到文件"""
    with open(output_file, 'w') as file:
        for data in scaled_data:
            file.write(','.join(map(str, data)) + '\n')

def process_all_tracking_files(tracking_dir, output_dir, scale_x, scale_y):
    """处理所有跟踪文件并保存缩放后的数据"""
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(tracking_dir):
        if filename.endswith('.txt'):
            input_file = os.path.join(tracking_dir, filename)
            output_file = os.path.join(output_dir, filename)
            
            # 读取和缩放跟踪数据
            tracking_data = get_tracking_info(input_file)
            scaled_data = scale_tracking_data(tracking_data, scale_x, scale_y)
            
            # 保存缩放后的数据
            save_scaled_data(scaled_data, output_file)
            print(f'done: {output_file}')

# 参数设定
tracking_dir = 'out/yolox_dets_OnTheFly:0_each_sample2:0.8:LastFrame:0.7LenThresh:0RemUnconf:0.0LastNFrames:10MM:1KalmanInactPat:50DetConf:0.45NewTrackConf:0.6'
output_dir = 'out/scaled'

# 缩放参数
scale_x = 3.0
scale_y = 3.0

# 处理所有文件
process_all_tracking_files(tracking_dir, output_dir, scale_x, scale_y)
