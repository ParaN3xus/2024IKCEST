import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def extract_video_url(webpage_url):
    try:
        response = requests.get(webpage_url)
        response.raise_for_status()
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        video_tag = soup.find('video') or soup.find('source')

        if video_tag:
            video_url = video_tag.get('src') or video_tag.get('data-src')
            if video_url:
                video_url = urljoin(webpage_url, video_url)
                return video_url
        print("找不到视频。")
        return None
    except Exception as e:
        print(f"视频提取发生错误: {e}")
        return None

def download_part(video_url, start, end, part_path, thread_lock):
    headers = {'Range': f'bytes={start}-{end}'}
    try:
        response = requests.get(video_url, headers=headers, stream=True)
        response.raise_for_status()
        with open(part_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Part {start}-{end} 下载成功。")
    except Exception as e:
        print(f"下载部分 {start}-{end} 发生错误: {e}")

def download_video_from_url(video_url, download_dir='downloads', num_threads=8):
    try:
        # 获取视频文件大小
        head = requests.head(video_url)
        head.raise_for_status()
        if 'Content-Length' not in head.headers:
            print("无法获取视频文件大小，使用单线程下载。")
            filename = video_url.split('/')[-1].split('?')[0]
            video_id = str(uuid.uuid4())
            local_path = os.path.join(download_dir, video_id, filename)
            os.makedirs(os.path.join(download_dir, video_id), exist_ok=True)
            with requests.get(video_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"视频下载成功，位于: {local_path}")
            md5_hash = calculate_md5(local_path)[:20]
            return local_path, md5_hash

        total_size = int(head.headers['Content-Length'])
        filename = video_url.split('/')[-1].split('?')[0]
        video_id = str(uuid.uuid4())
        video_dir = os.path.join(download_dir, video_id)
        os.makedirs(video_dir, exist_ok=True)
        local_path = os.path.join(video_dir, filename)
        
        part_size = total_size // num_threads
        futures = []
        part_paths = []
        thread_lock = Lock()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(num_threads):
                start = part_size * i
                # 确保最后一个线程下载到文件结尾
                end = start + part_size - 1 if i < num_threads - 1 else total_size - 1
                part_path = os.path.join(video_dir, f'part_{i}')
                part_paths.append(part_path)
                futures.append(executor.submit(download_part, video_url, start, end, part_path, thread_lock))
        
        # 等待所有线程完成
        for future in futures:
            future.result()
        
        # 合并所有部分
        with open(local_path, 'wb') as outfile:
            for part_path in part_paths:
                with open(part_path, 'rb') as infile:
                    outfile.write(infile.read())
                os.remove(part_path)  # 删除部分文件
        print(f"视频下载成功，位于: {local_path}")
        
        md5_hash = calculate_md5(local_path)[:20]
        return local_path, md5_hash
    except Exception as e:
        print(f"下载发生错误: {e}")
        return None, None

def download_video(webpage_url):
    video_url = extract_video_url(webpage_url)
    if not video_url:
        return None, None
    return download_video_from_url(video_url)

# 示例用法：
# path, hash_prefix = download_video("http://example.com/video-page")
# print("下载到:", path)
# print("Hash前缀:", hash_prefix)
