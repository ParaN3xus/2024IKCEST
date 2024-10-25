import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import hashlib
import uuid

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def update_tasks_file(hash_value, tasks_file='../YIYAN/tasks/tasks.run'):
    with open(tasks_file, 'a') as f:
        f.write(f"{hash_value}\n")

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

def download_video_from_url(video_url, download_dir='downloads'):
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        filename = video_url.split('/')[-1].split('?')[0]
        video_id = str(uuid.uuid4())
        local_path = os.path.join(download_dir, video_id, filename)
        os.makedirs(download_dir, exist_ok=True)
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"视频下载成功，位于: {local_path}")
        md5_hash = calculate_md5(local_path)[:20]
        update_tasks_file(md5_hash)

        return local_path, md5_hash
    except Exception as e:
        print(f"下载发生错误: {e}")
        return None, None

def download_video(webpage_url):
    video_url = extract_video_url(webpage_url)
    if not video_url:
        return None, None
    return download_video_from_url(video_url)

# Example usage:
# path, hash_prefix = download_video("http://example.com/video-page")
# print("Downloaded to:", path)
# print("Hash prefix:", hash_prefix)
