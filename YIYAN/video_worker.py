import cv2
import requests
from bs4 import BeautifulSoup
from frame_worker import save_video_frames


def extract_video_url(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 查找视频标签（例如 <video> 或 <source>）
    video_tag = soup.find('video') or soup.find('source')
    
    if video_tag:
        # 获取视频的URL
        video_url = video_tag.get('src')
        if not video_url.startswith('http'):
            # 如果URL是相对路径，将其转换为绝对路径
            base_url = "https://www.ikcest.org"  # 请替换为实际的网页基础URL
            video_url = base_url + video_url
        return video_url
    else:
        print("Error: No video tag found in the HTML content.")
        return None


def read_video_from_url(url):
    # 获取网页内容
    response = requests.get(url)
    response.raise_for_status()
    html_content = response.text

    # 提取视频URL
    video_url = extract_video_url(html_content)
    if not video_url:
        return

    # 使用OpenCV读取视频流
    video_stream = cv2.VideoCapture(video_url)

    if not video_stream.isOpened():
        print("Error: Could not open video stream.")
        return

    save_video_frames(video_stream)

if __name__ == "__main__":
    # 示例网页链接
    webpage_url = "https://www.ikcest.org/bigdata2024/spzs/content/240950485065250646.html"
    read_video_from_url(webpage_url)
