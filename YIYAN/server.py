from flask import Flask, request, send_file, make_response, jsonify
from video_download.get_web_video import download_video
from flask_cors import CORS
import json
from datetime import datetime
from work import *
from config import YOLOXConfig
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
executor = ThreadPoolExecutor(max_workers=8)
tasks_file = 'tasks/tasks.json'

def make_json_response(data, status_code=200):
    response = make_response(json.dumps(data, ensure_ascii=False), status_code)
    response.headers["Content-Type"] = "application/json"
    return response

@app.route("/task", methods=['POST'])
def check_task():
    try:
        data = request.get_json()
        url = data.get('video_url', "")
        if not url:
            app.logger.warning("缺少视频链接")
            return make_json_response({"error": "缺少视频链接"}, 400)
        
        vid_path, task_hash = download_video(url)

        with open('tasks_file.json', 'r') as f:
            task_data = json.load(f)

        task = next((t for t in task_data['tasks'] if t['id'] == task_hash), None)
        running_task_exists = any(t['status'] == 'running' for t in task_data['tasks'])

        if task:
            if task['status'] == "created":
                if not running_task_exists:
                    future = executor.submit(run, vid_path, task_hash)
                    tasks[task_hash] = future
                    task['status'] = 'running'
                    with open('tasks_file.json', 'w') as f:
                        json.dump(task_data, f, indent=4)
                    return jsonify({"message": "任务已启动", "task_id": task_hash}), 202
                else:
                    return jsonify({"message": "任务已创建", "task_id": task_hash}), 202

            elif task['status'] == "running":
                return jsonify({"message": "任务正在运行中，无需再次创建！"}), 202

            elif task['status'] == "completed":
                result_filename = task_hash + '.json'
                result_file_path = os.path.join('out', 'match', result_filename)
                with open(result_file_path, 'r') as f:
                    result_data = json.load(f)
                return jsonify({"message": "任务已完成", "result": result_data}), 200

        else:
            if not running_task_exists:
                future = executor.submit(run, vid_path, task_hash)
                tasks[task_hash] = future
                status = "running"
            else:
                status = "created"
            task_data['tasks'].append({
                "user": "",
                "id": task_hash,
                "status": status,
                "create_time": datetime.now().isoformat()
            })
            with open('tasks_file.json', 'w') as f:
                json.dump(task_data, f, indent=4)
            
            if status == "running":
                return jsonify({"message": "任务已启动", "task_id": task_hash}), 202
            else:
                return jsonify({"message": "任务已创建", "task_id": task_hash}), 202

    except Exception as e:
        app.logger.exception("处理请求时出现异常")
        return make_json_response({"error": "处理请求时出现异常"}, 500)

@app.route("/query", methods=['POST'])
def query_task():
    try:
        data = request.get_json()
        task_id = data.get('task_id', "")
        if not task_id:
            app.logger.warning("缺少任务 ID")
            return make_json_response({"error": "缺少任务 ID"}, 400)
        
        with open('tasks_file.json', 'r') as f:
            task_data = json.load(f)

        task = next((t for t in task_data['tasks'] if t['id'] == task_id), None)
        if not task:
            return jsonify({"message": "任务不存在"}), 404

        if task['status'] == "running":
            return jsonify({"message": "任务正在运行中"}), 200
        elif task['status'] == "completed":
            result_filename = task_id + '.json'
            result_file_path = os.path.join('out', 'match', result_filename)
            with open(result_file_path, 'r') as f:
                result_data = json.load(f)
            return jsonify({"message": "任务已完成", "result": result_data}), 200
        else:
            return jsonify({"message": "任务状态异常"}), 500

    except Exception as e:
        app.logger.exception("处理请求时出现异常")
        return make_json_response({"error": "处理请求时出现异常"}, 500)

@app.route("/logo.png")
async def plugin_logo():
    """
    注册用的：返回插件的 logo，要求 48 x 48 大小的 png 文件.
    注意：API路由是固定的，事先约定的。
    """
    return send_file('logo.png', mimetype='image/png')

@app.route("/.well-known/ai-plugin.json")
async def plugin_manifest():
    """
    注册用的：返回插件的描述文件，描述了插件是什么等信息。
    注意：API 路由是固定的，事先约定的。
    """
    host = request.host_url
    with open(".well-known/ai-plugin.json", encoding="utf-8") as f:
        text = f.read().replace("PLUGIN_HOST", host)
        return text, 200, {"Content-Type": "application/json"}

@app.route("/.well-known/openapi.yaml")
async def openapi_spec():
    """
    注册用的：返回插件所依赖的插件服务的API接口描述，参照 openapi 规范编写。
    注意：API 路由是固定的，事先约定的。
    """
    with open(".well-known/openapi.yaml", encoding="utf-8") as f:
        text = f.read()
        return text, 200, {"Content-Type": "text/yaml"}


@app.route("/example.yaml")
async def exampleSpec():
    with open("example.yaml") as f:
        text = f.read()
        return text, 200, {"Content-Type": "text/yaml"}

@app.route('/')
def index():
    return '欢迎使用足球解说词生成插件！'

if __name__ == '__main__':
    # 配置日志
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("server.log"),
            logging.StreamHandler()
        ]
    )

    app.run(debug=False, host='0.0.0.0', port=8081)
