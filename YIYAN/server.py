from flask import Flask, request, send_file, make_response, jsonify
from flask_cors import CORS
import hashlib
import json
import uuid
from work import *
from config import YOLOXConfig
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
executor = ThreadPoolExecutor(max_workers=5)
tasks = {}

def make_json_response(data, status_code=200):
    response = make_response(json.dumps(data, ensure_ascii=False), status_code)
    response.headers["Content-Type"] = "application/json"
    return response

@app.route("/generate_commentary", methods=['GET'])
def generate_commentary():
    try:
        url = request.args.get('video_url', "")
        if not url:
            app.logger.warning("缺少视频链接")
            return make_json_response({"error": "缺少视频链接"}, 400)
        task_id = hashlib.md5(url.encode()).hexdigest()
        
        if task_id in tasks:
            # 检查任务状态
            future = tasks[task_id]
            if future.running():
                return jsonify({"message": "任务正在运行中"}), 202
            elif future.done():
                return jsonify({"message": "任务已完成", "result": result}), 200
        else:
            future = executor.submit(run, url)
            tasks[task_id] = future
            return jsonify({"message": "任务已启动"}), 202

    except Exception as e:
        app.logger.exception("处理请求时出现异常")
        return jsonify({"error": str(e)}), 500

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
