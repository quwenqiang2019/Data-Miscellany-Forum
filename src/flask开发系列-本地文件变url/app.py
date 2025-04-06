from flask import Flask, send_from_directory, Response
import os

app = Flask(__name__)

# 设置音频文件存储的目录
AUDIO_FOLDER = 'audio_files'

# 确保音频文件目录存在
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

@app.route('/audio/<filename>')
def serve_audio(filename):
    """
    提供音频文件的下载
    """
    try:
        # 检查文件是否存在
        file_path = os.path.join(AUDIO_FOLDER, filename)
        if os.path.exists(file_path):
            # 设置响应头，触发浏览器下载
            response = send_from_directory(AUDIO_FOLDER, filename, as_attachment=True)
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
        else:
            return "File not found", 404
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)