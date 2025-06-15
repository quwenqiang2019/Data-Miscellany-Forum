from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

# 请替换为你的 OpenAI API 密钥
openai.api_key = ''
openai.api_base = ''

def check_for_profane_language(text):
    # 这里可以使用 OpenAI 的 API 来检测脏话
    try:
        # 使用 ChatGPT 接口
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个可以检测文本是否包含脏话的助手。"},
                {"role": "user", "content": f"请判断以下文本是否包含脏话：\n\n{text}\n\n请只返回 '包含' 或 '不包含'。"}
            ],
            max_tokens=20,
            temperature=0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return str(e)


@app.route('/check_swearing', methods=['POST'])
def check_swearing():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': '请提供文本'}), 400

    result = check_for_profane_language(text)
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
