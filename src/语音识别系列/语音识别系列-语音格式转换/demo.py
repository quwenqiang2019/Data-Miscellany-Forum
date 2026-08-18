import os
from pydub import AudioSegment

# 文件夹路径
input_folder = "D:\\quwen\Documents\录音-副本"
wav_folder = "D:\\quwen\Documents\录音-副本-wav"

os.makedirs(wav_folder, exist_ok=True)

# 批量处理 M4A 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith(".m4a"):
        print(file_name)
        input_path = os.path.join(input_folder, file_name)
        print(input_path)
        wav_path = os.path.join(wav_folder, os.path.splitext(file_name)[0] + ".wav")

        # M4A 转 WAV
        song = AudioSegment.from_file(input_path, format="m4a")
        song.export(wav_path, format="wav")
        print(f"转换完成：{wav_path}")

