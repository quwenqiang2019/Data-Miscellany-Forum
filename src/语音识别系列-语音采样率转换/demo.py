import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

"""
将音频文件的采样率从48kHz转化为16kHz
"""

# # ======方法1：使用scipy库进行采样率转换
# # 读取48kHz的音频文件
# input_file = "录音.wav"
# output_file = "录音-16k.wav"
# sample_rate_48k, data = wavfile.read(input_file)
#
# # 确保音频是单声道
# if data.ndim > 1:
#     data = data[:, 0]
#
# # 计算采样后的样本数量
# num_samples_16k = int(len(data) * 16000 / sample_rate_48k)
#
# # 使用scipy的resample函数进行重采样
# data_16k = resample(data, num_samples_16k)
#
# # 保存为16kHz的音频文件
# wavfile.write(output_file, 16000, data_16k.astype(np.int16))


# ======方法2：使用pydub库进行采样率转换
from pydub import AudioSegment

# 加载48kHz的音频文件
audio = AudioSegment.from_wav('录音.wav')

# 转换采样率为16kHz
audio_16k = audio.set_frame_rate(16000)

# 导出转换后的音频文件
audio_16k.export('录音-16k-2.wav', format='wav')