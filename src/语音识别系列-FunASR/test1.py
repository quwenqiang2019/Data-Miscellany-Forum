import torch
import torchaudio
import funasr
from funasr import AutoModel
import modelscope
import numpy as np
import pandas as pd
from IPython.display import Audio,clear_output,Video,display
import torchaudio
from IPython.display import Audio
print("funasr: ", funasr.__version__) # funasr:  1.0.25
print("torch: ", torch.__version__) # torch:  2.1.2
print("torchaudio: ", torchaudio.__version__) # torchaudio:  2.1.2
print("modelscope: ", modelscope.__version__) # modelscope:  1.15.0


# 准备音频素材
'''
!edge-tts --voice zh-CN-YunyangNeural --text "曾经有一份真诚的爱情放在我面前，我没有珍惜，等我失去的时候我才后悔莫及，人世间最痛苦的事莫过
于此。如果上天能够给我一个再来一次的机会，我会对那个女孩子说三个字：我爱你。如果非要在这份爱上加上一个期限，我希望是……一万年" --write-media speaker2.wav
'''
speaker2_wav = "speaker2.wav"
waveform, sample_rate = torchaudio.load(speaker2_wav)
Audio(waveform, rate=sample_rate, autoplay=True)


# 利用paraformer-zh-语音文字识别
model = AutoModel(model="paraformer-zh")
res = model.generate(input=speaker2_wav)
print("识别出的结果:", res[0]['text'])