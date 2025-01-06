import json

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


# # 准备音频素材
# '''
# !edge-tts --voice zh-CN-YunyangNeural --text "曾经有一份真诚的爱情放在我面前，我没有珍惜，等我失去的时候我才后悔莫及，人世间最痛苦的事莫过
# 于此。如果上天能够给我一个再来一次的机会，我会对那个女孩子说三个字：我爱你。如果非要在这份爱上加上一个期限，我希望是……一万年" --write-media speaker2.wav
# '''
# speaker2_wav = "speaker2.wav"
# waveform, sample_rate = torchaudio.load(speaker2_wav)
# Audio(waveform, rate=sample_rate, autoplay=True)
#
#
# # 利用paraformer-zh-语音文字识别
# model = AutoModel(model="paraformer-zh")
# res = model.generate(input=speaker2_wav)
# print("识别出的结果:", res[0]['text'])
#
# # 利用fsmn_vad_zh来-语音结束点识别
# model=AutoModel(model="fsmn-vad")
# res=model.generate(input=speaker2_wav)
# print(res)
#
# # 利用ct-punc-对语音识别文本进行标点符号预测
# model = AutoModel(model="ct-punc")
# res = model.generate(input="曾 经 有 一 份 真 诚 的 爱 情 放 在 我 面 前 我 没 有 珍 惜 等 我 失 去 的 时 候 我 才 后 悔 莫 及 人 世 间 最 痛 苦 的 是 莫 过 于 此 如 果 上 天 能 够 给 我 一 个 再 来 一 次 的 机 会 我 会 对 那 个 女 孩 子 说 三 个 字 我 爱 你 如 果 非 要 在 这 份 爱 上 加 上 一 个 期 限 我 希 望 是 一 万 年")
# print(res)
#
# # 利用cam++ 对语音中说话人身份识别
# model = AutoModel(model="cam++")
# res = model.generate(input=speaker2_wav)
# print(res)

# 对多说话人进行语音识别
speaker1_wav = ("E:\\data\\a2.wav")
waveform, sample_rate = torchaudio.load(speaker1_wav)
Audio(waveform, rate=sample_rate, autoplay=True)

# funasr_model = AutoModel(model="C:\\Users\quwen\.cache\modelscope\hub\iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
#                         vad_model="C:\\Users\quwen\.cache\modelscope\hub\iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
#                         punc_model="C:\\Users\quwen\.cache\modelscope\hub\iic\punc_ct-transformer_cn-en-common-vocab471067-large",
#                         spk_model="C:\\Users\quwen\.cache\modelscope\hub\iic/speech_campplus_sv_zh-cn_16k-common",
#                         )

funasr_model = AutoModel(model="E:\\model\speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                        vad_model="E:\\model\speech_fsmn_vad_zh-cn-16k-common-pytorch",
                        punc_model="E:\\model\punc_ct-transformer_cn-en-common-vocab471067-large",
                        spk_model="E:\\model\speech_campplus_sv_zh-cn_16k-common",
                        )

res = funasr_model.generate(input=speaker1_wav,
                            batch_size_s=300,
                            hotword='苏珊银行')
print(res)
print(type(res))

conv = ''
sentence_info = res[0]['sentence_info']
for sentence in sentence_info:
    conv = conv + f"spk {sentence['spk']} : {sentence['text']}\n"

print(conv)

