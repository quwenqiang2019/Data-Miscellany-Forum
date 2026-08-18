#pip install kaldi_native_fbank
from pathlib import Path
from modelscope import snapshot_download
model_dir = snapshot_download('dengcunqin/SenseVoiceSmall_hotword')
import sys
sys.path.append(model_dir)

from sensevoice_bin_hot import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
model = SenseVoiceSmall(model_dir, batch_size=10, quantize=False)

wav_or_scp = [model_dir+"/a2.wav".format(Path.home(), model_dir)]

res = model(wav_or_scp,hotwords_str='苏商银行',hotwords_score=1.0)
print('SenseVoiceSmall_hotword output:',[rich_transcription_postprocess(i) for i in res])

