import soundfile
import os
import numpy as np
import librosa
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# 添加计时统计
timing_stats = {
    'separation': [],
    'sv_embedding': [],
    'asr': [],
    'total_chunk': [],
    'initial_5s_processing': None
}

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_speaker_embeddings(speech_chunk, sample_rate):
    """提取说话人声纹"""
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        import numpy as np
        import os
        
        # 获取当前脚本的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 初始化说话人分离模型（如果尚未初始化）
        if 'separation_pipeline' not in globals():
            global separation_pipeline
            separation_pipeline = pipeline(
                task=Tasks.speech_separation,
                model='iic/speech_mossformer2_separation_temporal_8k'
            )
        
        # 初始化声纹识别模型（如果尚未初始化）
        if 'sv_pipeline' not in globals():
            global sv_pipeline
            sv_pipeline = pipeline(
                task='speaker-verification',
                model='iic/speech_eres2netv2_sv_zh-cn_16k-common',
                model_revision='v1.0.2'
            )
        
        # 创建临时目录（如果不存在）
        temp_dir = os.path.join(script_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # 将音频重采样到8kHz用于说话人分离
        speech_chunk_8k = librosa.resample(speech_chunk, orig_sr=sample_rate, target_sr=8000)
        
        # 保存临时文件用于说话人分离
        temp_file = os.path.join(temp_dir, 'chunk_8k.wav')
        soundfile.write(temp_file, speech_chunk_8k, 8000)
        
        # 说话人分离
        result = separation_pipeline(temp_file)
        separated_chunks = []
        speech_energies = []
        
        # 处理分离后的音频
        for spk_idx, signal in enumerate(result['output_pcm_list']):
            # 将PCM转换为浮点数组并重采样回原始采样率
            spk_audio = np.frombuffer(signal, dtype=np.int16).astype(np.float32) / 32768.0
            spk_audio = librosa.resample(spk_audio, orig_sr=8000, target_sr=sample_rate)
            separated_chunks.append(spk_audio)
            
            # 计算音频能量
            energy = np.sum(spk_audio ** 2)
            speech_energies.append(energy)
        
        # 清理临时文件
        os.remove(temp_file)
        
        # 提取每个分离音频的声纹
        speaker_embeddings = []
        for i, chunk in enumerate(separated_chunks):
            # 提取声纹
            result_emb = sv_pipeline([chunk], output_emb=True)
            speaker_embeddings.append(result_emb['embs'][0])
        
        return speaker_embeddings, speech_energies, separated_chunks
    except Exception as e:
        logging.error(f"错误：声纹提取失败 - {str(e)}")
        return None, None, None

async def extract_target_embedding(audio_data, sample_rate):
    """异步提取目标说话人声纹"""
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            logging.info("\n开始提取目标说话人声纹...")
            t_5s_start = time.time()
            
            # 在线程池中运行耗时操作
            embeddings, energies, _ = await loop.run_in_executor(
                pool, 
                extract_speaker_embeddings,
                audio_data, 
                sample_rate
            )
            target_embedding = embeddings[0]
            
            processing_time = time.time() - t_5s_start
            logging.info(f"目标说话人声纹提取完成，耗时: {processing_time:.3f}s")
            logging.info("已提取目标说话人的声纹特征")
            
            return target_embedding, processing_time
    except Exception as e:
        logging.error(f"错误：目标说话人声纹提取失败 - {str(e)}")
        return None, None

def crossfade(chunk1, chunk2, overlap_length=100):
    """应用交叉淡入淡出效果"""
    if len(chunk1) == 0:
        return chunk2
    if len(chunk2) == 0:
        return chunk1
        
    # 确保overlap_length不超过任一chunk的长度
    overlap_length = min(overlap_length, len(chunk1), len(chunk2))
    
    # 创建淡入淡出窗口
    fade_in = np.linspace(0, 1, overlap_length)
    fade_out = np.linspace(1, 0, overlap_length)
    
    # 应用交叉淡入淡出
    chunk1[-overlap_length:] *= fade_out
    chunk2[:overlap_length] *= fade_in
    
    # 拼接音频
    result = np.concatenate([chunk1[:-overlap_length], chunk1[-overlap_length:] + chunk2[:overlap_length], chunk2[overlap_length:]])
    return result

async def process_audio_stream(audio_chunks, sample_rate=16000, chunk_duration=0.6):
    """处理音频流的主函数"""
    try:
        global target_embedding  # 使用全局变量存储目标说话人声纹
        
        # 初始化变量
        collected_samples_count = 0
        five_seconds_samples = int(5 * sample_rate)
        target_embedding = None
        embedding_future = None
        timing_stats = {
            'separation': [],
            'sv_embedding': [],
            'asr': [],
            'total_chunk': [],
            'initial_5s_processing': None
        }
        
        # 初始化缓存
        cache = {}
        chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms
        encoder_chunk_look_back = 4
        decoder_chunk_look_back = 1
        separated_speech = np.array([])
        
        # 获取总的chunk数量
        total_chunk_num = len(audio_chunks)
        
        for i, speech_chunk in enumerate(audio_chunks):
            t_chunk_start = time.time()
            
            # 前5秒：收集音频块并启动异步处理
            if collected_samples_count < five_seconds_samples:
                # 收集音频块
                if 'collected_chunks' not in locals():
                    collected_chunks = []
                collected_chunks.append(speech_chunk)
                
                # 更新已收集的样本数
                collected_samples_count += len(speech_chunk)
                
                # 如果刚好达到或超过5秒，启动异步处理
                if collected_samples_count >= five_seconds_samples and target_embedding is None and embedding_future is None:
                    # 合并所有收集的音频块
                    five_second_audio = np.concatenate(collected_chunks)
                    
                    # 启动异步处理
                    embedding_future = asyncio.ensure_future(
                        extract_target_embedding(five_second_audio, sample_rate)
                    )
                    
                    # 清理不再需要的数据
                    del collected_chunks
        
            # 检查异步处理是否完成
            if embedding_future and embedding_future.done():
                target_embedding, processing_time = await embedding_future
                timing_stats['initial_5s_processing'] = processing_time
                embedding_future = None
                logging.info("目标说话人声纹提取完成！")
        
            # 如果目标说话人尚未确定，直接处理原始音频
            if target_embedding is None:
                is_final = i == total_chunk_num - 1
                t_asr_start = time.time()
                res = asr_model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, 
                                   encoder_chunk_look_back=encoder_chunk_look_back,
                                   decoder_chunk_look_back=decoder_chunk_look_back)
                timing_stats['asr'].append(time.time() - t_asr_start)
                logging.info("前5秒识别结果: %s", res)
                
                # 更新separated_speech为原始音频
                separated_speech = crossfade(separated_speech, speech_chunk)
                
                # 模拟实时处理的延迟
                await asyncio.sleep(chunk_duration)
                continue
        
            # 后续音频：使用目标说话人声纹进行过滤
            t_sep_start = time.time()
            embeddings, energies, separated_chunks = extract_speaker_embeddings(speech_chunk, sample_rate)
            timing_stats['separation'].append(time.time() - t_sep_start)
            
            # 计算相似度并选择目标说话人
            t_sv_start = time.time()
            target_sim = []
            threshold = 0.3
            for emb in embeddings:
                v1 = emb.flatten()
                v2 = target_embedding.flatten()
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                target_sim.append(sim)
            timing_stats['sv_embedding'].append(time.time() - t_sv_start)
            
            # 选择相似度最高的说话人
            max_sim_idx = np.argmax(target_sim)
            max_sim = target_sim[max_sim_idx]
            
            # 如果相似度低于阈值，插入静音
            if max_sim < threshold:
                # 生成与当前音频块等长的静音
                silence = np.zeros_like(speech_chunk)
                
                # 更新separated_speech
                separated_speech = crossfade(separated_speech, silence)
                
                # ASR处理静音
                is_final = True
                t_asr_start = time.time()
                res = asr_model.generate(input=silence, cache=cache, is_final=is_final, chunk_size=chunk_size, 
                                   encoder_chunk_look_back=encoder_chunk_look_back,
                                   decoder_chunk_look_back=decoder_chunk_look_back)
                timing_stats['asr'].append(time.time() - t_asr_start)
                logging.info("无目标说话人: %s", res)
            else:
                # 选择目标说话人的音频
                selected_speech = separated_chunks[max_sim_idx]
                
                # 更新separated_speech
                separated_speech = crossfade(separated_speech, selected_speech)
                
                # ASR处理选中的说话人音频
                is_final = i == total_chunk_num - 1
                t_asr_start = time.time()
                res = asr_model.generate(input=selected_speech, cache=cache, is_final=is_final, chunk_size=chunk_size, 
                                   encoder_chunk_look_back=encoder_chunk_look_back,
                                   decoder_chunk_look_back=decoder_chunk_look_back)
                timing_stats['asr'].append(time.time() - t_asr_start)
                logging.info("目标说话人识别结果: %s", res)

            timing_stats['total_chunk'].append(time.time() - t_chunk_start)
            
            # 模拟实时处理的延迟
            await asyncio.sleep(chunk_duration)

        # 保存处理后的音频
        output_wav = os.path.join("example", "target_speaker_separated.wav")
        soundfile.write(output_wav, separated_speech, sample_rate)
        logging.info(f"\n目标说话人的分离音频已保存至: {output_wav}")

        return timing_stats
    except Exception as e:
        logging.error(f"错误：音频处理失败 - {str(e)}")
        return None

def preload_models():
    """预加载所有需要的模型"""
    try:
        logging.info("\n开始预加载模型...")
        
        # 加载ASR模型
        logging.info("1. 加载ASR模型...")
        from funasr import AutoModel
        global asr_model
        asr_model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4", 
                            disable_update=True)  # 禁用更新检查
        logging.info("ASR模型加载完成")
        
        # 加载说话人分离模型
        logging.info("\n2. 加载说话人分离模型...")
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        global separation_pipeline
        separation_pipeline = pipeline(
            task=Tasks.speech_separation,
            model='iic/speech_mossformer2_separation_temporal_8k',
            model_revision='v0.9.0'  # 明确指定版本
        )
        logging.info("说话人分离模型加载完成")
        
        # 加载声纹识别模型
        logging.info("\n3. 加载声纹识别模型...")
        global sv_pipeline
        sv_pipeline = pipeline(
            task='speaker-verification',
            model='iic/speech_eres2netv2_sv_zh-cn_16k-common',
            model_revision='v1.0.2'
        )
        logging.info("声纹识别模型加载完成")
        logging.info("\n所有模型加载完成！\n")
        return True
    except Exception as e:
        logging.error(f"错误：模型加载失败 - {str(e)}")
        return False

async def main():
    """主函数"""
    try:
        # 预加载所有模型
        if not preload_models():
            logging.error("错误：模型加载失败，程序退出")
            return
        
        # 设置音频参数
        sample_rate = 16000  # 采样率
        chunk_duration = 0.6  # 每个音频块的持续时间（秒）
        chunk_size = int(sample_rate * chunk_duration)  # 每个音频块的样本数
        
        # 从文件加载音频数据
        logging.info("加载音频文件...")
        audio_file = "example/test.wav"
        if not os.path.exists(audio_file):
            logging.error(f"错误：音频文件不存在 - {audio_file}")
            return
            
        audio_data, sr = soundfile.read(audio_file)
        if sr != sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
        
        # 将音频数据分成块
        audio_chunks = [audio_data[i:i + chunk_size] for i in range(0, len(audio_data), chunk_size)]
        
        # 处理音频流
        timing_stats = await process_audio_stream(audio_chunks, sample_rate, chunk_duration)
        if timing_stats is None:
            return
            
        # 打印性能统计
        logging.info("\n=== 性能统计 ===")
        if timing_stats['initial_5s_processing'] is not None:
            logging.info(f"\n目标说话人声纹提取耗时: {timing_stats['initial_5s_processing']:.3f}s")
        
        def print_stats(name, data):
            if len(data) == 0:
                return
            mean = np.mean(data)
            median = np.median(data)
            max_val = np.max(data)
            min_val = np.min(data)
            std = np.std(data)
            logging.info(f"\n{name}统计:")
            logging.info(f"  平均耗时: {mean:.3f}s")
            logging.info(f"  中位数: {median:.3f}s")
            logging.info(f"  最大耗时: {max_val:.3f}s")
            logging.info(f"  最小耗时: {min_val:.3f}s")
            logging.info(f"  标准差: {std:.3f}s")
            logging.info(f"  总样本数: {len(data)}")
        
        print_stats("说话人分离", timing_stats['separation'])
        print_stats("声纹提取", timing_stats['sv_embedding'])
        print_stats("ASR识别", timing_stats['asr'])
        print_stats("总处理时间", timing_stats['total_chunk'])
    except Exception as e:
        logging.error(f"错误：程序执行失败 - {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
