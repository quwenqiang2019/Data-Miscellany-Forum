import pandas as pd
import re
import string
import Levenshtein
import numpy as np

def calculate_error_rates(references, hypotheses):
    # 计算Levenshtein距离和编辑操作
    sum = len(references)
    wer = 0
    WER = 0
    insertion_rate = 0
    deletion_rate = 0
    substitution_rate = 0
    accuracy = 0
    for i in range(sum):
        editops = Levenshtein.editops(references[i], hypotheses[i])
        print(editops)
        distance = Levenshtein.distance(references[i], hypotheses[i])
        print(distance)
        wer += distance / len(references[i])

        # 计算插入率、删除率和替换率
        insertions = 0
        deletions = 0
        substitutions = 0
        correctnum = 0

        for op in editops:
            if op[0] == 'insert':
                insertions += 1
            elif op[0] == 'delete':
                deletions += 1
            elif op[0] == 'replace':
                substitutions += 1

        correctnum += len(references[i]) - distance
        accuracy += correctnum/len(references[i])

        # 计算总字符数
        total_chars = len(references[i])
        WER += (insertions + deletions + substitutions) / total_chars

        # 计算插入率、删除率和替换率
        insertion_rate += insertions / total_chars
        deletion_rate += deletions / total_chars
        substitution_rate += substitutions / total_chars


    wer /= sum
    WER /= sum
    accuracy /= sum
    insertion_rate /= sum
    deletion_rate /= sum
    substitution_rate /= sum

    return wer, WER, accuracy, insertion_rate, deletion_rate, substitution_rate



def levenshtein_distance(hypothesis: list, reference: list):
    """编辑距离
    计算两个序列的levenshtein distance，可用于计算 WER/CER

    词错率（Word Error Rate, WER）是一项用于评价ASR性能的重要指标，用来评价预测文本与标准文本之间错误率，因此词错率最大的特点是越小越好。
    像英语、阿拉伯语语音转文本或语音识别任务中研究者常用WER衡量ASR效果好坏。
    因为英文语句中句子的最小单位是单词，而中文语句中的最小单位是汉字，
    因此在中文语音转文本任务或中文语音识别任务中使用字错率（Character Error Rate, CER）来衡量中文ASR效果好坏。
    两者计算方式相同，为行文统一，下文统一使用WER表示该性能。

    WER = (S+D+I)/N = (S+D+I)/(S+D+C)
    参考资料：
        https://www.cuelogic.com/blog/the-levenshtein-algorithm
        https://martin-thoma.com/word-error-rate-calculation/

    N: number         真实序列的字数
    C: correct        预测序列识别正确的字数
    W: wrong          预测序列识别错误的字数
    I: insert         为了和真实序列一致，需要插入的数量
    D: delete         为了和真实序列一致，需要删除的数量
    S: substitution   为了和真实序列一致，需要替换的数量

    :param hypothesis: 预测序列
    :param reference: 真实序列
    :return: 1: 错误操作，所需要的 S，D，I 操作的次数;
             2: ref 与 hyp 的所有对齐下标
             3: 返回N、C、W、S、D、I 各自的数量
    """
    len_hyp = len(hypothesis)
    len_ref = len(reference)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

    # 记录所有的操作，0-equal；1-insertion；2-deletion；3-substitution
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    # 生成 cost 矩阵和 operation矩阵，i:外层hyp，j:内层ref
    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i-1] == reference[j-1]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution = cost_matrix[i-1][j-1] + 1
                insertion = cost_matrix[i-1][j] + 1
                deletion = cost_matrix[i][j-1] + 1

                # compare_val = [insertion, deletion, substitution]   # 优先级
                compare_val = [substitution, insertion, deletion]   # 优先级

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []  # 保存 hyp与ref 中所有对齐的元素下标
    i = len_hyp
    j = len_ref
    nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:     # correct
            if i-1 >= 0 and j-1 >= 0:
                match_idx.append((j-1, i-1))
                nb_map['C'] += 1

            # 出边界后，这里仍然使用，应为第一行与第一列必然是全零的
            i -= 1
            j -= 1
        # elif ops_matrix[i_idx][j_idx] == 1:   # insert
        elif ops_matrix[i_idx][j_idx] == 2:   # insert
            i -= 1
            nb_map['I'] += 1
        # elif ops_matrix[i_idx][j_idx] == 2:   # delete
        elif ops_matrix[i_idx][j_idx] == 3:   # delete
            j -= 1
            nb_map['D'] += 1
        # elif ops_matrix[i_idx][j_idx] == 3:   # substitute
        elif ops_matrix[i_idx][j_idx] == 1:   # substitute
            i -= 1
            j -= 1
            nb_map['S'] += 1

        # 出边界处理
        if i < 0 and j >= 0:
            nb_map['D'] += 1
        elif j < 0 and i >= 0:
            nb_map['I'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    nb_map["W"] = wrong_cnt

    try:
        wer = (nb_map["S"] + nb_map["D"] + nb_map["I"])/(nb_map["S"] + nb_map["D"] + nb_map["C"])
    except:
        wer = float('inf')

    # print("ref: %s" % " ".join(reference))
    # print("hyp: %s" % " ".join(hypothesis))
    # print(nb_map)
    # print("match_idx: %s" % str(match_idx))
    return wrong_cnt, match_idx, nb_map, wer


def tiqu(dialogue):
    # 使用正则表达式找到所有客户说的话
    customer_replies = re.findall(r'客户[:；：](.*?)(?=\n|$)', dialogue, re.DOTALL)
    # print(customer_replies)
    chinese_punctuation = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏'
    punctuation = string.punctuation + chinese_punctuation
    # 去掉标点符号
    customer_replies_cleaned = [''.join([char for char in reply if char not in punctuation]) for reply in
                                customer_replies]
    # print(customer_replies_cleaned)
    # 组装成一个字符串
    customer_string = ''.join(customer_replies_cleaned)
    customer_string = customer_string.replace(" ", "")
    # 输出结果
    # print(customer_string)

    return customer_string

def main(file_name):
    df = pd.read_excel(f"{file_name}.xlsx")
    # print(df)

    for index, row in df[:].iterrows():
        print(index)
        print(row)
        changshang = row['厂商']
        paraformer = row['paraformer']
        ren_gong = row['人工翻译']
        # print(changshang)
        print(tiqu(changshang))
        print("====================")
        # print(paraformer)
        print(tiqu(paraformer))
        print("====================")
        # print(ren_gong)
        print(tiqu(ren_gong))


        df.at[index, 'kehu_changshang'] = tiqu(changshang)
        df.at[index, 'kehu_paraformer'] = tiqu(paraformer)
        df.at[index, 'kehu_ren_gong'] = tiqu(ren_gong)

        wrong_cnt, match_idx, nb_map, wer = levenshtein_distance(list(tiqu(changshang)),list(tiqu(ren_gong)))
        N_changshang = nb_map["N"]
        C_changshang = nb_map["C"]
        W_changshang = nb_map["W"]
        I_changshang = nb_map["I"]
        D_changshang = nb_map["D"]
        S_changshang = nb_map["S"]
        wer_score_changshang = wer

        df.at[index, 'N_changshang'] = N_changshang
        df.at[index, 'C_changshang'] = C_changshang
        df.at[index, 'W_changshang'] = W_changshang
        df.at[index, 'I_changshang'] = I_changshang
        df.at[index, 'D_changshang'] = D_changshang
        df.at[index, 'S_changshang'] = S_changshang
        df.at[index, 'wer_score_changshang'] = wer_score_changshang


        print("Word Error Rate (WER):", wer_score_changshang)

        wrong_cnt, match_idx, nb_map, wer = levenshtein_distance(list(tiqu(paraformer)),list(tiqu(ren_gong)))

        N_paraformer = nb_map["N"]
        C_paraformer = nb_map["C"]
        W_paraformer = nb_map["W"]
        I_paraformer = nb_map["I"]
        D_paraformer = nb_map["D"]
        S_paraformer = nb_map["S"]
        wer_score_paraformer = wer

        df.at[index, 'N_paraformer'] = N_paraformer
        df.at[index, 'C_paraformer'] = C_paraformer
        df.at[index, 'W_paraformer'] = W_paraformer
        df.at[index, 'I_paraformer'] = I_paraformer
        df.at[index, 'D_paraformer'] = D_paraformer
        df.at[index, 'S_paraformer'] = S_paraformer
        df.at[index, 'wer_score_paraformer'] = wer_score_paraformer


        print("Word Error Rate (WER):", wer_score_paraformer)

    print(df)
    df.to_excel(f"{file_name}-评测结果.xlsx", index=False)


if __name__ == "__main__":
    wrong_cnt, match_idx, nb_map, wer = levenshtein_distance(list("你吃了么"), list("你吃了吗"))
    print(wrong_cnt, match_idx, nb_map, wer)

    wrong_cnt, match_idx, nb_map, wer = levenshtein_distance(list("今天天气很好啊"), list("今天天气很好"))
    print(wrong_cnt, match_idx, nb_map, wer)

    # main("单纯文字转译错误对比v2")
    # main("噪音错误对比v2")
    # main("方言错误对比v2")