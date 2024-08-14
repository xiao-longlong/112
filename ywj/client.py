import os
import sys
from ywj.ClientExecutor import ASROnlineClientExecutor
from ywj.record import record


def always(Path):
    asrclient_executor = ASROnlineClientExecutor()
    print("开始检测{}文件夹".format(Path))
    if not os.path.exists(Path):
        os.makedirs(Path)
    while True:
        files = os.listdir(Path)
        # 筛选后缀为.wav的文件
        wav_files = [file for file in files if file.endswith(".wav")]

        if len(wav_files) == 0:
            # 清空文件夹
            continue

        for wav_file in wav_files:
            wave_path = os.path.join(Path, wav_file)
            res = asrclient_executor(input=wave_path, server_ip="furina.x-contion.top", port=8090, sample_rate=16000, lang="zh_cn", audio_format="wav")
            os.remove(wave_path)
            print(res)


def once(path):
    asrclient_executor = ASROnlineClientExecutor()
    res = asrclient_executor(input=path, server_ip="furina.x-contion.top", port=8090, sample_rate=16000, lang="zh_cn", audio_format="wav")
    return res

def detect_keywords_in_file(txt_result):
    keyword_groups = [
        {"集群侦查": "1", "单对单侦查": "2", "构建": "5"}, 
        {"飞机": "1", "坦克": "2", "装甲车": "3",  "爱国者": "4"},  
        {"背景": "71", "有遮挡": "72", "光照多样": "73", "尺度变换": "74",
        "尺度差异": "75", "跨场景": "76", "迷彩": "77" , "夜视": "78",},  
    ]
    lines = txt_result
    code_positions = ["0", "0", "00"]
    for i, group in enumerate(keyword_groups):
        for keyword, code in group.items():
            if keyword in lines:
                code_positions[i] = code

    int_list = [int(digit) for item in code_positions for digit in item]
    return int_list

def speech_recognize():
    record("/home/wxl/wxlcode/112/ywj/output.wav", time=5)  # modify time to how long you want
    Path1 = "/home/wxl/wxlcode/112/ywj/output.wav"
    res = once(Path1)
    print(res)
    int_list = detect_keywords_in_file(res)
    return int_list


if __name__ == "__main__":

    record("/home/wxl/wxlcode/112/ywj/output.wav", time=5)  # modify time to how long you want
    Path1 = "/home/wxl/wxlcode/112/ywj/output.wav"
    res = once(Path1)
    int_list = detect_keywords_in_file(res)
    print(int_list)
