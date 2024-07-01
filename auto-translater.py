# -*- coding: utf-8 -*-
import os
import sys
import re
import yaml  # pip install PyYAML
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import logging
import tensorflow as tf

logging.basicConfig(level=logging.ERROR)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1,"
gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPU: ", gpus)
pipeline_ins = pipeline(task=Tasks.translation, model="damo/nlp_csanmt_translation_zh2en", device="cuda")

# 设置翻译的路径
dir_to_translate = "testdir/to-translate"
dir_translated = {
    "en": "testdir/docs/en",
}

processed_list = "processed_list.txt"

def translate_text(text, lang, type):
    print("Input:  \"", text, "\"");

    # 开头的空格要保留，影响markdown格式
    spaces = ""
    match = re.match(r'^\s+', text)
    if(match):
        spaces = match.group(0)

    outputs = pipeline_ins(input=text)
    output = spaces + outputs['translation']
    print("Output: \"", output, "\"")
    return output

def translate_file(input_file, filename, lang):
    print(f"Translating into {lang}: {filename}")
    sys.stdout.flush()

    # 定义输出文件
    if lang in dir_translated:
        output_dir = dir_translated[lang]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, filename)

    # 读取输入文件内容
    with open(input_file, "r", encoding="utf-8") as f:
        input_text = f.read()

    # 拆分文章
    punctuation_pattern = re.compile('[。！？；#!?;:\'"*{}()-+\n|-]')
    paragraphs = []
    matches = list(punctuation_pattern.finditer(input_text))
    start_index = 0
    for match in matches:  
        # 添加分隔符之前的文本（如果有的话）  
        if match.start() > start_index:
            paragraphs.append(input_text[start_index:match.start()])  
        # 添加分隔符本身  
        paragraphs.append(match.group())  
        # 更新开始索引为分隔符之后的位置  
        start_index = match.end()

    if start_index < len(input_text):  
        paragraphs.append(input_text[start_index:]) 

    input_text = ""
    output_paragraphs = []

    pattern = re.compile(r'[\u4e00-\u9fff]+')
    for paragraph in paragraphs:
        match = pattern.search(paragraph)
        if match:
            output_paragraphs.append(translate_text(paragraph, lang, "main-body"))
        else:
            output_paragraphs.append(paragraph)

    # 将输出段落合并为字符串
    output_text = "".join(output_paragraphs)

    # 写入输出文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)

# 按文件名称顺序排序
file_list = []
for root, dirs, files in os.walk(dir_to_translate):  
        for file in files:  
            if file.endswith(".md"):
                file_list.append(os.path.join(root, file))  

sorted_file_list = sorted(file_list)

try:
    # 创建一个外部列表文件，存放已处理的 Markdown 文件名列表
    if not os.path.exists(processed_list):
        with open(processed_list, "w", encoding="utf-8") as f:
            print("processed_list created")

    # 遍历目录下的所有.md文件，并进行翻译
    for filename in sorted_file_list:
        if filename.endswith(".md"):
            input_file = filename

            # 读取 Markdown 文件的内容
            with open(input_file, "r", encoding="utf-8") as f:
                md_content = f.read()

            # 读取processed_list内容
            with open(processed_list, "r", encoding="utf-8") as f:
                processed_list_content = f.read()

            if filename in processed_list_content:  # 不进行翻译
                print(f"Pass the post in processed_list: {filename}")
            translate_file(input_file, filename, "en")

            # 将处理完成的文件名加到列表，下次跳过不处理
            if filename not in processed_list_content:
                print(f"Added into processed_list: {filename}")
                with open(processed_list, "a", encoding="utf-8") as f:
                    f.write("\n")
                    f.write(filename)

    # 所有任务完成的提示
    print("Congratulations! All files processed done.")

except Exception as e:
    # 捕获异常并输出错误信息
    print(f"An error has occurred: {e}")
    raise SystemExit(1)  # 1 表示非正常退出，可以根据需要更改退出码