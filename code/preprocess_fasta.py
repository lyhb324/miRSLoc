# -*- coding: utf-8 -*-
import sys

# 定义标签顺序
label_order = ["Nucleus", "Exosome", "Cytoplasm", "Microvesicle"]

# 输入输出文件路径
input_fasta = "./data/miRNA_Localization_train.fasta"   # 替换成你的文件名
output_fasta = "./data/train.fasta"

with open(input_fasta, "r") as infile, open(output_fasta, "w") as outfile:
    for line in infile:
        line = line.strip()
        if line.startswith(">"):
            # 拆分 header
            parts = line[1:].split("|")
            # parts[1] 是 mir id, parts[2] 是标签, parts[3] 是长度
            mir_id = parts[1]
            labels = parts[2].split(",")  # 多标签可能用逗号分隔
            seq_len = parts[3]
            
            # 构建多标签编码
            label_code = "".join(["1" if label in labels else "0" for label in label_order])
            
            # 写入新的 header
            new_header = f">{label_code},{mir_id},{seq_len}\n"
            outfile.write(new_header)
        else:
            # 写入序列行
            outfile.write(line + "\n")

print(f"转换完成，结果已保存到 {output_fasta}")
