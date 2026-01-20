import re, os,sys
import torch
import argparse
from Bio import SeqIO
import itertools
from collections import Counter
import pandas as pd
from model2 import GNNm_DNABERT  # 使用model2
from metrics import Metric
from graphData import mRNAGraphDataset,mRNAGraphDataset_1
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--input_fasta', default="data/test_data2.fasta", help='input fasta')
parser.add_argument('--output_path', default=".", help='output path')
parser.add_argument('--device', default="cuda", help='cpu or cuda')
parser.add_argument('--model_path', default="model/best.pth", help='model checkpoint path')
opt = parser.parse_args()

n_classes = 4
device = opt.device

def RNAFold_folding(fasta_file):
  # Compute structure
    os.system(f"RNAfold --noPS {fasta_file} > RNAfold_tmp_dot.fasta")
    file ='RNAfold_tmp_dot.fasta'
    with open(file) as f:
        records=f.read()

    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    seq_dotbracket = {}
    for fasta in records:
        valueList=[]
        array = fasta.split('\n')
        sequence,dot_bracket =array[1].replace('U','T'),array[2]
        dot_bracket_list=dot_bracket.split()

        ev = float(re.search(r'\(\s*(-?\d+\.?\d*)\)', dot_bracket).group(1))
        valueList.append(dot_bracket_list[0])
        valueList.append(ev)
        seq_dotbracket[sequence]=valueList
  
    return seq_dotbracket

def Linear_folding(fasta_file):
  out_fasta_name = "Linearfold_dot"
  with open(fasta_file, "r") as handle:
    records = list(SeqIO.parse(handle, "fasta"))
  for row in records:
    seq = str(row.seq)
    lnc=str(row.description)
    with open("LinearFold_tmp.fasta", "w") as ofile: 
      ofile.write(f">{lnc}\n{seq}\n")
    tmp_name='LinearFold_tmp.fasta'
    os.system(f"cat {tmp_name} | /home/lyh/LinearFold/home/tu/mypro/external_algorithms/LinearFold/bin/linearfold_v > LinearFold_tmp.dot")
    out_file_name = "clean_tmp.dot"
    in_lines = open("LinearFold_tmp.dot","r").readlines()
    with open(out_file_name,"w") as out_file:
      for line in in_lines:
        if ">" in line:
          out_file.write(':'.join(line.split(":")[1:]).strip() + "\n")
        else:
          out_file.write(line)

    os.system("cat " + out_file_name + " >> " + out_fasta_name + ".fasta") 
  file ='Linearfold_dot.fasta'
  with open(file) as f:
      records1=f.read()

  if re.search('>', records1) == None:
      print('Error: the input file %s seems not in FASTA format!' % file)
      sys.exit(1)
  records1 = records1.split('>')[1:]
  seq_dotbracket = {}
  for fasta in records1:
      valueList=[]
      array = fasta.split('\n')
      sequence,dot_bracket =array[1],array[2]
      dot_bracket_list=dot_bracket.split()

      ev = float(re.search(r'\(\s*(-?\d+\.?\d*)\)', dot_bracket).group(1))
      valueList.append(dot_bracket_list[0])
      valueList.append(ev)
      seq_dotbracket[sequence]=valueList
  return seq_dotbracket

def get_min_sequence_length(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen

def read_nucleotide_sequences(file):
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACGTU-]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        label = header_array[1] if len(header_array) >= 2 else '0'
        label_train = header_array[2] if len(header_array) >= 3 else 'training'
        sequence = re.sub('U', 'T', sequence) 
        fasta_sequences.append([name, sequence, label, label_train])
    return fasta_sequences

def CKSNAP(fastas, gap=5, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if get_min_sequence_length(fastas) < gap + 2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    AA = kw['order'] if kw['order'] != None else 'ACGT'

    encodings = {}
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    for i in fastas:
        seq_key=i[1]
        name, sequence, label = i[0], i[1], i[2]
        code=[]
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings[seq_key]=code
    return encodings

def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer

def Kmer(fastas, k=2, type="DNA", upto=False, normalize=True, **kw):
    encoding = {}
    header = ['#', 'label']
    NA = 'ACGT'
    if type in ("DNA", 'RNA'):
        NA = 'ACGT'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'

    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0

    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                header.append(''.join(kmer))
        for i in fastas:
            seq_key=i[1]
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            code=[]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding[seq_key]=code
    else:
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))
        for i in fastas:
            seq_key=i[1]
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            kmers = kmerArray(sequence, k)
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = []
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding[seq_key]=code
    return encoding


def predictor(model,test_loader,test_loader1,test_records,outputpath):
    model.eval()

    with torch.no_grad():

        y_pred_list2 = []
        y_prob_list = []  # 添加:保存概率值
        y_true_list = []
        t = torch.Tensor([0.5, 0.5, 0.5, 0.5]).to(device)
        dataloader_iterator = iter(test_loader1)
        for i, data in enumerate(test_loader):
            # 提取DNA序列
            sequences = [item.sequence for item in data.to_data_list()]

            data.x=data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.edge_attr = data.edge_attr.to(device)
            data.batch = data.batch.to(device)
            targets = data.y.to(device)
            y_true_list.append(targets)
            
            try:
                data1 = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(test_loader1)
                data1 = next(dataloader_iterator)
            data1.x=data1.x.to(device)  
            data1.edge_index = data1.edge_index.to(device)
            data1.edge_attr = data1.edge_attr.to(device)
            data1.batch = data1.batch.to(device)
            
            out = model(data, data1, sequences)  # 传入sequences
            out = torch.sigmoid(out)  # 换损失添加

            pred = (out > t).float() * 1
            y_pred_list2 += list(pred)
            y_prob_list += list(out)  # 添加:保存概率

        y_pred_tensor2 = torch.stack(y_pred_list2)
        y_predClass=y_pred_tensor2.cpu().numpy()

        y_prob_tensor = torch.stack(y_prob_list)  # 添加
        y_prob = y_prob_tensor.cpu().numpy()      # 添加
        
        # 计算评估指标
        y_true_tensor = torch.cat(y_true_list, dim=0)
        y_true = y_true_tensor.cpu().numpy()
        
        metric = Metric(4)
        metric.reset()
        metric.update(y_predClass, y_true)
        accuracy = metric.Accuracy()
        print(f"Test Accuracy: {accuracy:.4f}")

        subset_acc = metric.accuracy_subset(threash=0.5)
        print(f"Subset Accuracy: {subset_acc:.4f}")
        
        macro_auc, _ = metric.MacroAUC()
        print(f"Macro AUC: {macro_auc:.4f}")
        
        hamming_dist = metric.hamming_distance(threash=0.5)
        print(f"Hamming Distance: {hamming_dist:.4f}")
        
        f1_example, f1_micro, f1_macro = metric.f1(threash=0.5)
        print(f"F1-Example: {f1_example:.4f}, F1-Micro: {f1_micro:.4f}, F1-Macro: {f1_macro:.4f}")

        # 计算每个类别的AUC
        from sklearn.metrics import roc_auc_score
        label_names = ['Nucleus', 'Exosome', 'Cytoplasm', 'Microvesicle']
        print("\nPer-class AUC:")
        for i, label_name in enumerate(label_names):
            try:
                class_auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                print(f"  {label_name}: {class_auc:.4f}")
            except ValueError:
                print(f"  {label_name}: N/A (only one class present)")

                
        # 修改输出DataFrame
        resultDF=pd.DataFrame({
            # 'Nucleus': y_predClass[:, 0].astype(bool).astype(str),
            'Nucleus': y_prob[:, 0],  # 添加概率
            # 'Exosome': y_predClass[:, 1].astype(bool).astype(str),
            'Exosome': y_prob[:, 1],  # 添加概率
            # 'Cytoplasm': y_predClass[:, 2].astype(bool).astype(str),
            'Cytoplasm': y_prob[:, 2],  # 添加概率
            # 'Microvesicle': y_predClass[:, 3].astype(bool).astype(str),
            'Microvesicle': y_prob[:, 3]  # 添加概率
        })

        
        # resultDF=pd.DataFrame(y_predClass,columns=['Nucleus', 'Exosome', 'Cytoplasm', 'Microvesicle'])
        # resultDF=resultDF.astype(bool).astype(str)
        seqDes=[str(record.description) for record in test_records]
        resultDF.insert(0,'SequenceID',seqDes)
        resultDF.to_csv(outputpath+'/results_deepmodel.csv',index=False) 


def main(): 
   
    model = GNNm_DNABERT(
        n_features=10, 
        hidden_dim=64, 
        n_classes=4,
        n_conv_layers=3,
        conv_type1="GIN",
        conv_type2="GIN",
        dropout=0.1, 
        batch_norm=True, 
        batch_size=1,
        dnabert_path="DNABERT-2-117M",
        freeze_bert=True
    )
    model.load_state_dict(torch.load(opt.model_path, map_location=device))
    model.to(device)
    
    with open(opt.input_fasta, "r") as handle1:
        test_records = list(SeqIO.parse(handle1, "fasta"))
    
    RNAfoldDict=RNAFold_folding(opt.input_fasta)   
    os.system(f"rm RNAfold_tmp_dot.fasta")
    LinearfoldDict=Linear_folding(opt.input_fasta)
    os.system(f"rm LinearFold_tmp.fasta")
    os.system(f"rm Linearfold_dot.fasta")
    os.system(f"rm clean_tmp.dot")
    os.system(f"rm LinearFold_tmp.dot")

    fastas= read_nucleotide_sequences(opt.input_fasta)
    print(f"Loaded {len(fastas)} sequences")
    
    kw = {'order': 'ACGT'}
    CKSNAP_Dict=CKSNAP(fastas=fastas,gap=5,**kw)
    Kmer_Dict=Kmer(fastas, k=5, type="RNA", upto=True, normalize=True, **kw)
    
    test_set = mRNAGraphDataset(test_records, RNAfoldDict,CKSNAP_Dict,Kmer_Dict)
    test_set1 = mRNAGraphDataset_1(test_records, LinearfoldDict)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,drop_last=True)
    test_loader1 = DataLoader(test_set1, batch_size=1, shuffle=False,drop_last=True)
    
    predictor(model,test_loader,test_loader1,test_records,opt.output_path)

if __name__ == "__main__":
    main()
