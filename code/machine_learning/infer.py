import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, hamming_loss


# ==========================================
# 0. 基础函数定义 (必须与训练时一致)
# ==========================================
def Custom_Accuracy(y_hat, y):
    """ 自定义准确率指标 (Jaccard-based) """
    count = 0
    k = 0
    y_hat = np.array(y_hat)
    y = np.array(y)

    for i in range(y.shape[0]):
        p = sum(np.logical_and(y[i], y_hat[i]))
        q = sum(np.logical_or(y[i], y_hat[i]))
        if q == 0:
            k += 1
            continue
        count += p / q

    if (y.shape[0] - k) == 0:
        return 0.0

    return count / (y.shape[0] - k)


def get_basic_seq_features(sequences):
    """ 提取基础序列特征 """
    features = []
    eiip_dict = {'A': 0.1260, 'G': 0.0806, 'C': 0.1340, 'U': 0.1335}
    for seq in sequences:
        length = len(seq)
        if length == 0:
            features.append([0] * 9)
            continue

        c_A, c_C, c_G, c_U = seq.count('A'), seq.count('C'), seq.count('G'), seq.count('U')
        f_A, f_C, f_G, f_U = c_A / length, c_C / length, c_G / length, c_U / length
        gc_content = (c_G + c_C) / length
        avg_eiip = sum([eiip_dict.get(base, 0) for base in seq]) / length

        row = [length, f_A, f_C, f_G, f_U, gc_content, f_A + f_G, f_A + f_C, avg_eiip]
        features.append(row)
    return np.array(features)


# ==========================================
# 1. 主流程
# ==========================================
def main():
    model_path = 'best_lightgbm_model.pkl'

    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}，请先运行训练脚本。")
        return

    print("Step 1: 加载数据 (需要训练集来重建特征空间)...")
    try:
        train_df = pd.read_csv('./train.csv')
        test_df = pd.read_csv('./test.csv')
    except FileNotFoundError:
        print("错误: 找不到数据文件，请检查路径")
        return

    # -----------------------------------------------------------
    # 关键步骤：重建特征工程 (Feature Engineering Reconstruction)
    # -----------------------------------------------------------
    print("Step 2: 重建特征转换器...")

    # 1. 标签编码 (MLB)
    print("  - 拟合 Label Binarizer...")
    mlb = MultiLabelBinarizer()
    # 必须用训练集拟合，以保证类别顺序一致
    mlb.fit(train_df['label'].apply(lambda x: x.split(',')))
    y_test = mlb.transform(test_df['label'].apply(lambda x: x.split(',')))
    print(f"    类别列表: {mlb.classes_}")

    # 2. K-mer 特征 (CountVectorizer)
    print("  - 拟合 K-mer Vectorizer (这可能需要几秒)...")
    # 这里的参数必须和训练时完全一样！
    cv = CountVectorizer(analyzer='char', ngram_range=(3, 5))
    cv.fit(train_df['sequence'])  # 只拟合训练集
    X_test_kmer = cv.transform(test_df['sequence']).toarray()  # 转换测试集

    # 3. 基础特征 (StandardScaler)
    print("  - 拟合 StandardScaler...")
    X_train_basic_raw = get_basic_seq_features(train_df['sequence'])
    X_test_basic_raw = get_basic_seq_features(test_df['sequence'])

    scaler = StandardScaler()
    scaler.fit(X_train_basic_raw)  # 拟合训练集统计量
    X_test_basic = scaler.transform(X_test_basic_raw)  # 标准化测试集

    # 4. 合并特征
    print("  - 合并特征矩阵...")
    X_test = np.hstack([X_test_basic, X_test_kmer])
    print(f"    测试集维度: {X_test.shape}")

    # -----------------------------------------------------------
    # 加载模型与预测
    # -----------------------------------------------------------
    print(f"\nStep 3: 加载模型 {model_path}...")
    model = joblib.load(model_path)

    print("Step 4: 进行预测...")
    y_test_probas = model.predict_proba(X_test)
    y_test_pred = model.predict(X_test)

    # -----------------------------------------------------------
    # 计算并打印指标
    # -----------------------------------------------------------
    print("\nStep 5: 评估结果")
    test_exact = accuracy_score(y_test, y_test_pred)
    test_custom = Custom_Accuracy(y_test_pred, y_test)
    test_f1 = f1_score(y_test, y_test_pred, average='samples')
    try:
        test_auc = roc_auc_score(y_test, y_test_probas, average='macro')
    except ValueError:
        test_auc = 0.0  # 防止某些类别在测试集中不存在导致报错
    test_hamming = hamming_loss(y_test, y_test_pred)

    print("-" * 30)
    print(f"Exact Accuracy : {test_exact:.4f}")
    print(f"Custom Accuracy: {test_custom:.4f}")
    print(f"F1 Score : {test_f1:.4f}")
    print(f"Macro AUC      : {test_auc:.4f}")
    print(f"Hamming Loss   : {test_hamming:.4f}")
    print("-" * 30)

    # 可选：保存预测结果
    save_path = 'independent_test_predictions.csv'
    # 获取ID，防止报错
    ids = test_df['id'].values if 'id' in test_df.columns else range(len(test_df))
    sub_df = pd.DataFrame(ids, columns=['id'])
    for idx, label in enumerate(mlb.classes_):
        sub_df[label] = y_test_probas[:, idx]

    sub_df.to_csv(save_path, index=False)
    print(f"\n预测概率结果已保存至: {save_path}")


if __name__ == "__main__":
    main()