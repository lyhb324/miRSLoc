import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, hamming_loss
import warnings

# 忽略可能的警告（如除零警告）
warnings.filterwarnings('ignore')



def Custom_Accuracy(y_hat, y):
    """ 用户提供的自定义准确率指标 (Jaccard-based) """
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



print("正在加载和处理数据...")

# 加载文件
lgb_df = pd.read_csv('./machine_learning/independent_test_predictions.csv')
dnabert_df = pd.read_csv('results_deepmodel.csv')
test_df = pd.read_csv('./machine_learning/test.csv')


# 处理 DNABERT 的 SequenceID
def clean_dnabert_id(row_val):
    parts = str(row_val).split(',')
    if len(parts) >= 2:
        return parts[1].strip()
    return row_val


dnabert_df['clean_id'] = dnabert_df['SequenceID'].apply(clean_dnabert_id)

# lightgbm 使用 'id', dnabert 使用 'clean_id'
pred_merged = pd.merge(lgb_df, dnabert_df, left_on='id', right_on='clean_id', suffixes=('_lgb', '_bert'))

# 将真实标签 (Test) 合并进来，确保行一一对应
# 这样我们只需要做一次对齐，后面循环时不需要再 merge
full_df = pd.merge(pred_merged, test_df[['id', 'label']], on='id')

# 定义类别列
cols = ['Cytoplasm', 'Exosome', 'Microvesicle', 'Nucleus']


# 准备真实标签 y_true
print("正在构建真实标签矩阵...")
full_df['label_list'] = full_df['label'].apply(lambda x: [item.strip() for item in str(x).split(',')])
mlb = MultiLabelBinarizer(classes=cols)
y_true = mlb.fit_transform(full_df['label_list'])

# 提取预测概率矩阵 (转换为 numpy 数组以加速计算)
# 形状: [样本数, 类别数]
lgb_probs = full_df[[c + '_lgb' for c in cols]].values
bert_probs = full_df[[c + '_bert' for c in cols]].values


alpha = 0.75


# 4.1 加权融合
# 公式: alpha * LGBM + (1-alpha) * DNABERT
y_pred_proba = alpha * lgb_probs + (1 - alpha) * bert_probs

# 4.2 应用阈值 0.5
y_pred = (y_pred_proba >= 0.5).astype(int)

# 4.3 计算指标
test_exact = accuracy_score(y_true, y_pred)
test_custom = Custom_Accuracy(y_pred, y_true)
test_f1 = f1_score(y_true, y_pred, average='samples')
test_auc = roc_auc_score(y_true, y_pred_proba, average='macro')
test_hamming = hamming_loss(y_true, y_pred)

# ==========================================
# 5. 输出结果
# ==========================================
print("\n" + "=" * 40)
print(f"评估结果 (Alpha = {alpha}):")
print("=" * 40)
print(f"Exact Accuracy   : {test_exact:.4f}")
print(f"Custom Accuracy  : {test_custom:.4f}")
print(f"F1 Score         : {test_f1:.4f}")
print(f"Macro AUC        : {test_auc:.4f}")
print(f"Hamming Loss     : {test_hamming:.4f}")
print("=" * 40)

# 可选：保存预测结果
results_df = pd.DataFrame({
    'id': full_df['id'],
    'true_label': full_df['label'],
    'Cytoplasm_prob': y_pred_proba[:, 0],
    'Exosome_prob': y_pred_proba[:, 1],
    'Microvesicle_prob': y_pred_proba[:, 2],
    'Nucleus_prob': y_pred_proba[:, 3],
    'Cytoplasm_pred': y_pred[:, 0],
    'Exosome_pred': y_pred[:, 1],
    'Microvesicle_pred': y_pred[:, 2],
    'Nucleus_pred': y_pred[:, 3]
})

output_filename = 'ensemble_predictions.csv'
results_df.to_csv(output_filename, index=False)
print(f"\n预测结果已保存至: {output_filename}")