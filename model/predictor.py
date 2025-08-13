import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

def load_and_prepare_data(control_path, depressed_path, ratio=19):
    """
    加载数据并准备训练集，支持不同比例的样本分布
    ratio: int, 控制组与抑郁组的比例 (默认19:1)
    简化逻辑：
    - 如果控制组数量足够：抑郁组数量 = 控制组数量 / 19
    - 如果抑郁组数量更多：控制组数量 = 抑郁组数量 * 19
    """
    # 加载数据
    with open(control_path, 'rb') as f:
        control_data = pickle.load(f)
    
    with open(depressed_path, 'rb') as f:
        depressed_data = pickle.load(f)
    
    print(f"Control data: {len(control_data)} users")
    print(f"Depressed data: {len(depressed_data)} users")
    
    # 提取所有文本的嵌入
    control_texts = []
    depressed_texts = []
    
    for user_texts in control_data:
        for text in user_texts:
            control_texts.append(text)
    
    for user_texts in depressed_data:
        for text in user_texts:
            depressed_texts.append(text)
    
    print(f"Total control texts: {len(control_texts)}")
    print(f"Total depressed texts: {len(depressed_texts)}")
    
    # 简化的比例调整逻辑
    control_count = len(control_texts)
    depressed_count = len(depressed_texts)
    
    # 计算19:1比例下各组应有的数量
    if control_count >= depressed_count * ratio:
        # 控制组足够多，限制抑郁组数量
        final_depressed = depressed_count
        final_control = depressed_count * ratio
        print(f"Strategy: Using all {final_depressed} depressed texts, limiting control to {final_control}")
    else:
        # 抑郁组相对更多，限制控制组数量  
        final_control = control_count
        final_depressed = control_count // ratio
        print(f"Strategy: Using all {final_control} control texts, limiting depressed to {final_depressed}")
    
    # 确保不超过可用数量
    final_control = min(final_control, control_count)
    final_depressed = min(final_depressed, depressed_count)
    
    # 截取数据
    control_texts = control_texts[:final_control]
    depressed_texts = depressed_texts[:final_depressed]
    
    print(f"Final: {final_control} control texts and {final_depressed} depressed texts")
    print(f"Actual ratio: {final_control/final_depressed:.1f}:1")
    
    # 转换为numpy数组
    control_texts = np.array(control_texts)
    depressed_texts = np.array(depressed_texts)
    
    # 合并数据和标签
    X = np.concatenate([control_texts, depressed_texts], axis=0)
    y = np.concatenate([np.zeros(final_control), np.ones(final_depressed)], axis=0)
    
    return X, y

def detect_embedding_format(X):
    """
    检测embedding的格式类型
    """
    sample = X[0]
    
    if isinstance(sample, dict):
        return 'comprehensive'
    elif isinstance(sample, list) and len(sample) > 0:
        if isinstance(sample[0], np.ndarray):
            if len(sample[0].shape) == 1:
                return 'layer_wise'  # [29, d_model] - 每层一个向量
            elif len(sample[0].shape) == 2:
                return 'attention_heads'  # [29, num_heads, head_dim] - 注意力头
        return 'unknown_list'
    else:
        return 'unknown'

def train_layer_predictors(X, y, test_size=0.2, random_state=42):
    """
    为每一层训练预测器 - 适用于传统的层级embedding
    """
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training labels distribution: Control={np.sum(y_train==0)}, Depressed={np.sum(y_train==1)}")
    
    num_layers = X.shape[1]  # 29层
    results = []
    
    print("Training predictors for each layer...")
    for layer in tqdm(range(num_layers)):
        # 提取当前层的特征
        X_train_layer = X_train[:, layer, :]
        X_test_layer = X_test[:, layer, :]
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_layer)
        X_test_scaled = scaler.transform(X_test_layer)
        
        # 训练逻辑回归分类器
        classifier = LogisticRegression(
            max_iter=1000, 
            random_state=random_state,
            class_weight='balanced'
        )
        classifier.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = classifier.predict(X_test_scaled)
        y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'layer': layer,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_score': auc,
            'classifier': classifier,
            'scaler': scaler
        })
    
    return results, y_test, X_test

def train_attention_head_predictors(X, y, test_size=0.2, random_state=42):
    """
    为每个注意力头训练预测器
    X: List of [num_heads, head_dim] arrays
    """
    # 转换数据格式
    X_array = np.array(X)  # [num_samples, num_layers, num_heads, head_dim]
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    num_layers, num_heads = X_train.shape[1], X_train.shape[2]
    results = np.zeros((num_layers, num_heads, 3))  # [layers, heads, metrics]
    
    print("Training predictors for each attention head...")
    for layer in tqdm(range(num_layers)):
        for head in range(num_heads):
            # 提取当前层当前头的特征
            X_train_head = X_train[:, layer, head, :]
            X_test_head = X_test[:, layer, head, :]
            
            # 标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_head)
            X_test_scaled = scaler.transform(X_test_head)
            
            # 训练分类器
            classifier = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced'
            )
            classifier.fit(X_train_scaled, y_train)
            
            # 预测
            y_pred = classifier.predict(X_test_scaled)
            y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]
            
            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[layer, head, 0] = accuracy
            results[layer, head, 1] = f1
            results[layer, head, 2] = auc
    
    return results, y_test

def train_comprehensive_predictors(X, y, test_size=0.2, random_state=42):
    """
    为综合特征训练预测器
    """
    # 检查第一个样本的结构
    sample = X[0]
    if not isinstance(sample, dict):
        raise ValueError("Expected dict format for comprehensive embeddings")
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    all_results = {}
    
    # 处理每种特征类型
    for feature_type in sample.keys():
        print(f"Training predictors for {feature_type}...")
        
        # 提取特征
        train_features = [x[feature_type] for x in X_train]
        test_features = [x[feature_type] for x in X_test]
        
        if feature_type == 'hidden_states':
            # 按层训练
            train_array = np.array(train_features)
            test_array = np.array(test_features)
            results, _, _ = train_layer_predictors_from_arrays(
                train_array, test_array, y_train, y_test, random_state
            )
            all_results[feature_type] = results
        
        elif feature_type == 'attention_patterns':
            # 按层和头训练
            train_array = np.array(train_features)  # [samples, layers, heads]
            test_array = np.array(test_features)
            results = train_attention_patterns_from_arrays(
                train_array, test_array, y_train, y_test, random_state
            )
            all_results[feature_type] = results
    
    return all_results, y_test

def train_layer_predictors_from_arrays(X_train, X_test, y_train, y_test, random_state):
    """
    从数组训练层级预测器的辅助函数
    """
    num_layers = X_train.shape[1]
    results = []
    
    for layer in range(num_layers):
        X_train_layer = X_train[:, layer, :]
        X_test_layer = X_test[:, layer, :]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_layer)
        X_test_scaled = scaler.transform(X_test_layer)
        
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced'
        )
        classifier.fit(X_train_scaled, y_train)
        
        y_pred = classifier.predict(X_test_scaled)
        y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'layer': layer,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_score': auc
        })
    
    return results, y_test, X_test

def train_attention_patterns_from_arrays(X_train, X_test, y_train, y_test, random_state):
    """
    从注意力模式数组训练预测器
    """
    num_layers, num_heads = X_train.shape[1], X_train.shape[2]
    results = np.zeros((num_layers, num_heads, 3))
    
    for layer in range(num_layers):
        for head in range(num_heads):
            X_train_head = X_train[:, layer, head].reshape(-1, 1)
            X_test_head = X_test[:, layer, head].reshape(-1, 1)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_head)
            X_test_scaled = scaler.transform(X_test_head)
            
            classifier = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced'
            )
            classifier.fit(X_train_scaled, y_train)
            
            y_pred = classifier.predict(X_test_scaled)
            y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]
            
            results[layer, head, 0] = accuracy_score(y_test, y_pred)
            results[layer, head, 1] = f1_score(y_test, y_pred)
            results[layer, head, 2] = roc_auc_score(y_test, y_pred_proba)
    
    return results

def visualize_layer_results(results, save_path):
    """
    可视化层级结果 - 折线图
    """
    layers = [r['layer'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    auc_scores = [r['auc_score'] for r in results]
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Depression Prediction Performance Across LLM Layers (1:19 Ratio)', 
                 fontsize=16, fontweight='bold')
    
    # 准确率
    axes[0, 0].plot(layers, accuracies, 'b-o', linewidth=2, markersize=5)
    axes[0, 0].set_title('Accuracy by Layer')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([min(accuracies) - 0.02, max(accuracies) + 0.02])
    
    # F1分数
    axes[0, 1].plot(layers, f1_scores, 'g-o', linewidth=2, markersize=5)
    axes[0, 1].set_title('F1 Score by Layer')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([min(f1_scores) - 0.02, max(f1_scores) + 0.02])
    
    # AUC分数
    axes[1, 0].plot(layers, auc_scores, 'r-o', linewidth=2, markersize=5)
    axes[1, 0].set_title('AUC Score by Layer')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('AUC Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([min(auc_scores) - 0.02, max(auc_scores) + 0.02])
    
    # 综合对比
    axes[1, 1].plot(layers, accuracies, 'b-o', label='Accuracy', linewidth=2, markersize=4)
    axes[1, 1].plot(layers, f1_scores, 'g-o', label='F1 Score', linewidth=2, markersize=4)
    axes[1, 1].plot(layers, auc_scores, 'r-o', label='AUC Score', linewidth=2, markersize=4)
    axes[1, 1].set_title('All Metrics Comparison')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印最佳结果
    best_accuracy_layer = max(results, key=lambda x: x['accuracy'])
    best_f1_layer = max(results, key=lambda x: x['f1_score'])
    best_auc_layer = max(results, key=lambda x: x['auc_score'])
    
    print("\n" + "="*50)
    print("BEST PERFORMING LAYERS:")
    print("="*50)
    print(f"Best Accuracy: Layer {best_accuracy_layer['layer']} ({best_accuracy_layer['accuracy']:.4f})")
    print(f"Best F1 Score: Layer {best_f1_layer['layer']} ({best_f1_layer['f1_score']:.4f})")
    print(f"Best AUC Score: Layer {best_auc_layer['layer']} ({best_auc_layer['auc_score']:.4f})")

def visualize_attention_heatmap(results, save_path):
    """
    可视化注意力头结果 - 热力图
    results: [num_layers, num_heads, 3] array
    """
    metrics = ['Accuracy', 'F1 Score', 'AUC Score']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Attention Head Performance Heatmaps (1:1 Ratio)', 
                 fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        # 创建热力图
        im = axes[i].imshow(results[:, :, i], cmap='viridis', aspect='auto')
        
        # 设置标题和标签
        axes[i].set_title(f'{metric} by Layer and Head')
        axes[i].set_xlabel('Attention Head')
        axes[i].set_ylabel('Layer')
        
        # 设置刻度
        axes[i].set_xticks(range(0, results.shape[1], 4))
        axes[i].set_yticks(range(0, results.shape[0], 4))
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[i])
        cbar.set_label(metric)
        
        # 添加数值标注（可选，对于大矩阵可能太密集）
        if results.shape[0] <= 10 and results.shape[1] <= 10:
            for layer in range(results.shape[0]):
                for head in range(results.shape[1]):
                    text = axes[i].text(head, layer, f'{results[layer, head, i]:.3f}',
                                      ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 修复最佳注意力头的查找逻辑
    print("\n" + "="*50)
    print("BEST PERFORMING ATTENTION HEADS:")
    print("="*50)
    
    for i, metric in enumerate(metrics):
        # 对每个指标单独找最佳位置
        metric_results = results[:, :, i]  # 取出当前指标的2D数组
        best_flat_index = np.argmax(metric_results)  # 找到最大值的扁平化索引
        best_layer, best_head = np.unravel_index(best_flat_index, metric_results.shape)  # 转换为2D坐标
        score = results[best_layer, best_head, i]
        print(f"Best {metric}: Layer {best_layer}, Head {best_head} ({score:.4f})")

def main():
    import sys
    
    # 解析命令行参数
    embedding_type = 'direct'  # 默认值
    if len(sys.argv) > 1:
        if sys.argv[1] == '--attention-heads':
            embedding_type = 'attention_heads'
        elif sys.argv[1] == '--residual':
            embedding_type = 'residual'
        elif sys.argv[1] == '--mlp':
            embedding_type = 'mlp'
        elif sys.argv[1] == '--comprehensive':
            embedding_type = 'comprehensive'
        elif sys.argv[1] == '--direct':
            embedding_type = 'direct'
        elif sys.argv[1] == '--psych-encoder':
            embedding_type = 'psych_encoder'

    
    # 构建文件路径
    suffix_map = {
        'direct': 'direct',
        'attention_heads': 'attn_heads',
        'residual': 'residual',
        'mlp': 'mlp',
        'comprehensive': 'comprehensive',
        'psych_encoder': 'psych_encoder'
    }
    
    suffix = suffix_map[embedding_type]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    control_path = os.path.join(base_dir, f"data/preprocessed/control_text_emb_{suffix}_100.pkl")
    depressed_path = os.path.join(base_dir, f"data/preprocessed/depressed_text_emb_{suffix}_100.pkl")
    
    print(f"Loading {embedding_type} embeddings...")
    print(f"Control path: {control_path}")
    print(f"Depressed path: {depressed_path}")
    
    # 检查文件是否存在
    if not os.path.exists(control_path) or not os.path.exists(depressed_path):
        print(f"Error: Embedding files not found. Please run psych_encoder.py first with appropriate flags.")
        return
    
    # 加载和准备数据（1:19比例）
    X, y = load_and_prepare_data(control_path, depressed_path, ratio=1)
    print(f"Final data shape: {np.array(X).shape if not isinstance(X[0], dict) else 'Complex structure'}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: Control={np.sum(y==0)}, Depressed={np.sum(y==1)}")
    
    # 检测embedding格式
    # format_type = detect_embedding_format(X)
    format_type = 'layer_wise'
    # format_type = 'attention_heads'
    print(f"Detected embedding format: {format_type}")
    
    # 根据格式选择训练和可视化方法
    save_dir = os.path.join(base_dir, f"results")
    os.makedirs(save_dir, exist_ok=True)
    
    if format_type == 'layer_wise':
        print("\nTraining layer predictors...")
        X_array = np.array(X)
        results, y_test, X_test = train_layer_predictors(X_array, y)
        
        # 可视化和保存
        save_path = f'{save_dir}/layer_analysis_{suffix}_ratio19.png'
        visualize_layer_results(results, save_path)
        
        # 保存结果
        results_to_save = [{k: v for k, v in result.items() if k not in ['classifier', 'scaler']} 
                          for result in results]
        with open(f'{save_dir}/layer_analysis_{suffix}_ratio19.pkl', 'wb') as f:
            pickle.dump(results_to_save, f)
    
    elif format_type == 'attention_heads':
        print("\nTraining attention head predictors...")
        results, y_test = train_attention_head_predictors(X, y)
        
        # 可视化和保存
        save_path = f'{save_dir}/attention_heatmap_{suffix}_ratio19.png'
        visualize_attention_heatmap(results, save_path)
        
        # 保存结果
        with open(f'{save_dir}/attention_analysis_{suffix}_ratio19.pkl', 'wb') as f:
            pickle.dump(results, f)
    
    elif format_type == 'comprehensive':
        print("\nTraining comprehensive predictors...")
        all_results, y_test = train_comprehensive_predictors(X, y)
        
        # 为每种特征类型生成可视化
        for feature_type, results in all_results.items():
            if isinstance(results, list):  # layer-wise results
                save_path = f'{save_dir}/layer_analysis_{feature_type}_ratio19.png'
                visualize_layer_results(results, save_path)
            elif isinstance(results, np.ndarray):  # attention head results
                save_path = f'{save_dir}/attention_heatmap_{feature_type}_ratio19.png'
                visualize_attention_heatmap(results, save_path)
        
        # 保存所有结果
        with open(f'{save_dir}/comprehensive_analysis_{suffix}_ratio19.pkl', 'wb') as f:
            pickle.dump(all_results, f)
    
    else:
        print(f"Unsupported embedding format: {format_type}")
        return
    
    print(f"\nAnalysis complete! Results saved to {save_dir}")

if __name__ == "__main__":
    main()