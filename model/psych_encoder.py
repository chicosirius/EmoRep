import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import json
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class PsychologicalEncoder(nn.Module):
    def __init__(self, lexicon_path, embedding_model):
        super().__init__()

        # 初始化 Qwen2.5 的 tokenizer 和模型（默认值为 Qwen2.5）
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(embedding_model, trust_remote_code=True, device_map="auto")
        self.model.eval()

        # 冻结 Qwen 参数（不训练）
        for p in self.model.parameters():
            p.requires_grad = False

        # 读取心理词汇表（K 个词）
        with open(lexicon_path, "r") as f:
            self.lexicon = json.load(f)
        self.words = [w for group in self.lexicon.values() for w in group]
        self.K = len(self.words)  # K = 100

        # 获取模型的词嵌入层
        self.embedding_layer = self.model.get_input_embeddings()  # shape: [vocab_size, d_qwen]
        self.embedding_dim = self.embedding_layer.embedding_dim  # d_qwen = 3584

        # 使用 tokenizer 获取每个词的 token_id，然后查嵌入
        word_ids = self.tokenizer(self.words, add_special_tokens=False)["input_ids"]
        max_len = max(len(ids) for ids in word_ids)
        input_ids = torch.zeros((self.K, max_len), dtype=torch.long)

        for i, ids in enumerate(word_ids):
            input_ids[i, :len(ids)] = torch.tensor(ids)

        # 获取词的 embedding（取平均 over all subtokens）
        with torch.no_grad():
            word_embeddings = []
            for ids in word_ids:
                ids_tensor = torch.tensor(ids).to(self.embedding_layer.weight.device)
                emb = self.embedding_layer(ids_tensor)
                emb = emb.mean(dim=0)  # 平均所有 sub-tokens, shape: [d_qwen]
                word_embeddings.append(emb)

            # 拼接成 [K, d_qwen]
            self.word_embeddings = F.normalize(torch.stack(word_embeddings), dim=-1)  # [K, d]

        self.output_dim = self.embedding_dim  # 输出维度为 d_qwen = 3584

    def forward(self, token_embeddings):
        """
        token_embeddings: Tensor of shape [n, d_qwen]
        return: psychological vector of shape [1, d_qwen]
        """
        # Step 1: Normalize input token embeddings
        token_embeddings = F.normalize(token_embeddings, dim=-1)  # [n, d]

        # Step 2: Compute attention scores between lexicon words and tokens
        attn_scores = torch.matmul(self.word_embeddings.to(token_embeddings.device), token_embeddings.T)  # [K, n]

        # Step 3: Aggregate max attention per word (dim=1: over tokens), then softmax across words
        alpha = F.softmax(attn_scores.max(dim=1).values, dim=0)  # [K]

        # Step 4: Weighted sum of word embeddings
        z_t = torch.sum(alpha.unsqueeze(1) * self.word_embeddings.to(token_embeddings.device), dim=0)  # [d]

        return z_t.unsqueeze(0)  # [1, d_qwen] = [1, 3584]

# 0. 直接使用LLM原始embedding再经过PsychologicalEncoder
class PsychLLMEmbedder(nn.Module):
    def __init__(self, model_name='Qwen/Qwen2.5-7B-Instruct'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        self.model.eval()

    def forward(self, text):
        """
        给定文本，返回每一层的平均token embedding
        text: 输入文本字符串
        return: List[Tensor], 每个元素形状为 [d_qwen=3584]，表示该层所有token的平均embedding
        """
        inputs = self.tokenizer(text, return_tensors='pt')
        
        # 将输入移到模型设备上
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        layer_embeddings = []
        for hidden_state in outputs.hidden_states:
            token_embeddings = hidden_state.squeeze(0)  # [seq_len, d_qwen]
            avg_embedding = token_embeddings.mean(dim=0)  # [d_qwen]
            layer_embeddings.append(avg_embedding)
        
        return layer_embeddings  # List of 29 tensors, each [3584]

# 1. 传统的层级embedding提取器
class LLMEmbedder(nn.Module):
    def __init__(self, model_name='Qwen/Qwen2.5-7B-Instruct'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        self.model.eval()

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states  # 返回所有层的输出


# 2. 直接使用LLM原始embedding的类
class DirectLLMEmbedder(nn.Module):
    def __init__(self, model_name='Qwen/Qwen2.5-7B-Instruct'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        self.model.eval()

    def forward(self, text):
        """
        给定文本，返回每一层的平均token embedding
        text: 输入文本字符串
        return: List[Tensor], 每个元素形状为 [d_qwen=3584]，表示该层所有token的平均embedding
        """
        inputs = self.tokenizer(text, return_tensors='pt')
        
        # 将输入移到模型设备上
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        layer_embeddings = []
        for hidden_state in outputs.hidden_states:
            token_embeddings = hidden_state.squeeze(0)  # [seq_len, d_qwen]
            avg_embedding = token_embeddings.mean(dim=0)  # [d_qwen]
            layer_embeddings.append(avg_embedding)
        
        return layer_embeddings  # List of 29 tensors, each [3584]


# 3. 新增：注意力头输出提取器
class AttentionHeadEmbedder(nn.Module):
    def __init__(self, model_name='Qwen/Qwen2.5-7B-Instruct'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        self.model.eval()
        
        # 获取模型配置
        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers  # 28 for Qwen2.5-7B
        self.num_heads = self.config.num_attention_heads  # 28 for Qwen2.5-7B
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads  # 3584 // 28 = 128

    def forward(self, text):
        """
        提取每个注意力头的输出
        return: List[Tensor], 每个元素形状为 [num_heads, head_dim]，表示该层每个头的平均输出
        """
        inputs = self.tokenizer(text, return_tensors='pt')
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        attention_outputs = []
        
        def attention_hook(module, input, output):
            # output[0] 是注意力输出 [batch_size, seq_len, hidden_size]
            # 需要reshape到 [batch_size, seq_len, num_heads, head_dim]
            attn_output = output[0]  # [1, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = attn_output.shape
            
            # Reshape to separate heads
            head_output = attn_output.view(batch_size, seq_len, self.num_heads, self.head_dim)
            # Average over sequence length
            avg_head_output = head_output.mean(dim=1)  # [1, num_heads, head_dim]
            attention_outputs.append(avg_head_output.squeeze(0))  # [num_heads, head_dim]
        
        # 注册hook到每个attention层
        hooks = []
        for layer in self.model.layers:
            hook = layer.self_attn.register_forward_hook(attention_hook)
            hooks.append(hook)
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        return attention_outputs  # List of 28 tensors, each [num_heads=28, head_dim=128]


# 4. 修复：Residual Stream激活提取器
class ResidualStreamEmbedder(nn.Module):
    def __init__(self, model_name='Qwen/Qwen2.5-7B-Instruct'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        self.model.eval()

    def forward(self, text):
        """
        提取residual stream在每层的激活
        return: dict包含不同类型的激活
        """
        inputs = self.tokenizer(text, return_tensors='pt')
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        activations = {
            'pre_attention': [],    # 注意力前的residual stream
            'post_attention': [],   # 注意力后的residual stream  
            'pre_mlp': [],         # MLP前的residual stream
            'post_mlp': []         # MLP后的residual stream (= 层输出)
        }
        
        # 修复：添加错误检查的pre_hook
        def pre_attention_hook(module, input):
            try:
                # 检查input是否为空或格式不对
                if input and len(input) > 0 and hasattr(input[0], 'mean'):
                    activation = input[0].mean(dim=1)  # 平均序列长度 [1, hidden_size]
                    activations['pre_attention'].append(activation.squeeze(0).cpu())
                else:
                    print(f"Warning: Invalid input in pre_attention_hook: {type(input)}")
            except Exception as e:
                print(f"Error in pre_attention_hook: {e}")
        
        # 修复：添加错误检查的post_hook
        def post_attention_hook(module, input, output):
            try:
                # 检查output是否为空或格式不对
                if output and len(output) > 0 and hasattr(output[0], 'mean'):
                    activation = output[0].mean(dim=1)  # [1, hidden_size]
                    activations['post_attention'].append(activation.squeeze(0).cpu())
                elif hasattr(output, 'mean'):  # output可能不是tuple而是直接的tensor
                    activation = output.mean(dim=1)
                    activations['post_attention'].append(activation.squeeze(0).cpu())
                else:
                    print(f"Warning: Invalid output in post_attention_hook: {type(output)}")
            except Exception as e:
                print(f"Error in post_attention_hook: {e}")
        
        # 修复：添加错误检查的pre_mlp_hook
        def pre_mlp_hook(module, input):
            try:
                if input and len(input) > 0 and hasattr(input[0], 'mean'):
                    activation = input[0].mean(dim=1)  # [1, hidden_size]
                    activations['pre_mlp'].append(activation.squeeze(0).cpu())
                else:
                    print(f"Warning: Invalid input in pre_mlp_hook: {type(input)}")
            except Exception as e:
                print(f"Error in pre_mlp_hook: {e}")
        
        # 修复：添加错误检查的post_mlp_hook
        def post_mlp_hook(module, input, output):
            try:
                if hasattr(output, 'mean'):
                    activation = output.mean(dim=1)  # [1, hidden_size]
                    activations['post_mlp'].append(activation.squeeze(0).cpu())
                else:
                    print(f"Warning: Invalid output in post_mlp_hook: {type(output)}")
            except Exception as e:
                print(f"Error in post_mlp_hook: {e}")
        
        # 注册hooks时添加错误检查
        hooks = []
        try:
            for i, layer in enumerate(self.model.layers):
                try:
                    # 检查layer是否有expected的组件
                    if hasattr(layer, 'self_attn'):
                        hooks.append(layer.self_attn.register_forward_pre_hook(pre_attention_hook))
                        hooks.append(layer.self_attn.register_forward_hook(post_attention_hook))
                    
                    if hasattr(layer, 'mlp'):
                        hooks.append(layer.mlp.register_forward_pre_hook(pre_mlp_hook))
                        hooks.append(layer.mlp.register_forward_hook(post_mlp_hook))
                        
                except Exception as e:
                    print(f"Error registering hooks for layer {i}: {e}")
                    continue
            
            print(f"Successfully registered {len(hooks)} hooks")
            
            # 执行前向传播
            with torch.no_grad():
                _ = self.model(**inputs)
                
        except Exception as e:
            print(f"Error during forward pass: {e}")
        finally:
            # 确保清理hooks
            for hook in hooks:
                try:
                    hook.remove()
                except:
                    pass
        
        # 检查收集到的数据
        print(f"Collected activations:")
        for key, values in activations.items():
            print(f"  {key}: {len(values)} items")
        
        return activations

# 5. 新增：MLPs激活提取器
class MLPActivationEmbedder(nn.Module):
    def __init__(self, model_name='Qwen/Qwen2.5-7B-Instruct'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        self.model.eval()

    def forward(self, text):
        """
        提取每层MLP的中间激活
        return: dict包含不同MLP组件的激活
        """
        inputs = self.tokenizer(text, return_tensors='pt')
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        mlp_activations = {
            'gate_proj': [],     # Gate projection激活
            'up_proj': [],       # Up projection激活  
            'down_proj': [],     # Down projection激活
            'intermediate': []   # 中间激活 (gate * up)
        }
        
        def gate_hook(module, input, output):
            activation = output.mean(dim=1)  # [1, intermediate_size]
            mlp_activations['gate_proj'].append(activation.squeeze(0))
        
        def up_hook(module, input, output):
            activation = output.mean(dim=1)  # [1, intermediate_size]
            mlp_activations['up_proj'].append(activation.squeeze(0))
        
        def down_hook(module, input, output):
            activation = output.mean(dim=1)  # [1, hidden_size]
            mlp_activations['down_proj'].append(activation.squeeze(0))
        
        # 注册hooks到MLP组件
        hooks = []
        for layer in self.model.layers:
            if hasattr(layer.mlp, 'gate_proj'):
                hooks.append(layer.mlp.gate_proj.register_forward_hook(gate_hook))
            if hasattr(layer.mlp, 'up_proj'):
                hooks.append(layer.mlp.up_proj.register_forward_hook(up_hook))
            if hasattr(layer.mlp, 'down_proj'):
                hooks.append(layer.mlp.down_proj.register_forward_hook(down_hook))
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        return mlp_activations


# 6. 新增：注意力权重提取器
class AttentionWeightEmbedder(nn.Module):
    def __init__(self, model_name='Qwen/Qwen2.5-7B-Instruct'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        self.model.eval()

    def forward(self, text):
        """
        提取注意力权重模式
        return: List[Tensor], 每个元素形状为 [num_heads, seq_len, seq_len]
        """
        inputs = self.tokenizer(text, return_tensors='pt')
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # outputs.attentions 是 tuple of attention weights
        # 每个元素形状: [batch_size, num_heads, seq_len, seq_len]
        attention_patterns = []
        for attn_weights in outputs.attentions:
            # 计算每个头的注意力模式特征
            attn_squeezed = attn_weights.squeeze(0)  # [num_heads, seq_len, seq_len]
            
            # 可以提取不同的注意力模式特征:
            # 1. 每个头的平均注意力熵
            # 2. 注意力的最大值位置
            # 3. 自注意力vs交叉注意力的比例等
            
            attention_patterns.append(attn_squeezed)
        
        return attention_patterns


# 主要的提取函数，支持多种方法
def extract_comprehensive_embeddings(extraction_method='direct'):
    """
    使用不同方法提取embeddings
    extraction_method: str, 选择提取方法
        - 'direct': 直接LLM embeddings
        - 'attention_heads': 注意力头输出
        - 'residual_stream': Residual stream激活
        - 'mlp_activations': MLP激活
        - 'attention_weights': 注意力权重模式
        - 'comprehensive': 综合提取多种特征
    """
    MODEL_PATH = '/data/home/xixian_yong/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
    
    # 根据方法选择不同的embedder
    if extraction_method == 'direct':
        embedder = DirectLLMEmbedder(model_name=MODEL_PATH)
        suffix = 'direct'
    elif extraction_method == 'attention_heads':
        embedder = AttentionHeadEmbedder(model_name=MODEL_PATH)
        suffix = 'attn_heads'
    elif extraction_method == 'residual_stream':
        embedder = ResidualStreamEmbedder(model_name=MODEL_PATH)
        suffix = 'residual'
    elif extraction_method == 'mlp_activations':
        embedder = MLPActivationEmbedder(model_name=MODEL_PATH)
        suffix = 'mlp'
    elif extraction_method == 'attention_weights':
        embedder = AttentionWeightEmbedder(model_name=MODEL_PATH)
        suffix = 'attn_weights'
    else:
        raise ValueError(f"Unknown extraction method: {extraction_method}")
    
    # 文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    CONTROL_FILE_PATH = os.path.join(base_dir, "data/SWDD/control_cleaned_0804.jsonl")
    CONTROL_SAVE_PATH = os.path.join(base_dir, f"data/preprocessed/control_text_emb_{suffix}.pkl")

    DEPRESSED_FILE_PATH = os.path.join(base_dir, "data/SWDD/depressed_cleaned_0804.jsonl")
    DEPRESSED_SAVE_PATH = os.path.join(base_dir, f"data/preprocessed/depressed_text_emb_{suffix}.pkl")
    
    def process_file(file_path, save_path, num_users=100):
        all_embeddings = []
        
        with open(file_path, "r", encoding="utf8") as f:
            sampled_users = random.sample(list(f), num_users)
            
            for item in tqdm(sampled_users, desc=f"Processing {extraction_method}"):
                item = json.loads(item)
                user_texts = []
                
                for tweet in item['tweets']:
                    if tweet['raw_text'].strip() == '':
                        continue
                    user_texts.append(tweet['raw_text'])

                if user_texts:
                    user_embeddings = []
                    for text in user_texts:  # opt: 限制每个用户的文本数量
                        try:
                            emb = embedder(text)
                            # 转换为numpy并处理不同的输出格式
                            if isinstance(emb, dict):
                                # comprehensive或其他复杂输出
                                emb_processed = {}
                                for key, value in emb.items():
                                    if isinstance(value, list):
                                        emb_processed[key] = [v.cpu().numpy() if torch.is_tensor(v) else v for v in value]
                                    else:
                                        emb_processed[key] = value.cpu().numpy() if torch.is_tensor(value) else value
                                user_embeddings.append(emb_processed)
                            elif isinstance(emb, list):
                                # 列表输出 (如attention_heads, hidden_states等)
                                user_embeddings.append([e.cpu().numpy() if torch.is_tensor(e) else e for e in emb])
                            else:
                                # 单个tensor
                                user_embeddings.append(emb.cpu().numpy() if torch.is_tensor(emb) else emb)
                        except Exception as e:
                            print(f"Error processing text: {e}")
                            continue
                    
                    if user_embeddings:
                        all_embeddings.append(user_embeddings)
        
        # 保存结果
        with open(save_path, 'wb') as f:
            pickle.dump(all_embeddings, f)
        
        print(f"Saved {extraction_method} embeddings to: {save_path}")
    
    # 处理两个数据集
    print(f"Processing control group with {extraction_method}...")
    process_file(CONTROL_FILE_PATH, CONTROL_SAVE_PATH)
    
    print(f"Processing depressed group with {extraction_method}...")
    process_file(DEPRESSED_FILE_PATH, DEPRESSED_SAVE_PATH)


# 原有函数保持不变
def extract_direct_embeddings():
    """使用DirectLLMEmbedder提取原始LLM embeddings"""
    extract_comprehensive_embeddings('direct')

def extract_psychological_embeddings():
    """使用PsychologicalEncoder提取心理学导向的embeddings"""
    # ... 保持原有代码不变


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        method = sys.argv[1]
        if method == "--direct":
            print("Extracting direct LLM embeddings...")
            extract_comprehensive_embeddings('direct')
        elif method == "--attention-heads":
            print("Extracting attention head embeddings...")
            extract_comprehensive_embeddings('attention_heads')
        elif method == "--residual":
            print("Extracting residual stream embeddings...")
            extract_comprehensive_embeddings('residual_stream')
        elif method == "--mlp":
            print("Extracting MLP activation embeddings...")
            extract_comprehensive_embeddings('mlp_activations')
        elif method == "--attention-weights":
            print("Extracting attention weight patterns...")
            extract_comprehensive_embeddings('attention_weights')
        elif method == "--psychological":
            print("Extracting psychological embeddings...")
            extract_psychological_embeddings()
        else:
            print(f"Unknown method: {method}")
    else:
        print("Available extraction methods:")
        print("  --direct: Direct LLM hidden states")
        print("  --attention-heads: Attention head outputs")
        print("  --residual: Residual stream activations")
        print("  --mlp: MLP intermediate activations")
        print("  --attention-weights: Attention weight patterns")
        print("  --psychological: Psychological encoder (original)")