import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class DNABERT2Encoder(nn.Module):
    def __init__(self, model_path="DNABERT-2-117M", freeze_bert=False, pooling='mean'):
        super(DNABERT2Encoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 禁用Flash Attention，使用标准attention
        self.bert_model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True,
            # attn_implementation="eager"
        )
        
        self.pooling = pooling
        
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
    
    def forward(self, sequences):
        """
        Args:
            sequences: list of DNA sequence strings
        Returns:
            embeddings: [batch_size, 768]
        """
        device = next(self.bert_model.parameters()).device
        inputs = self.tokenizer(sequences, return_tensors='pt', 
                               padding=True, truncation=True, 
                               max_length=512).to(device)
        
        outputs = self.bert_model(**inputs)
        
        # DNABERT-2返回tuple: (hidden_states, ...)
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]  # 第一个元素是hidden_states
        else:
            hidden_states = outputs.last_hidden_state
        
        # hidden_states shape: [batch, seq_len, 768]
        
        if self.pooling == 'mean':
            embeddings = torch.mean(hidden_states, dim=1)
        elif self.pooling == 'max':
            embeddings = torch.max(hidden_states, dim=1)[0]
        elif self.pooling == 'cls':
            embeddings = hidden_states[:, 0, :]
        
        return embeddings