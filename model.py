from transformers import BertModel,BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

time_gap=10000.0

class CSIBERT(nn.Module):
    def __init__(self,bertconfig, input_dim):
        super().__init__()
        self.bertconfig=bertconfig
        self.bert=BertModel(bertconfig)
        self.hidden_dim=bertconfig.hidden_size
        self.input_dim=input_dim
        self.len=bertconfig.max_position_embeddings

        # self.query = nn.Linear(self.len, self.len, bias=False)
        # self.key = nn.Linear(self.len, self.len, bias=False)
        # self.value = nn.Linear(self.len, self.len, bias=False)
        # self.head_num = 4
        # self.head_dim = self.len//self.head_num
        # self.softmax = nn.Softmax(dim=-1)
        # # self.norm = nn.LayerNorm(self.len)
        # self.norm = nn.LayerNorm(self.input_dim)

        # self.norm1 = nn.LayerNorm(self.hidden_dim*2)
        # self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.Norm1 = nn.LayerNorm(self.input_dim)
        self.Norm2 = nn.LayerNorm(self.hidden_dim)
        self.Norm3 = nn.LayerNorm(self.hidden_dim)

        self.csi_emb=nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.time_emb=nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # self.time_emb = nn.Sequential(
        #     nn.Linear(1, self.hidden_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim//2, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim)
        # )

        self.fusion_emb=nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim*2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # self.sigmoid=nn.Sigmoid()

        # self.arl = nn.Sequential(
        #     nn.Linear(self.len, self.len//2),
        #     nn.ReLU(),
        #     nn.Linear(self.len//2, self.len//2),
        #     nn.ReLU(),
        #     nn.Linear(self.len//2, self.len)
        #

        self.arl = nn.Sequential(
            nn.Linear(self.len, self.len // 2),
            nn.ReLU(),
            nn.Linear(self.len // 2, self.len // 4),
            nn.ReLU(),
            nn.Linear(self.len // 4, 1)
        )

    def forward(self,x,timestamp,attention_mask=None):
        x=x.to(torch.float32)

        x=self.attention(x)
        # x=self.Norm1(x)
        x=self.csi_emb(x)
        # x=self.Norm2(x)
        # x_ori=x.clone()


        x_time=self.time_embedding(timestamp)

        # x=torch.concat([x,x_time],dim=-1)
        # # x=self.norm1(x)
        # x=self.fusion_emb(x)
        # # x=self.norm2(x)

        x = x + x_time
        # x = self.Norm3(x)


        # x = x + x_ori

        y=self.bert(inputs_embeds=x, attention_mask=attention_mask, output_hidden_states=False)
        # y=y.hidden_states[-1]
        y=y.last_hidden_state
        return y

    def time_embedding(self,timestamp,t=1):
        device=timestamp.device

        # timestamp = (timestamp - timestamp[:,0:1]) / time_gap
        # timestamp = (timestamp - timestamp[:, 0:1]) / (timestamp[:,-1:] - timestamp[:, 0:1])
        timestamp = (timestamp - timestamp[:, 0:1]) / (timestamp[:,-1:] - timestamp[:, 0:1]) * self.len

        timestamp**=t
        d_model=self.input_dim
        dim=torch.tensor(list(range(d_model))).to(device)
        batch_size,length=timestamp.shape
        timestamp=timestamp.unsqueeze(2).repeat(1, 1, d_model)
        dim=dim.reshape([1,1,-1]).repeat(batch_size,length,1)
        sin_emb = torch.sin(timestamp/10000**(dim//2*2/d_model))
        cos_emb = torch.cos(timestamp/10000**(dim//2*2/d_model))
        mask=torch.zeros(d_model).to(device)
        mask[::2]=1
        emb=sin_emb*mask+cos_emb*(1-mask)
        emb=self.time_emb(emb)

        # timestamp = torch.unsqueeze(timestamp, -1)
        # emb=self.time_emb(timestamp)

        return emb

    # def attention(self,x):
    #     y = torch.transpose(x, -1, -2)
    #     batch_size = y.shape[0]
    #     queries = self.query(y).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
    #     keys = self.key(y).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
    #     values = self.value(y).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2)
    #     attention_weights = self.softmax(torch.matmul(queries, keys.transpose(-1, -2))/ (self.head_dim ** 0.5))
    #
    #     # attended_values = torch.matmul(attention_weights,values).transpose(1, 2)
    #     # attended_values = attended_values.reshape(batch_size,self.input_dim,self.len)
    #     # attended_values = self.norm(attended_values)
    #     # y = attended_values.transpose(1, 2)
    #
    #     attended_values = torch.matmul(attention_weights, values).transpose(-1, -2)
    #     attended_values = attended_values.reshape(batch_size, self.len, self.input_dim)
    #     y = self.norm(attended_values)
    #
    #     return y+x

    def attention(self, x):
        y = torch.transpose(x, -1, -2)
        attn = self.arl(y)
        y = y * attn
        y = torch.transpose(y, -1, -2)
        return y

class Token_Classifier(nn.Module):
    def __init__(self,bert,class_num=52):
        super().__init__()
        self.bert=bert
        self.classifier=nn.Sequential(
            nn.Linear(bert.hidden_dim, bert.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(bert.hidden_dim//2, class_num)
        )

    def forward(self,x,timestamp,attention_mask=None):
        x=self.bert(x,timestamp,attention_mask=attention_mask)
        x=self.classifier(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat


class Sequence_Classifier(nn.Module):
    def __init__(self, csibert, class_num, hs=128, da=128, r=4):
        super().__init__()
        self.bert = csibert
        self.attention = SelfAttention(hs, da, r)
        self.classifier = nn.Sequential(
            nn.Linear(hs * r, hs * r // 2),
            nn.ReLU(),
            nn.Linear(hs * r // 2, class_num)
        )

    def forward(self, x, timestamp,attention_mask=None):
        x = self.bert(x, timestamp,attention_mask=attention_mask)
        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.classifier(flatten)
        return res

#test
if __name__ == "__main__":
    configuration = BertConfig(max_position_embeddings=100, hidden_size=128, num_hidden_layers=6,num_attention_heads=8)
    csibert=CSIBERT(configuration,52)
    token_classifier=Token_Classifier(csibert,class_num=52)
    sequence_classifier=Sequence_Classifier(csibert,class_num=6, hs=128, da=128, r=8)
    x=torch.randn([3,100,52])
    time=torch.randn([3,100])
    y1=token_classifier(x,time)
    y2=sequence_classifier(x,time)
    print(y1.shape)
    print(y2.shape)
