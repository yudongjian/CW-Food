import torch
import torch.nn as nn

class Image_adapter(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mask = nn.Parameter(torch.zeros(hidden_size))
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        out_feature = self.adapter( self.sigmoid(self.mask)*feature ) + self.sigmoid(self.mask)*feature

        return out_feature

class ydj_image_adapter(nn.Module):
    def __init__(self):
        super(ydj_image_adapter, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4)
        self.transformers_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.final_linear = nn.Linear(1024, 768)

    def forward(self, input):
        output = self.transformers_encoder(input)
        return self.final_linear(output)

class Common_image_adapter(nn.Module):
    def __init__(self):
        super(Common_image_adapter, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4)
        self.transformers_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.final_linear = nn.Linear(1024, 768)

    def forward(self, input):
        output = self.transformers_encoder(input)
        return self.final_linear(output)


class Private_image_adapter(nn.Module):
    def __init__(self):
        super(Private_image_adapter, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4)
        self.transformers_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.final_linear = nn.Linear(1024, 768)

    def forward(self, input):
        output = self.transformers_encoder(input)
        return self.final_linear(output)


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=4)
        self.transformers_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

    def forward(self, word_emb, image_emb):
        input = word_emb + image_emb
        output = self.transformers_encoder(input)
        return output


class FusionNet_V2(nn.Module):
    def __init__(self):
        super(FusionNet_V2, self).__init__()
        hidden_size = 768

        self.image_liner = nn.Linear(1024, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=4)
        self.transformers_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def transf(self, input):
        return self.image_liner(input)

    def forward(self, word_emb, image_emb):
        input = word_emb + image_emb
        output = self.transformers_encoder(input)
        return self.adapter(output)

class FusionNet_V3(nn.Module):
    def __init__(self):
        super(FusionNet_V3, self).__init__()
        hidden_size = 768

        self.image_liner = nn.Linear(1024, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=4)
        self.transformers_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )


        self.mask = nn.Parameter(torch.zeros(768))
        self.sigmoid = nn.Sigmoid()
        self.final_linear = nn.Linear(1024, hidden_size)

        self.scale1 = nn.Parameter(torch.tensor(1.0))  # 初始缩放因子为1.0
        self.scale2 = nn.Parameter(torch.tensor(1.0))  # 初始缩放因子为1.0

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=4, dropout=0.2)
        self.transformers_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=768, nhead=4)
        self.transformers_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=3)

        self.word_linear = nn.Linear(hidden_size, hidden_size)

    # 0.2
    def transf(self, word_emb, image_emb):  # image_emb = [batch_size, 1, 1024]
        image_emb = self.final_linear(image_emb)
        # image_emb = self.adapter(self.sigmoid(self.mask) * image_emb) + self.sigmoid(self.mask) * image_emb
        # word_emb = self.transformers_encoder2(word_emb)
        image_emb = self.transformers_encoder(image_emb)
        # word_emb = self.word_linear(word_emb)
        return word_emb, image_emb

    def scale(self, word_emb, image_emb):
        self.scale_word_embedding = self.scale1 * word_emb
        self.scale_image_embedding = self.scale2 * image_emb


    def fusion_loss(self):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        a = self.scale_word_embedding.mean(dim=1)
        b = self.scale_image_embedding.squeeze(0)
        sim = cos(a, b).mean()
        return sim


    def forward(self, word_emb, image_emb):
        self.scale(word_emb, image_emb)
        all_embedding = self.scale_word_embedding + self.scale_image_embedding
        # all_embedding = self.transformers_encoder(all_embedding)
        return all_embedding


class Self_Image_adapter(nn.Module):
    def __init__(self, hidden_size=1024, output=768):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mask = nn.Parameter(torch.zeros(hidden_size))
        self.sigmoid = nn.Sigmoid()
        self.final_linear = nn.Linear(hidden_size, output)
    def forward(self, feature):
        out_feature = self.adapter( self.sigmoid(self.mask)*feature ) + self.sigmoid(self.mask)*feature

        return self.final_linear(out_feature)

def cal_cos(text, img, cos):
    a = text.mean(dim=1)
    b = img.squeeze(0)
    sim = cos(a, b).mean()
    return sim


class ScalingFactor(nn.Module):
    def __init__(self, output=768):
        super(ScalingFactor, self).__init__()
        self.scale1 = nn.Parameter(torch.tensor(1.0))  # 初始缩放因子为1.0
        self.scale2 = nn.Parameter(torch.tensor(1.0))  # 初始缩放因子为1.0


    def forward(self, input1, input2):

        # return self.l1(input1 * self.scale1),  self.l2(input2 * self.scale2)
        return input1 * self.scale1,  input2 * self.scale2
        # 增加一个线性层。
