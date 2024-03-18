import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import time
from PIL import Image
from flask import Flask, request, redirect, url_for, render_template # , Markup
from markupsafe import Markup
from werkzeug.utils import secure_filename
import torch.nn.functional as F



# 現時点(2024/03/18時点)でのメイン





# 2024/03/04 これが現在のメイン
# #ライブラリのインポート
import math
# import numpy as np
import pandas as pd
# # import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# import torch
# import torch.nn as nn
from torch.nn import LayerNorm
# from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

import pandas as pd

import datetime

# # 自分で集めたデータを前処理してモデルに入力
# # my_data内の画像から4種類を分類するモデル

# "hands-on/src/resnet/my_src/から移動"

# "Target Object Name"
# obj1 = "apple"
# obj2 = "orange"
# obj3 = "banana"
# obj4 = "pine"

# UPLOAD_FOLDER = "./static/images/"
UPLOAD_FOLDER = "./upload_data/images"
# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
# labels = ["りんご", "みかん", "バナナ", "パイナップル"]
# n_class = len(labels)
# img_size = 64 # 32
# n_result = 3  # 上位3つの結果を表示
# if __name__ == "__main__":

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
# パラメータなどの定義
d_input = 1
d_output = 1
d_model = 512
nhead = 8
dim_feedforward = 2048
num_encoder_layers = 1
num_decoder_layers = 1
dropout = 0.01
src_len = 20 # 36 # 18 # 3年分のデータから
tgt_len = 7 # 12 # 6 # 1年先を予測する
batch_size = 1
epochs = 100 # 30 # 100 # 5 # 0 # 30 # 5 # 0 # 30+70 # 300
best_loss = float('Inf')
best_model = None

load_model = True




multi_data = False # True

if not multi_data:
    "*** 変更前 ***"
    # df = pd.read_csv("/home/ubuntu/App-Project/action_list_app/input/input_data_from_local.csv",sep=",")
    df = pd.read_csv("./input/input_data_from_local.csv",sep=",")
    df.columns = ["date", "actions"]
    from datetime import datetime as dt
    # df.date = df.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))
    df.date = df.date.apply(lambda d: dt.strptime(str(d), '%m%d%H%M%S')) # YYYYを飛ばしてmmから取り込んでいる

    df = df.sort_values(['date'])
    print("df : ", df)
else:
    "*** 変更後 ***"
    import pandas as pd
    # df1 = pd.read_csv("kabuka_small_add_day.csv",sep=",")
    # df2 = pd.read_csv("gouseizyusi.csv",sep=",")
    # df3 = pd.read_csv("test_small.csv",sep=",") # pd.read_csv("kabuka_small.csv",sep=",")
    # df1.columns = ["date", "actions"]
    # df2.columns = ["date", "actions"]
    # df3.columns = ["date", "actions"]
    # from datetime import datetime as dt
    # df1.date = df1.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))
    # df2.date = df2.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))
    # df3.date = df3.date.apply(lambda d: dt.strptime(str(d), "%Y/%m/%d"))
    # # 時系列データを結合して入力とする
    # # input_data = torch.stack((time_series1, time_series2, time_series3), dim=1)
    # # input_data = torch.stack((df1, df2, df3), dim=1)
    # print("*****")
    # print(df1)
    # print("*****")
    # print(df2)
    # print("*****")
    # print(df3)
    # print("*****")
    pass


# app = Flask(__name__)
app = Flask(__name__, static_folder='upload_data', template_folder='./my_templates') # ../my_templates') # デフォルトはstatic
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS # 許容する拡張子内に含まれているならTrueを返す


@app.route("/", methods=["GET", "POST"]) #  直下のフォルダに来た場合は直下のindex関数を実行
def index():
    return render_template("index.html") # htmlを表示

"***** 重要 *****"
# このURLを指定してアクセスした場合、画像をアップロードすると以下の処理が行われて予測が出力される.
@app.route("/result", methods=["GET", "POST"]) # resultというURLに来た場合はresult関数を実行
def result():
    # if request.method == "POST": # 画像をweb上に投稿(アップロード)
    #     # ファイルの存在と形式を確認
    #     if "file" not in request.files:
    #         print("File doesn't exist!")
    #         return redirect(url_for("index")) # Topに戻る
    #     file = request.files["file"]
    #     if not allowed_file(file.filename):
    #         print(file.filename + ": File not allowed!")
    #         return redirect(url_for("index")) # Topに戻る

    #     # ファイルの保存
    #     # if os.path.isdir(UPLOAD_FOLDER):
    #     #     shutil.rmtree(UPLOAD_FOLDER)
    #     # os.mkdir(UPLOAD_FOLDER)
        
    #     # add
    #     if not os.path.isdir(UPLOAD_FOLDER):
    #         os.mkdir(UPLOAD_FOLDER)
    #     filename = secure_filename(file.filename)  # ファイル名を安全なものに
    #     filepath = os.path.join(UPLOAD_FOLDER, filename)
    #     file.save(filepath)

    #     # 画像の読み込み
    #     image = Image.open(filepath)
    #     image = image.convert("RGB") # アップロードされた画像が「モノクロや透明値αが含まれるかもしれない」からRGBに変換して入力画像を統一
    #     image = image.resize((img_size, img_size)) # 入力画像を32*32にする

    #     normalize = transforms.Normalize(
    #         (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # 平均値を0、標準偏差を1に
    #     to_tensor = transforms.ToTensor()
    #     transform = transforms.Compose([to_tensor, normalize])

    #     x = transform(image)
    #     x = x.reshape(1, 3, img_size, img_size) # バッチサイズ, チャンネル数, 高さ, 幅

    #     # これでようやくNNに入力できるデータ形式になる


    #     model = CNN(4)
    #     # パラメータの読み込み
    #     param_load = torch.load("model-for-ec2-mlp.param")
    #     model.load_state_dict(param_load)
    #     # validation(validation_loader, model)
    #     pred = model(x)
    #     pred = F.softmax(pred, dim=1)[0] # 10個の出力が確率になる
    #     sorted_idx = torch.argsort(-pred)  # 降順でソート # 大きい順で並べてindexをargsortで取得
    #     result = ""
    #     # 今回は結果を3つ表示
    #     for i in range(n_result):
    #         idx = sorted_idx[i].item() # 大きい順にソートしているので、最も大きい値が入る
    #         ratio = pred[idx].item()
    #         label = labels[idx]
    #         result += "<p>" + str(round(ratio*100, 1)) + \
    #             "%の確率で" + label + "です。</p>"
    #     return render_template("result.html", result=Markup(result), filepath=filepath) # result.htmlにこの結果を表示
    # else:
    #     return redirect(url_for("index")) # POSTがない場合はトップに戻る

    
    # パラメータなどの定義
    d_input = 1
    d_output = 1
    d_model = 512
    nhead = 8
    dim_feedforward = 2048
    num_encoder_layers = 1
    num_decoder_layers = 1
    dropout = 0.01
    # src_len = 36 # 18 # 3年分のデータから
    # tgt_len = 12 # 6 # 1年先を予測する

    "train, val, test すべてのデータが20+10以上必要 >> len < 0になってエラー"
    # src_len = 20
    # tgt_len = 10
    src_len = 20 # 36 # 18 # 3年分のデータから
    tgt_len = 7 # 12 # 6 # 1年先を予測する

    
    batch_size = 1
    epochs = 100 # 30
    best_loss = float('Inf')
    best_model = None

    load_model = True

    model = Transformer(num_encoder_layers=num_encoder_layers,
                        num_decoder_layers=num_decoder_layers,
                        d_model=d_model,
                        d_input=d_input, 
                        d_output=d_output,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout, nhead=nhead
                    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model = model.to(device)

    criterion = torch.nn.MSELoss() # オーバーフローに注意

    optimizer = torch.optim.RAdam(model.parameters(), lr=0.0001) # 0.00001) # 今回は値が大きいので小さすぎると全然変わらなくなる

    if not load_model:
        # 訓練と評価用データにおける評価
        valid_losses = []
        for epoch in range(1, epochs + 1):
            
            loss_train = train(
                model=model, data_provider=data_provider('train', src_len, tgt_len, batch_size), optimizer=optimizer,
                criterion=criterion
            )
                
            loss_valid = evaluate(
                flag='val', model=model, data_provider=data_provider('val', src_len, tgt_len, batch_size), criterion=criterion
            )
            
            if epoch%10==0:
            # if epoch%1==0:
                print('[{}/{}] train loss: {:.2f}, valid loss: {:.2f}'.format(
                    epoch, epochs,
                    loss_train, loss_valid,
                ))
                
            valid_losses.append(loss_valid)
            
            if best_loss > loss_valid:
                best_loss = loss_valid
                best_model = model

        print("data provider : ", data_provider)

        # テスト用データにおける予測
        r = evaluate(flag='test', model=best_model, data_provider=data_provider('test', src_len, tgt_len, batch_size), criterion=criterion)
        print(r)

        # パラメータの保存
        params = model.state_dict()
        torch.save(params, "weights_model.param")

    else:
        # パラメータの読み込み
        param_load = torch.load("weights_model.param")
        model.load_state_dict(param_load)
        
        # 訓練と評価用データにおける評価
        valid_losses = []
        val_epochs = 1
        for epoch in range(1, val_epochs + 1):
            
            loss_valid = evaluate(
                flag='val', model=model, data_provider=data_provider('val', src_len, tgt_len, batch_size), criterion=criterion
            )
            
            
            if epoch%1==0:
                print('[{}/{}] valid loss: {:.2f}'.format(
                    epoch, epochs,
                    loss_valid,
                ))
                
            valid_losses.append(loss_valid)
            
            if best_loss > loss_valid:
                best_loss = loss_valid
                best_model = model

        # テスト用データにおける予測
        r, date, next_action_list, next_action, ret = evaluate(flag='test', model=best_model, data_provider=data_provider('test', src_len, tgt_len, batch_size), criterion=criterion)
        print(r)
    
    result = ""
    # # 今回は結果を3つ表示
    # # for i in range(3): # n_result):
    # #     # idx = sorted_idx[i].item() # 大きい順にソートしているので、最も大きい値が入る
    # #     # ratio = pred[idx].item()
    # #     # label = labels[idx]
    # #     # result += "<p>" + str(round(ratio*100, 1)) + \
    # #     #     "%の確率で" + label + "です。</p>"
    # #     result += "<p>" + str(round(ratio*100, 1)) + \
    # #         "%の確率で" + label + "です。</p>"
    print("***** 予測結果 *****")
    print(f"{date[0]} t=1, action:{next_action_list[round(next_action[0][0][0])]}")
    print(f"{date[1]} t=2, action:{next_action_list[round(next_action[0][1][0])]}")
    print(f"{date[2]} t=3, action:{next_action_list[round(next_action[0][2][0])]}")
    # result = f"[Next Action is ...]{date[0]} t=1, action:{next_action_list[round(next_action[0][0][0])]}"
    result += "<p>" + f"[Next Action is ...] (t+1)'s action:{next_action_list[round(next_action[0][0][0])]}</p>"

    result += "<p>" + f"[Next Action is ...] (t+2)'s action:{next_action_list[round(next_action[0][1][0])]}</p>"
    result += "<p>" + f"[Next Action is ...] (t+3)'s action:{next_action_list[round(next_action[0][2][0])]}</p>"




    # dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    # JSTとUTCの差分
    DIFF_JST_FROM_UTC = 9
    dt_now = datetime.datetime.utcnow() + datetime.timedelta(hours=DIFF_JST_FROM_UTC)
    result += "<p>" + f"Date Time Now : \"{dt_now}\"</p>"

    result += "<p>" + f"Nearest Date : \"{ret[0]}\" >> Action : \"{ret[1]}\"</p>"


    # if not os.path.exists(UPLOAD_FOLDER):
    #     # os.mkdir(UPLOAD_FOLDER)
    #     os.makedirs(UPLOAD_FOLDER)
    filename = 'actions_by_load_model.png' # secure_filename(file.filename)  # ファイル名を安全なものに
    filepath = os.path.join(UPLOAD_FOLDER, filename) # '/home/ubuntu/App-Project/action_list_app/actions_by_load_model.png' # 
    # file = request.files["file"]
    # file.save(filepath)
    
    return render_template("result.html", result=Markup(result), filepath=filepath) # result.htmlにこの結果を表示


# データのロードと実験用の整形
class AirPassengersDataset(Dataset):
    def __init__(self, flag, seq_len, pred_len):
        #学習期間と予測期間の設定
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        #訓練用、評価用、テスト用を分けるためのフラグ
        type_map = {'train': 0, 'val': 1, 'test':2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        
        if not multi_data:
            #seabornのデータセットから飛行機の搭乗者数のデータをロード
            df_raw = df # sns.load_dataset('flights')
            
            "*****"
            data_len = len(df_raw)
            # print("data len : {}".format(data_len))
            # border1s = [0, 56-self.seq_len, 50] # [train0, val0, test0] # test0~test1=24
            # border2s = [56, 38+30, 74]          # [train1, val1, test1]
            # 1~66
            # border1s = [0, 20, 36] # [train0, val0, test0] # test0~test1=24
            # border2s = [40, 50, 66]          # [train1, val1, test1]
            border1s = [0, 20, 39] # [train0, val0, test0] # test0~test1=24
            border2s = [39, 47, 66]          # [train1, val1, test1]

            
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            data = df_raw[['actions']].values
            ss = StandardScaler()

            print("20240318 data : ", data)

            # 2024/03/18
            # srcが小さくなっていた原因
            # data = ss.fit_transform(data) 正規化してしまっている
            # print("20240318 data : ", data)

            print("border1: {}, boredr2: {}".format(border1, border2))
            self.data = data[border1:border2]

            "# 全部のデータで学習しちゃっているかも 2024/03/06"
            # # 試しにかっこを一つ削除
            # data = df_raw[['actions']].values # *0.001 大きすぎるとnanになる
            # # data = df_raw['actions'].values # *0.001 大きすぎるとnanになる
            # self.data = data
            "*****"
        else:
            # input_data = torch.stack((torch.tensor(df1['actions'].values), torch.tensor(df2['actions'].values), torch.tensor(df3['actions'].values)), dim=1)
            # data = input_data
            # self.data = data
            pass

    def __getitem__(self, index):
        #学習用の系列と予測用の系列を出力
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        src = self.data[s_begin:s_end]
        tgt = self.data[r_begin:r_end]

        return src, tgt
    
    def __len__(self):
        print("data : ", self.data)
        print("data len : ", len(self.data)) # 15(予測時)
        print("len : ", len(self.data) - self.seq_len - self.pred_len + 1) # -32
        return len(self.data) - self.seq_len - self.pred_len + 1

# DataLoaderの定義
def data_provider(flag, seq_len, pred_len, batch_size):
    #flagに合ったデータを出力
    data_set = AirPassengersDataset(flag=flag, 
                                    seq_len=seq_len, 
                                    pred_len=pred_len
                                   )
    #データをバッチごとに分けて出力できるDataLoaderを使用
    data_loader = DataLoader(data_set,
                             batch_size=batch_size, 
                            #  shuffle=True # これが原因
                            shuffle=False # これが原因
                            )
    
    return data_loader

# エンべディングの定義
#位置エンコーディングの定義
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#モデルに入力するために次元を拡張する
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Linear(c_in, d_model) 

    def forward(self, x):
        x = self.tokenConv(x)
        return x

# Transformerの定義
class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers,
        d_model, d_input, d_output,
        dim_feedforward = 512, dropout = 0.1, nhead = 8):
        
        super(Transformer, self).__init__()
        

        #エンべディングの定義
        self.token_embedding_src = TokenEmbedding(d_input, d_model)
        self.token_embedding_tgt = TokenEmbedding(d_output, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        #エンコーダの定義
        encoder_layer = TransformerEncoderLayer(d_model=d_model, 
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                               )
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 
                                                      num_layers=num_encoder_layers,
                                                      norm=encoder_norm
                                                     )
        
        #デコーダの定義
        decoder_layer = TransformerDecoderLayer(d_model=d_model, 
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                               )
        decoder_norm = LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(decoder_layer, 
                                                      num_layers=num_decoder_layers, 
                                                      norm=decoder_norm)
        
        #出力層の定義
        self.output = nn.Linear(d_model, d_output)
        

    def forward(self, src, tgt, mask_src, mask_tgt):
        #mask_src, mask_tgtはセルフアテンションの際に未来のデータにアテンションを向けないためのマスク
        
        embedding_src = self.positional_encoding(self.token_embedding_src(src))
        memory = self.transformer_encoder(embedding_src, mask_src)
        
        embedding_tgt = self.positional_encoding(self.token_embedding_tgt(tgt))
        outs = self.transformer_decoder(embedding_tgt, memory, mask_tgt)
        
        output = self.output(outs)
        return output

    def encode(self, src, mask_src):
        return self.transformer_encoder(self.positional_encoding(self.token_embedding_src(src)), mask_src)

    def decode(self, tgt, memory, mask_tgt):
        return self.transformer_decoder(self.positional_encoding(self.token_embedding_tgt(tgt)), memory, mask_tgt)

# マスクの定義
def create_mask(src, tgt):
    
    seq_len_src = src.shape[1]
    seq_len_tgt = tgt.shape[1]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt).to(device)
    mask_src = generate_square_subsequent_mask(seq_len_src).to(device)

    return mask_src, mask_tgt


def generate_square_subsequent_mask(seq_len):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask

# 訓練、評価の処理を定義
def train(model, data_provider, optimizer, criterion):
    model.train()
    total_loss = []
    for src, tgt in data_provider:
        
        src = src.float().to(device)
        tgt = tgt.float().to(device)

        input_tgt = torch.cat((src[:,-1:,:],tgt[:,:-1,:]), dim=1)

        mask_src, mask_tgt = create_mask(src, input_tgt)

        output = model(
            src=src, tgt=input_tgt, 
            mask_src=mask_src, mask_tgt=mask_tgt
        )

        optimizer.zero_grad()

        loss = criterion(output, tgt)
        loss.backward()
        total_loss.append(loss.cpu().detach())
        optimizer.step()
        
    return np.average(total_loss)


def evaluate(flag, model, data_provider, criterion):
    model.eval()
    total_loss = []
    

    for src, tgt in data_provider:
        
        src = src.float().to(device)
        tgt = tgt.float().to(device)

        seq_len_src = src.shape[1]
        mask_src = (torch.zeros(seq_len_src, seq_len_src)).type(torch.bool)
        mask_src = mask_src.float().to(device)
    
        memory = model.encode(src, mask_src)
        outputs = src[:, -1:, :]
        seq_len_tgt = tgt.shape[1]
    
        for i in range(seq_len_tgt - 1): # tgtの長さ分だけループ
        
            mask_tgt = (generate_square_subsequent_mask(outputs.size(1))).to(device)
        
            output = model.decode(outputs, memory, mask_tgt)
            output = model.output(output)

            outputs = torch.cat([outputs, output[:, -1:, :]], dim=1) # ここで予測を1つずつ追加している
            # print("TEST 2024/02/10")
        
        loss = criterion(outputs, tgt)
        total_loss.append(loss.cpu().detach())
        print("out : ", output)
        
    if flag=='test':
        print("src: ", src)
        true = torch.cat((src, tgt), dim=1)
        print("true: ", true)
        pred = torch.cat((src, output), dim=1) # src=予測したい時刻




        
        # pred = torch.cat((src, outputs), dim=1) # ???
        print("target: ", tgt)
        print("output: ", output.detach().numpy())
        print("pred  : ", pred.detach().numpy())
        # plt.plot(true.squeeze().cpu().detach().numpy(), label='true')
        # plt.plot(pred.squeeze().cpu().detach().numpy(), label='pred')

        # plt.style.use("ggplot")

        
        plt.plot(df['date'][0:-1:1], df['actions'][0:-1:1], label='true', alpha=0.5)
        "***"
        "main__this.py ver."
        index = df["date"][-(seq_len_src)-2:-(seq_len_tgt):1] # -37 ~ -11
        index2 = df["date"][-(seq_len_tgt)-1:-2:1] # -11 ~ の予測
        plt.plot(index, pred.squeeze().cpu().detach().numpy()[-(seq_len_src)-1:-(seq_len_tgt-1):1], label='src(input)') # 'true') # , color="blue") # true.detach())
        plt.plot(index2, pred.squeeze().cpu().detach().numpy()[-(seq_len_tgt):-1:1], label='pred(output)')
        index_tgt = df["date"][-(seq_len_tgt)-1:-1:1] # target
        plt.plot(index_tgt, true.squeeze().cpu().detach().numpy()[-(seq_len_tgt)-1:-1:1], label='tgt(true)', alpha=0.5) # , color="green") # true.detach())
        "***"
        plt.legend()
        plt.grid(True)
        
        # y軸を行動に変更, 次の行動を予測
        " *** ADD *** "
        next_action_list = [
            "wakeup",
            "go work",
            "breakfast",
            "coffee",
            "working",
            "lunch",
            "tooth brush",
            "going out",
            "meeting",
            "study(English)",
            "gym",
            "buy dinner",
            "back home",
            "bath",
            "study(ML)",
            "go bed",
            "study",
            "Movie",
            "study(AWS)",
            "study(other)",
            "TV/Youtube",
            "sleep",
        ]
        # next_action_list = {
        #     "wakeup":0,
        #     "go work":1,
        #     "breakfast":2,
        #     "coffee":3,
        #     "working":4,
        #     "lunch":5,
        #     "tooth brush":6,
        #     "going out":7,
        #     "meeting":8,
        #     "study(English)":9,
        #     "gym":10,
        #     "buy dinner":11,
        #     "back home":12,
        #     "bath":13,
        #     "study(ML)":14,
        #     "go bed":15,
        #     "study":16,
        #     "Movie":17,
        #     "study(AWS)":18,
        #     "study(other)":19,
        #     "TV/Youtube":20,
        #     "sleep":21,
        # }
        # total_data = df
        # # total_data = total_data.reset_index(drop=True)
        # # replace = total_data.sort_values(['date'])
        # replace = df
        # print(replace)
        # # # plt.plot(total_data['date'], pre['actions'])
        # # plt.plot(replace['date'], replace['actions'], label='actions', color='orange')
        # plt.legend()
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], next_action_list)

        # plt.show()


        print("outputs:", outputs)
        print("output: ", output)
        print("target: ", tgt)
        print("**********")
        next_action = output.detach().numpy()
        print("next action pred1: ", next_action[0][0][0])
        print("next action pred2: ", next_action[0][1][0])
        print("next action pred3: ", next_action[0][2][0])
        # print("next action pred1: ", int(next_action[0][0]))
        # next_action_list = [ "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R"]
        print("(t+1)next action is ... ", next_action_list[round(next_action[0][0][0])])
        print("(t+2)next action is ... ", next_action_list[round(next_action[0][1][0])])
        print("(t+3)next action is ... ", next_action_list[round(next_action[0][2][0])])
        
        # 予測期間の日付
        date = df["date"][-(seq_len_tgt)-1:-2:1]
        date = date.reset_index(drop=True)

        pred_date = pd.DataFrame(
            df["date"][-(seq_len_tgt)-1:-2:1],
            columns=['date']
        )
        # print("date : ", date)
        pred_date = pred_date.reset_index(drop=True)
        
        next = [
            next_action_list[round(next_action[0][0][0])],
            next_action_list[round(next_action[0][1][0])],
            next_action_list[round(next_action[0][2][0])],
            next_action_list[round(next_action[0][3][0])],
            next_action_list[round(next_action[0][4][0])],
            next_action_list[round(next_action[0][5][0])],
            # next_action_list[round(next_action[0][6][0])],
            # next_action_list[round(next_action[0][7][0])],
            # next_action_list[round(next_action[0][8][0])],
            # next_action_list[round(next_action[0][9][0])],
            # next_action_list[round(next_action[0][10][0])],
        ]
        print("next : ", next)
        Actions = pd.DataFrame(
            next,
            #  index=[df.shape[0]]
            columns=['actions'] # , 'date']
        )
        print(date)
        print(Actions)
        # Actions = pd.DataFrame(
        #     [date, next],
        #     #  index=[df.shape[0]]
        #     columns=['date', 'actions'] # , 'date']
        # )

        pred_dateand_actions = pd.concat([pred_date, Actions], axis=1)
        # date = pd.DataFrame(date, Actions)
        print(pred_dateand_actions)

        pred_dateand_actions.to_csv("pred_date_actions.csv")

        
        
        
        
        
        "***** add 2024/03/04 *****"
        # 本来はここに一週間分や一日分の推論結果をお指定するが、今回はダミーデータを推論結果として出力されたとして扱う
        
        # df にするとエラー
        # df2 = pd.read_csv("./pred_date_actions.csv",sep=",")
        df2 = pd.read_csv("./pred_date_actions_pre.csv",sep=",") # 今はテストデータを読み込む(本来は推論結果を読み込む)
        "これはテストデータ >> 次回は現在時刻に一番近い時間=t+1となるようにする >> ということは推論結果は未来の時刻にならないといけない（＝学習データは現在時刻以前になるようにする）"
        
        # # df2 =  pred_dateand_actions
        # df2 = df2.drop(df2.columns[0], axis=1)
        # df2 = df2.reset_index(drop=True)
        # print(df2)
        # # df = df.set_index('date',drop=True)
        # print(df2)
        # df2 = df2.to_dict()
        # import pprint
        # pprint.pprint(df2)
        # # print(df['date'])
        # # print(df[1900-02-27 07:03:00])
        # print("*****")
        # key = 5
        # print("key = ", key)
        # print("date : ", df2['date'][key])
        # print("action : ", df2['actions'][key])
        # print("*****")

        "*****"
        # 2024/03/04
        def idx_of_the_nearest(data, value):
            idx = np.argmin(np.abs(data - value))
            print("idx : ", idx)
            return idx

        print("df2 : ", df2)
        # df2 = df2.set_index(df['date'], drop=True)
        
        "次回、この時間は入力した値を入れる"
        # import datetime
        # dt_now = datetime.datetime.now()
        # JSTとUTCの差分
        DIFF_JST_FROM_UTC = 9
        dt_now = datetime.datetime.utcnow() + datetime.timedelta(hours=DIFF_JST_FROM_UTC)
        print(dt_now)

        time = dt_now # '1900-02-27 12:00:00'

        time = pd.to_datetime(time) # '1900-02-27 07:03:00')
        # target_index = df.index[df.index.get_loc(time, method='nearest')]
        # print(target_index)
        print(time)

        table_time = pd.to_datetime(df2['date'])
        print('-----')
        print(table_time)
        n = idx_of_the_nearest(table_time, time)
        print(n)
        key = n
        print("key = ", key)
        print("date : ", df2['date'][key])
        print("action : ", df2['actions'][key])
        ret_date = df2['date'][key]
        ret_action = df2['actions'][key]
        ret = [ret_date, ret_action] # これを返す
        print("*****")
        "*****"

        "***** add 2024/03/04 *****"




        



        # date.index
        # 今後を予測したいなら、predのindex(x軸, date)を今後の日付にする
        print("***** 予測結果 *****")
        print(f"{date[0]} t=1, action:{next_action_list[round(next_action[0][0][0])]}")
        print(f"{date[1]} t=2, action:{next_action_list[round(next_action[0][1][0])]}")
        print(f"{date[2]} t=3, action:{next_action_list[round(next_action[0][2][0])]}")

        true_next_action = tgt.detach().numpy()
        print("***** 正解 *****")
        print("true (t+1)next action is ... ", next_action_list[round(true_next_action[0][0][0])])
        print("true (t+2)next action is ... ", next_action_list[round(true_next_action[0][1][0])])
        print("true (t+3)next action is ... ", next_action_list[round(true_next_action[0][2][0])])
        " *** ADD *** "
        print("***** test *****")
        
        plt.savefig('./upload_data/images/actions_by_load_model.png')
        
        return np.average(total_loss), date, next_action_list, next_action, ret

    # print("test : {}".format(seq_len_src))
        
    return np.average(total_loss) # , next_action_list, next_action




if __name__ == "__main__":
    # app.run(debug=True)
    app.debug = True
    app.run(host='0.0.0.0', port=80) # 888) # 0)