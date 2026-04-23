import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

SEQ_LEN = 20

# ==========================================
# 3. ユーティリティ関数（前処理など）
# ==========================================
class Utils:
    @staticmethod
    def resample(points, num_points=20):
        """可変長の点配列を固定長にリサンプリング"""
        if len(points) < 2: return np.zeros((num_points, 2))

        # 累積距離を計算
        dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cum_dists = np.concatenate(([0], np.cumsum(dists)))
        total_len = cum_dists[-1]

        if total_len == 0: return np.zeros((num_points, 2))

        # 等間隔の距離を作成
        new_dists = np.linspace(0, total_len, num_points)

        # 線形補間で新しい座標を取得
        new_x = np.interp(new_dists, cum_dists, points[:, 0])
        new_y = np.interp(new_dists, cum_dists, points[:, 1])

        return np.stack([new_x, new_y], axis=1)

    @staticmethod
    def normalize(points):
        """正規化（中心化とスケーリング）"""
        points = points - np.mean(points, axis=0)
        max_val = np.max(np.abs(points))
        if max_val > 0:
            points = points / max_val
        return points

    @staticmethod
    def add_noise(points, scale=0.05):
        """手書き風のブレ（ノイズ）を加える"""
        noise = np.random.normal(0, scale, points.shape)
        return points + noise
    
    def __init__(self):
       self.model = StrokeDirectionModel(input_size=SEQ_LEN*2)
       # 2. モデルのパラメータをロードする (CPU/GPUの自動設定)
       self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       self.model.load_state_dict(torch.load('model_weights.pth', map_location=self.device))
       
       # 3. 推論モードに切り替える (必須)
       self.model.eval()

    def sort_stroke(self,path):
        resampled = Utils.resample(path)
        normalized = Utils.normalize(resampled)

        # 推論
        tensor_in = torch.from_numpy(normalized.flatten().astype(np.float32))
        with torch.no_grad():
            prob = self.model(tensor_in).item()

        if prob < 0.5:
            return path
        else:
            return path[::-1]

# ==========================================
# 5. モデル定義
# ==========================================
class StrokeDirectionModel(nn.Module):
    def __init__(self, input_size=40):
        super(StrokeDirectionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)