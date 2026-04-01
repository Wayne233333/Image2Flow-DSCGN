import os
import pandas as pd
import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)

import sys
sys.path.append("..")
import config as config

# ================= 1. 参数配置 =================
csv_path = os.path.join(config.DATA_DIR, "Vis", f"train_on_{config.REGION}_{config.YEAR}.csv")  # 影像特征向量 CSV
shp_path = os.path.join(config.DATA_DIR, config.REGION, "shapefile_poi", f"tl_{config.YEAR}_{config.REGION}_tract_poi.shp")  # 刚才导出的带有 POI 的 SHP
model_save_path = os.path.join("ckpt", f"{config.REGION}_poi_predictor_model.pth")  # 模型保存路径
log_path = os.path.join("..", "log", f"train_poi_predictor_{config.REGION}_{config.YEAR}.log")  # 日志文件路径

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_path),
    logging.StreamHandler()
])
logger = logging.getLogger('POIPrediction')

id_col = "GEOID"

feature_dim = 128

poi_columns = ['amenity', 'shop', 'building', 'highway', 'railway', 'tourism', 'leisure', 'office', 'industrial']

batch_size = 32
learning_rate = 0.001
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 2. 数据对齐与预处理 =================
logging.info("正在读取并对齐数据...")
img_df = pd.read_csv(csv_path)
poi_gdf = gpd.read_file(shp_path)

img_df[id_col] = img_df['geocode'].astype(str)
poi_gdf[id_col] = poi_gdf[id_col].astype(str)

img_features = img_df.drop(columns=[id_col]).values
img_ids = img_df[id_col].values
feature_dim = img_features.shape[1]

poi_gdf[poi_columns] = np.log1p(poi_gdf[poi_columns].values)

merged_df = pd.merge(img_df, poi_gdf[[id_col] + poi_columns], on=id_col, how='inner')
logging.info(f"对齐后的有效普查区样本量: {len(merged_df)}")

X = merged_df.drop(columns=[id_col] + poi_columns).values.astype(np.float32)
y = merged_df[poi_columns].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ================= 3. 构建 PyTorch 数据加载器 =================
class POIDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


train_dataset = POIDataset(X_train, y_train)
test_dataset = POIDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ================= 4. 构建多层感知机 (MLP) =================
class POIPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(POIPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU()  # POI 预测值不应为负数
        )

    def forward(self, x):
        return self.mlp(x)


model = POIPredictor(input_dim=feature_dim, output_dim=len(poi_columns)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ================= 5. 训练闭环 =================
logging.info(f"开始在 {device} 上训练模型...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_X.size(0)

    train_loss /= len(train_loader.dataset)

    # 验证集评估
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)

    test_loss /= len(test_loader.dataset)

    if (epoch + 1) % 10 == 0:
        logging.info(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# ================= 6. 保存模型 =================
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'poi_columns': poi_columns
}, model_save_path)

logging.info(f"模型与标准化器已保存至: {model_save_path}")