import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# ===================== 离线预处理脚本 =====================
# 运行一次，将 Excel 转换为 .parquet 和 嵌入向量 .pt

# 1. 文件路径
base_dir = os.path.dirname(os.path.abspath(__file__))
NEVO_FILE = os.path.join(base_dir, "NEVO2023_database.xlsx")
PARQUET_OUT = os.path.join(base_dir, "NEVO2023_database.parquet")
EMBEDDING_OUT = os.path.join(base_dir, "nevo_embeddings.pt")

# 2. 加载 Excel 文件
print("[INFO] 读取 Excel 文件...")
df = pd.read_excel(NEVO_FILE)
df.columns = [c.strip() for c in df.columns]
df["Food name"] = df["Food name"].astype(str).str.replace(",", "").str.strip()
df["CleanName"] = df["Food name"].str.lower()

# 3. 转换数值列
numeric_cols = ["Calories", "Fat", "protein", "Carbs"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. 添加 WheelGroup
print("[INFO] 添加 Wheel 分组标签...")
def wheel_group(cat: str):
    if not isinstance(cat, str):
        return None
    c = cat.lower()
    if "fats and oils" in c:
        return "Fats"
    if "vegetables" in c or "fruit" in c:
        return "Veg&Fruit"
    if any(x in c for x in ["bread", "cereals", "potatoes", "rice"]):
        return "Carbs"
    if any(x in c for x in ["milk", "cheese", "egg", "meat", "fish", "nut"]):
        return "Protein"
    if "beverages" in c:
        return "Drinks"
    return None

df["WheelGroup"] = df["Category"].apply(wheel_group)

# 5. 保存为 .parquet
print("[INFO] 保存为 Parquet 文件...")
df.to_parquet(PARQUET_OUT, index=False)

# 6. 编码 CleanName 向量并保存
print("[INFO] 加载模型并编码向量...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
embeddings = model.encode(df["CleanName"].tolist(), convert_to_tensor=True)
torch.save(embeddings, EMBEDDING_OUT)

print("✅ 预处理完成：")
print(" - 数据文件：", PARQUET_OUT)
print(" - 向量文件：", EMBEDDING_OUT)