from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import torch
import requests
import subprocess
import tempfile
import os

from fastapi import UploadFile, Form
import requests
import base64


# ===================== 1) Initialize FastAPI + CORS =====================
app = FastAPI(
    title="NEVO Food Nutrition API (Extended)",
    description="""
Extended version: 
 - /api/text (original text input)
 - /api/image (Google Vision)
 - /api/audio (OpenAI Whisper)
 - /api/confirm (user-edited items)
""",
    version="2.1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== 2) Load SentenceTransformer Model =====================
model = SentenceTransformer('all-MiniLM-L6-v2')

# ===================== 3) Load NEVO Excel =====================
base_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接出 Excel 文件路径
NEVO_FILE = os.path.join(base_dir, "NEVO2023_database.xlsx")

# 尝试读取
try:
    nevo_df = pd.read_excel(NEVO_FILE)
    nevo_df.columns = [c.strip() for c in nevo_df.columns]
except FileNotFoundError as e:
    raise RuntimeError(f"未找到 NEVO Excel 文件，请确保 'NEVO2023_database.xlsx' 与 main.py 位于同一目录下。") from e

# Clean food names
nevo_df["Food name"] = nevo_df["Food name"].astype(str).str.replace(",", "").str.strip()
nevo_df["CleanName"] = nevo_df["Food name"].str.lower()

# Encode all NEVO food names
nevo_embeddings = model.encode(nevo_df["CleanName"].tolist(), convert_to_tensor=True)

# ===================== 4) Data Cleaning =====================
numeric_cols = ["Calories", "Fat", "protein", "Carbs"]
for col in numeric_cols:
    if col in nevo_df.columns:
        nevo_df[col] = pd.to_numeric(nevo_df[col], errors="coerce")

def wheel_group(cat: str) -> Optional[str]:
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

nevo_df["WheelGroup"] = nevo_df["Category"].apply(wheel_group)

# ===================== 5) Food Extraction (existing logic) =====================
def extract_food_items_simple(text: str) -> List[Dict[str, Any]]:
    """
    Simple regex approach to parse '100 grams apple' or '2 pieces cheese' from text.
    If no match, fallback to each word => quantity=1.
    """
    pattern = r"(\d+(?:\.\d+)?)\s*(grams?|g|ml|kg|l|pieces?|oz)?\s+([a-zA-Z ]+?)(?:,|and|\.|$)"
    matches = re.findall(pattern, text)

    results = []
    for qty_str, unit, food in matches:
        food_cleaned = food.strip().lower()
        try:
            qty = float(qty_str)
        except:
            qty = 1.0
        results.append({
            "foodName": food_cleaned,
            "quantity": qty
        })

    if not results:
        tokens = text.split()
        for word in tokens:
            if word.isalpha():
                results.append({
                    "foodName": word.lower(),
                    "quantity": 1.0
                })
    return results

# ===================== 6) SentenceTransformer Matching =====================
def vector_match_food(user_food: str, threshold: float = 0.6) -> Optional[str]:
    query_emb = model.encode(user_food.strip().lower(), convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, nevo_embeddings)[0]
    top_result = torch.argmax(cos_scores).item()
    top_score = cos_scores[top_result].item()
    if top_score < threshold:
        return None
    return nevo_df.iloc[top_result]["CleanName"]

def get_nutrition_info(row: pd.Series, qty: float) -> Dict[str, Any]:
    """
    NEVO data is per 100g => multiply by factor = qty/100
    """
    factor = qty / 100.0
    def scale(val):
        if pd.notna(val):
            return round(float(val)*factor, 2)
        return None

    return {
        "Calories": scale(row.get("Calories")),
        "Protein": scale(row.get("protein")),
        "Fat": scale(row.get("Fat")),
        "Carbs": scale(row.get("Carbs")),
        "Category": row.get("Category"),
        "WheelGroup": row.get("WheelGroup")
    }

# ===================== 7) Original /api/text Endpoint =====================
class TextInput(BaseModel):
    text: str

class ParsedFoodItem(BaseModel):
    foodName: str
    quantity: float
    unit: Optional[str] = None
    estimatedCategory: Optional[str] = None
    estimatedWeight: Optional[float] = None


@app.post("/api/text")
def parse_text(input: TextInput):
    items = extract_food_items_simple(input.text)
    output = []
    for item in items:
        fName, qty = item["foodName"], item["quantity"]
        matched = vector_match_food(fName)
        if matched is None:
            output.append({
                "foodName": fName,
                "quantity": qty,
                "NEVO_matched": None,
                "error": "No semantic match found in NEVO"
            })
            continue

        matched_row = nevo_df[nevo_df["CleanName"] == matched]
        if matched_row.empty:
            output.append({
                "foodName": fName,
                "quantity": qty,
                "NEVO_matched": matched,
                "error": "Matched name not found in NEVO database"
            })
            continue

        row = matched_row.iloc[0]
        info = get_nutrition_info(row, qty)
        output.append({
            "foodName": fName,
            "quantity": qty,
            "NEVO_matched": row["Food name"],
            **info
        })
    return {"foodItems": output}

@app.post("/api/text-parse")
def text_parse(input: TextInput):
    items = extract_food_items_simple(input.text)
    output = []
    seen = set()

    for it in items:
        name = it["foodName"]
        qty = it["quantity"]
        unit = it.get("unit")

        matched = vector_match_food(name)
        category, est_weight = None, None

        if matched:
            matched_row = nevo_df[nevo_df["CleanName"] == matched]
            if not matched_row.empty:
                row = matched_row.iloc[0]
                category = row.get("WheelGroup")
                est_weight = 100.0

        final_name = matched if matched else name

        # 去重逻辑：跳过已经加入过的 foodName
        if final_name in seen:
            continue
        seen.add(final_name)

        output.append({
            "foodName": final_name,
            "quantity": qty,
            "unit": unit,
            "estimatedCategory": category,
            "estimatedWeight": est_weight
        })

    return {"parsedItems": output}




# ===================== 8) New: /api/image (Google Vision) =====================


# @app.post("/api/image")
# async def recognize_from_image(file: UploadFile):
#     """
#     1) Read uploaded image
#     2) Send to Google Cloud Vision API -> labelAnnotations
#     3) Convert top labels to "food1 and food2"
#     4) Reuse parse_text logic -> final foodItems
#     """
#     try:
#         # 1) read file content
#         contents = await file.read()
#         encoded = base64.b64encode(contents).decode('utf-8')

#         # 2) call Google Vision
#         # replace with your real API key
#         google_vision_api_key = "AIzaSyBRa0fP7e27Kz5-jHWegOA6EYZullXlvMg"
#         url = f"https://vision.googleapis.com/v1/images:annotate?key=AIzaSyBRa0fP7e27Kz5-jHWegOA6EYZullXlvMg"
#         request_body = {
#           "requests": [
#             {
#               "image": {"content": encoded},
#               "features": [{"type": "LABEL_DETECTION", "maxResults": 5}]
#             }
#           ]
#         }
#         resp = requests.post(url, json=request_body)
#         data = resp.json()
#         label_annotations = data.get("responses",[{}])[0].get("labelAnnotations", [])
#         # Just get top 2 or 3
#         top_labels = [ann["description"] for ann in label_annotations[:3]]

#         # 3) create a textual sentence, e.g. "apple and pizza"
#         # (User must manually set quantity => or default 1.0)
#         if top_labels:
#             sentence = " and ".join(top_labels)
#         else:
#             sentence = "unknown"

#         # 4) reuse parse_text logic
#         # but we just do: for each label => quantity=1.0 => let user fix
#         # or we can do a single string => "apple and pizza" => parse
#         # best approach is to parse => but that might fail if no numeric
#         # So let's just return them as if partial parse
#         # We'll do direct approach: no quantity
#         items = []
#         for lab in top_labels:
#             items.append({"foodName": lab.lower(), "quantity": 1.0})

#         # 5) do partial matching with NEVO
#         output = []
#         for it in items:
#             fName, qty = it["foodName"], it["quantity"]
#             matched = vector_match_food(fName)
#             if matched is None:
#                 output.append({
#                     "foodName": fName,
#                     "quantity": qty,
#                     "NEVO_matched": None,
#                     "error": "No semantic match found in NEVO"
#                 })
#                 continue
#             matched_row = nevo_df[nevo_df["CleanName"] == matched]
#             if matched_row.empty:
#                 output.append({
#                     "foodName": fName,
#                     "quantity": qty,
#                     "NEVO_matched": matched,
#                     "error": "Matched name not found in NEVO DB"
#                 })
#                 continue
#             row = matched_row.iloc[0]
#             info = get_nutrition_info(row, qty)
#             output.append({
#                 "foodName": fName,
#                 "quantity": qty,
#                 "NEVO_matched": row["Food name"],
#                 **info
#             })
#         sentence = " and ".join(top_labels) if top_labels else "unknown"
#         result = text_parse(TextInput(text=sentence))

#         # 打印 Google Vision 原始标签，方便调试
#         print("📷 top_labels:", top_labels)

#         sentence = " and ".join(top_labels) if top_labels else "unknown"
#         result   = text_parse(TextInput(text=sentence))

#         # 打印经过 text_parse 后的解析结果
#         print("📷 parsedItems:", result)
#         return result 

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=400, detail=str(e))
from fastapi import UploadFile, HTTPException
import base64
import requests
from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

@app.post("/api/image")
async def recognize_from_image(file: UploadFile):
    try:
        # Step 1: 读取上传图像并编码
        contents = await file.read()
        encoded = base64.b64encode(contents).decode("utf-8")

        # Step 2: 调用 Google Cloud Vision API（最多返回5个标签）
        url = "https://vision.googleapis.com/v1/images:annotate?key=AIzaSyBRa0fP7e27Kz5-jHWegOA6EYZullXlvMg"
        request_body = {
            "requests": [{
                "image": {"content": encoded},
                "features": [{"type": "LABEL_DETECTION", "maxResults": 5}]
            }]
        }
        resp = requests.post(url, json=request_body)
        data = resp.json()

        # Step 3: 提取前5个标签
        all_labels = [
            ann["description"]
            for ann in data.get("responses", [{}])[0].get("labelAnnotations", [])[:5]
        ]

        # Step 4: 过滤掉不具体的标签
        blacklist = {"cooking", "food", "produce", "processed", "cup", "tableware", "drinkware", "ingredient"}
        top_labels = [label for label in all_labels if label.lower() not in blacklist]

        sentence = " ".join(top_labels) if top_labels else "unknown"

        print("📷 原始标签:", all_labels)
        print("📷 过滤后标签:", top_labels)
        print("📷 处理句子:", sentence)

        # Step 5: 用 parse_text 精确匹配
        parsed = parse_text(TextInput(text=sentence))
        parsed_items = parsed.get("parsedItems", [])

        # Step 6: 提取匹配成功项
        matched_items = [item for item in parsed_items if item.get("estimatedCategory") is not None]

        if matched_items:
            print("✅ 精准匹配成功，返回匹配项")
            return {"parsedItems": matched_items}
        else:
            print("⚠️ 全部未匹配，使用 text_parse fallback")
            return text_parse(TextInput(text=sentence))

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


# ===================== 9) New: /api/audio (Whisper) =====================
# @app.post("/api/audio")
# async def recognize_from_audio(file: UploadFile = File(...)):
#     try:
#         input_path = tempfile.mktemp(suffix=".webm")
#         output_path = tempfile.mktemp(suffix=".wav")
#         with open(input_path, "wb") as f:
#             f.write(await file.read())

#         subprocess.run(["ffmpeg", "-i", input_path, output_path], check=True)

#         openai_api_key = "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"
#         transcribe_url = "https://api.openai.com/v1/audio/transcriptions"

#         with open(output_path, "rb") as audio_file:
#             files = {"file": ("converted.wav", audio_file, "audio/wav")}
#             headers = {"Authorization": f"Bearer {openai_api_key}"}
#             data = {
#             "model": "whisper-1",
#             "language": "en"
#            }

#             resp = requests.post(transcribe_url, headers=headers, files=files, data=data)

#         if resp.status_code != 200:
#             raise HTTPException(status_code=resp.status_code, detail=resp.text)

#         recognized_text = resp.json().get("text", "")
#         print("[🎤 Whisper 转文字结果]:", recognized_text)

#         parse_input = TextInput(text=recognized_text)
#         result = parse_text(parse_input)
#         # return result
#         return text_parse(parse_input) 

#     except Exception as e:
#         print("[❌ Whisper 错误]:", e)
#         raise HTTPException(status_code=400, detail=str(e))



# @app.post("/api/audio")
# async def recognize_from_audio(file: UploadFile = File(...)):
#     try:
#         # 将上传的 WebM 保存为临时文件
#         input_path = tempfile.mktemp(suffix=".webm")
#         output_path = tempfile.mktemp(suffix=".wav")

#         with open(input_path, "wb") as f:
#             f.write(await file.read())

#         # 使用 ffmpeg 转换为 WAV
#         subprocess.run(["ffmpeg", "-i", input_path, output_path], check=True)

#         # 调用 Whisper 接口识别文字
#         openai_api_key = "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"
#         with open(output_path, "rb") as audio_file:
#             resp = requests.post(
#                 "https://api.openai.com/v1/audio/transcriptions",
#                 headers={"Authorization": f"Bearer {openai_api_key}"},
#                 files={"file": ("audio.wav", audio_file, "audio/wav")},
#                 data={"model": "whisper-1", "language": "en"}
#             )

#         if resp.status_code != 200:
#             raise HTTPException(status_code=resp.status_code, detail=resp.text)

#         recognized_text = resp.json().get("text", "")

#         print("[🎤 Whisper结果]", recognized_text)

#         # 直接调用轻量解析逻辑（返回 parsedItems）
#         from pydantic import BaseModel
#         class TextInput(BaseModel):
#             text: str
#         return text_parse(TextInput(text=recognized_text))

#     except Exception as e:
#         print("[❌ Whisper错误]:", e)
#         raise HTTPException(status_code=400, detail=str(e))

# ===================== 9) New: /api/audio (Whisper 直接处理 .webm) =====================
# from fastapi import UploadFile, File, HTTPException
# import requests, traceback

# @app.post("/api/audio")
# async def recognize_from_audio(file: UploadFile = File(...)):
#     """
#     1) 读取前端上传的 .webm 音频
#     2) 直接上传给 OpenAI Whisper（Whisper 官方支持 webm）
#     3) 将识别文本送入 text_parse，保持返回结构不变
#     """
#     try:
#         # 1️⃣ 读取 bytes
#         audio_bytes = await file.read()

#         # 2️⃣ Whisper 语音转文字
#         openai_api_key = (
#             "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"
#         )

#         resp = requests.post(
#             "https://api.openai.com/v1/audio/transcriptions",
#             headers={"Authorization": f"Bearer {openai_api_key}"},
#             files={
#                 # filename 任意，MIME 必须 audio/webm
#                 "file": ("audio.webm", audio_bytes, "audio/webm")
#             },
#             data={"model": "whisper-1", "language": "en"}
#         )

#         if resp.status_code != 200:
#             raise HTTPException(status_code=resp.status_code, detail=resp.text)

#         recognized_text = resp.json().get("text", "")
#         print("[🎤 Whisper结果]", recognized_text)

#         # 3️⃣ 轻量解析 → 返回 parsedItems，与原逻辑一致
#         from pydantic import BaseModel
#         class TextInput(BaseModel):
#             text: str

#         return text_parse(TextInput(text=recognized_text))

#     except Exception as e:
#         print("[❌ Whisper错误]:", e)
#         traceback.print_exc()
#         raise HTTPException(status_code=400, detail=str(e))

# from fastapi import UploadFile, File, HTTPException
# import requests
# import traceback
# from pydantic import BaseModel

# class TextInput(BaseModel):
#     text: str

# @app.post("/api/audio")
# async def recognize_from_audio(file: UploadFile = File(...)):
#     """
#     多设备音频上传兼容接口（无 ffmpeg）：
#     - 支持 .webm, .mp4, .m4a（iPhone）, .aac
#     - 自动判断 MIME 类型
#     - 发送至 OpenAI Whisper 进行识别
#     - 返回 text_parse 的结构
#     """
#     try:
#         # 1️⃣ 读取音频内容
#         audio_bytes = await file.read()
#         print("📦 文件大小:", len(audio_bytes), "bytes")

#         if len(audio_bytes) < 500:
#             raise HTTPException(status_code=400, detail="音频太短或无效")
#         if len(audio_bytes) > 2 * 1024 * 1024:
#             raise HTTPException(status_code=413, detail="音频文件过大（建议小于2MB）")

#         # 2️⃣ 检测真实 MIME 类型
#         real_mime = file.content_type
#         print("📦 实际 MIME 类型:", real_mime)

#         # 3️⃣ 设置上传给 Whisper 的 MIME 和文件名
#         if real_mime in ["audio/mp4", "video/mp4", "audio/aac", "audio/x-m4a", "audio/m4a"]:
#             filename = "audio.m4a"
#             whisper_mime = "audio/m4a"
#         else:
#             filename = "audio.webm"
#             whisper_mime = "audio/webm"

#         # 4️⃣ Whisper API 请求
#         openai_api_key = "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"  # ❗请替换为你的有效 OpenAI API key
#         resp = requests.post(
#             "https://api.openai.com/v1/audio/transcriptions",
#             headers={"Authorization": f"Bearer {openai_api_key}"},
#             files={"file": (filename, audio_bytes, whisper_mime)},
#             data={"model": "whisper-1", "language": "en"},
#             timeout=15
#         )

#         print("📡 Whisper响应码:", resp.status_code)
#         print("📡 Whisper响应体:", resp.text)

#         if resp.status_code != 200:
#             raise HTTPException(status_code=resp.status_code, detail=resp.text)

#         # 5️⃣ 提取 Whisper 返回的文本
#         recognized_text = resp.json().get("text", "")
#         print("[🎤 Whisper识别结果]:", recognized_text)

#         # 6️⃣ 调用 text_parse 返回结构
#         return text_parse(TextInput(text=recognized_text))

#     except Exception as e:
#         print("[❌ Whisper处理异常]:", e)
#         traceback.print_exc()
#         raise HTTPException(status_code=400, detail=str(e))

# from fastapi import UploadFile, File, HTTPException
# from pydantic import BaseModel
# import requests
# import traceback

# import magic

# SUPPORTED_MIME_MAP = {
#     "audio/webm": "webm",
#     "audio/mpeg": "mp3",
#     "audio/mp3": "mp3",
#     "audio/mp4": "mp4",
#     "audio/x-m4a": "m4a",
#     "audio/m4a": "m4a",
#     "video/mp4": "mp4",
#     "audio/aac": "m4a",
#     "audio/wav": "wav",
#     "audio/ogg": "ogg",
#     "audio/oga": "ogg",
#     "audio/mpga": "mp3",
#     "audio/flac": "flac"
# }

# @app.post("/api/audio")
# async def recognize_from_audio(file: UploadFile = File(...)):
#     try:
#         audio_bytes = await file.read()
#         print("📦 文件大小:", len(audio_bytes), "bytes")

#         if len(audio_bytes) < 500:
#             raise HTTPException(status_code=400, detail="音频太短或无效")
#         if len(audio_bytes) > 2 * 1024 * 1024:
#             raise HTTPException(status_code=413, detail="音频文件过大（建议小于2MB）")

#         # 1️⃣ magic 检测真实 MIME 类型
#         mime = magic.Magic(mime=True)
#         real_mime = mime.from_buffer(audio_bytes[:2048])
#         print("📦 magic 检测 MIME 类型:", real_mime)

#         if real_mime in SUPPORTED_MIME_MAP:
#             file_ext = SUPPORTED_MIME_MAP[real_mime]
#             filename = f"audio.{file_ext}"
#             whisper_mime = f"audio/{file_ext}"
#         else:
#             # 强制 fallback 为 webm
#             print(f"⚠️ 未知 MIME 类型 {real_mime}，强制设为 webm")
#             filename = "audio.webm"
#             whisper_mime = "audio/webm"

#         # 2️⃣ Whisper 请求
#         openai_api_key = "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"  # 替换为你的 key
#         resp = requests.post(
#             "https://api.openai.com/v1/audio/transcriptions",
#             headers={"Authorization": f"Bearer {openai_api_key}"},
#             files={"file": (filename, audio_bytes, whisper_mime)},
#             data={"model": "whisper-1", "language": "en"},
#             timeout=15
#         )

#         print("📡 Whisper响应码:", resp.status_code)
#         print("📡 Whisper响应体:", resp.text)

#         if resp.status_code != 200:
#             raise HTTPException(status_code=resp.status_code, detail=resp.text)

#         recognized_text = resp.json().get("text", "")
#         print("[🎤 Whisper识别结果]:", recognized_text)

#         return text_parse(TextInput(text=recognized_text))

#     except Exception as e:
#         print("[❌ Whisper处理异常]:", e)
#         traceback.print_exc()
#         raise HTTPException(status_code=400, detail=str(e))

from fastapi import UploadFile, File, HTTPException
from pydantic import BaseModel
import tempfile
import subprocess
import requests
import traceback
import magic   # python-magic
# 你的其余 import（nevo_df、vector_match_food、text_parse、TextInput 等）保持不变

SUPPORTED_MIME_MAP = {
    "audio/webm": "webm",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "video/mp4": "mp4",            # 识别并单独处理
    "audio/aac": "m4a",
    "audio/wav": "wav",
    "audio/ogg": "ogg",
    "audio/oga": "ogg",
    "audio/mpga": "mp3",
    "audio/flac": "flac"
}

@app.post("/api/audio")
async def recognize_from_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        print("📦 文件大小:", len(audio_bytes), "bytes")

        if len(audio_bytes) < 500:
            raise HTTPException(status_code=400, detail="音频太短或无效")
        if len(audio_bytes) > 2 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="音频文件过大（建议小于2MB）")

        # 1️⃣ magic 检测真实 MIME 类型
        mime = magic.Magic(mime=True)
        real_mime = mime.from_buffer(audio_bytes[:2048])
        print("📦 magic 检测 MIME 类型:", real_mime)

        # 2️⃣ video/mp4 -> 转 WAV；其余保持原来逻辑
        if real_mime == "video/mp4":
            # 保存临时 mp4
            tmp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_mp4.write(audio_bytes)
            tmp_mp4.close()

            # 转码为单声道 16 kHz WAV
            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_wav.close()

            print("🔄 ffmpeg 转码 video/mp4 → wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp4.name, "-vn",
                 "-ar", "16000", "-ac", "1", tmp_wav.name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            with open(tmp_wav.name, "rb") as f:
                audio_bytes = f.read()
            filename = "converted.wav"
            whisper_mime = "audio/wav"
        else:
            # 其余格式沿用原来逻辑
            if real_mime in SUPPORTED_MIME_MAP:
                ext = SUPPORTED_MIME_MAP[real_mime]
                filename = f"audio.{ext}"
                whisper_mime = f"audio/{ext}"
            else:
                print(f"⚠️ 未知 MIME 类型 {real_mime}，强制设为 webm")
                filename = "audio.webm"
                whisper_mime = "audio/webm"

        # 3️⃣ 调用 OpenAI Whisper
        openai_api_key = "sk-你的KEY"
        resp = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            files={"file": (filename, audio_bytes, whisper_mime)},
            data={"model": "whisper-1", "language": "en"},
            timeout=15
        )

        print("📡 Whisper响应码:", resp.status_code)
        print("📡 Whisper响应体:", resp.text)

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        recognized_text = resp.json().get("text", "")
        print("[🎤 Whisper识别结果]:", recognized_text)

        return text_parse(TextInput(text=recognized_text))

    except Exception as e:
        print("[❌ Whisper处理异常]:", e)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

# ===================== 10) New: /api/confirm (User-edited Items) =====================
class FoodItem(BaseModel):
    foodName: str
    quantity: float

@app.post("/api/confirm")
def confirm_food_items(items: List[FoodItem]):
    """
    1) front-end sends final list of {foodName, quantity}
    2) we do NEVO fuzzy match + nutrition => return
    3) if mismatch => NEVO_matched=null
    """
    output = []
    for it in items:
        fName = it.foodName.strip().lower()
        qty = it.quantity
        matched = vector_match_food(fName)
        if matched is None:
            output.append({
                "foodName": fName,
                "quantity": qty,
                "NEVO_matched": None,
                "error": "No semantic match in NEVO"
            })
            continue

        matched_row = nevo_df[nevo_df["CleanName"] == matched]
        if matched_row.empty:
            output.append({
                "foodName": fName,
                "quantity": qty,
                "NEVO_matched": matched,
                "error": "Row not found in NEVO"
            })
            continue

        row = matched_row.iloc[0]
        info = get_nutrition_info(row, qty)
        output.append({
            "foodName": fName,
            "quantity": qty,
            "NEVO_matched": row["Food name"],
            **info
        })

    # return {"foodItems": output}
    return { "parsedItems": output }


# ===================== Original root endpoint =====================
@app.get("/")
def root():
    return {"message": "Extended NEVO backend running with text/image/audio & confirm."}


# ===================== MAIN LAUNCH =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
