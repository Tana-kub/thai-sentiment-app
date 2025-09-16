import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from rapidfuzz import fuzz, process  # ✅ ใช้ rapidfuzz แทน fuzzywuzzy (เร็วกว่า)

# st.title("🍲 Thai Food Review Sentiment Analysis (Dataset + Model Fallback + Fuzzy Match)")
st.title("🍲 Thai Food Review รีวิวความรู้สึกที่พวกขี้เล้งได้แดกอาหาร")
# st.write("ระบบจะตรวจสอบ dataset ก่อน ถ้าไม่พบจะใช้โมเดล FlukeTJ/distilbert-base-thai-sentiment")
st.write("จงทดสอบระบบให้ทีครับ พวกขี้เล้งทั้งหลาย")
# โหลดโมเดล fallback
MODEL_NAME = "FlukeTJ/distilbert-base-thai-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels = ["negative(ความรู้สึกเป็นเกย์ลบ) 😞", "neutral(ความรู้สึกเป็นเกย์ปานกลาง) 😐", "positive(ความรู้สึกเป็นเกย์บวก) 😊"]
label2id = {"negative": 0, "neutral": 1, "positive": 2}

# โหลด dataset
with open("thai_food_reviews.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# normalize text
def normalize(text: str):
    return text.strip().lower().replace(" ", "")

# แปลง dataset เป็น dict
dataset_dict = {normalize(item["text"]): label2id[item["label"]] for item in dataset}
sorted_keywords = sorted(dataset_dict.keys(), key=len, reverse=True)  # keyword ยาวก่อน

# ฟังก์ชันเช็ค keyword + fuzzy match
def find_keyword_match(input_text, threshold=80):
    norm_input = normalize(input_text)

    # 1) เช็ค exact match (substring)
    for kw in sorted_keywords:
        if kw in norm_input:
            return dataset_dict[kw], kw, 100

    # 2) ใช้ fuzzy match
    match, score, _ = process.extractOne(norm_input, dataset_dict.keys(), scorer=fuzz.ratio)
    if score >= threshold:
        return dataset_dict[match], match, score

    return None, None, None

# ฟังก์ชันหา example
def find_similar_examples(input_text, top_k=5):
    norm_input = normalize(input_text)
    matches = process.extract(norm_input, dataset_dict.keys(), scorer=fuzz.partial_ratio, limit=top_k)
    return [m[0] for m in matches]

# Input จากผู้ใช้
user_input = st.text_input("ป้อนรีวิวอาหารของคุณ:")

if user_input:
    # 1) ตรวจ dataset ก่อน
    pred_label, matched_kw, score = find_keyword_match(user_input)

    if pred_label is not None:
        confidence = 100.0 if score == 100 else score
        # st.write(f"✅ พบใน dataset (match: '{matched_kw}', similarity: {score:.2f}%)")
        st.write(f"✅ พบใน dataset 
    else:
        # 2) ถ้าไม่เจอ → ใช้โมเดล FlukeTJ/distilbert-base-thai-sentiment
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item() * 100
        st.write("⚠️ ไม่พบ keyword → ใช้โมเดล FlukeTJ/distilbert-base-thai-sentiment วิเคราะห์")

    # แสดงผล
    st.write(f"**ผลลัพธ์:** {labels[pred_label]}")
    if confidence < 60:
        st.warning(f"⚠️ ความมั่นใจต่ำ ({confidence:.2f}%) → ผลลัพธ์อาจไม่แม่นยำ")
    else:
        st.write(f"**เปอร์เซ็นต์ความมั่นใจของระบบ:** {confidence:.2f}%")

    # ตัวอย่างรีวิวใกล้เคียง
    # st.write("🔎 ตัวอย่างรีวิวจาก dataset ที่คล้ายกัน:")
    # examples = find_similar_examples(user_input)
    # for e in examples:
    #     st.write(f"- {e}")

