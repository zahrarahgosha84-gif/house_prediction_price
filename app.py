import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

class HousePriceNN_v2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

feature_columns = joblib.load('feature_columns.pkl')
scaler = joblib.load('scaler.pkl')

model = HousePriceNN_v2(len(feature_columns))
model.load_state_dict(torch.load('house_price_model.pth', map_location='cpu'))
model.eval()

texts = {
    "fa": {
        "lang_label": "زبان",
        "title": "پیش‌بینی قیمت خانه در تهران",
        "subtitle": "مشخصات خانه را وارد کنید:",
        "area": "متراژ (متر مربع)",
        "room": "تعداد اتاق",
        "parking": "پارکینگ دارد؟",
        "warehouse": "انباری دارد؟",
        "elevator": "آسانسور دارد؟",
        "address": "محله",
        "other": "سایر (خارج از لیست)",
        "yes": "بله",
        "no": "خیر",
        "button": "پیش‌بینی قیمت",
        "result": "قیمت پیش‌بینی‌شده: {:,.0f} تومان"
    },
    "en": {
        "lang_label": "Language",
        "title": "Tehran House Price Prediction",
        "subtitle": "Enter house details:",
        "area": "Area (sq meters)",
        "room": "Number of rooms",
        "parking": "Has parking?",
        "warehouse": "Has warehouse?",
        "elevator": "Has elevator?",
        "address": "Neighborhood",
        "other": "Other (not in list)",
        "yes": "Yes",
        "no": "No",
        "button": "Predict Price",
        "result": "Predicted price: {:,.0f} Toman"
    }
}

lang = st.selectbox("زبان / Language", ["فارسی", "English"])
t = texts["fa"] if lang == "فارسی" else texts["en"]

st.title(t["title"])
st.write(t["subtitle"])

area = st.number_input(t["area"], min_value=20, max_value=1000, value=80)
room = st.number_input(t["room"], min_value=0, max_value=6, value=2)
parking = st.selectbox(t["parking"], [t["yes"], t["no"]])
warehouse = st.selectbox(t["warehouse"], [t["yes"], t["no"]])
elevator = st.selectbox(t["elevator"], [t["yes"], t["no"]])

address_columns = [c for c in feature_columns if c.startswith('Address_')]
address_names = [c.replace('Address_', '') for c in address_columns]
address = st.selectbox(t["address"], sorted(address_names) + [t["other"]])

if st.button(t["button"]):
    input_dict = {col: 0 for col in feature_columns}
    input_dict['Area'] = area
    input_dict['Room'] = room
    input_dict['Parking'] = 1 if parking == t["yes"] else 0
    input_dict['Warehouse'] = 1 if warehouse == t["yes"] else 0
    input_dict['Elevator'] = 1 if elevator == t["yes"] else 0

    address_col = f"Address_{address}"
    if address_col in input_dict:
        input_dict[address_col] = 1

    input_df = pd.DataFrame([input_dict])[feature_columns]
    input_scaled = scaler.transform(input_df)
    input_tensor = torch.tensor(input_scaled.astype(np.float32))

    with torch.no_grad():
        pred_log = model(input_tensor).item()

    pred_price = np.expm1(pred_log)
    st.success(t["result"].format(pred_price))