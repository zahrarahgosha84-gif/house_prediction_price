<div align="center">

![header](https://capsule-render.vercel.app/api?type=waving&color=0:6C63FF,100:00C9A7&height=200&section=header&text=Tehran%20House%20Price%20Prediction&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Predicting%20real%20estate%20prices%20with%20classic%20ML%20%2B%20Deep%20Learning&descAlignY=58&descSize=16)


[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=1000&color=6C63FF&center=true&vCenter=true&width=600&lines=Trained+8+classical+ML+models+with+GridSearchCV;Built+2+Neural+Nets+from+scratch+in+PyTorch;Tracked+everything+with+Weights+%26+Biases;Shipped+a+bilingual+Streamlit+app+%F0%9F%9A%80)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)

</div>

---

## What this is


A house price prediction model for Tehran, built on real Divar listing data (area, rooms, amenities, neighborhood). It started as a teaching notebook with a handful of classic ML models; I took it further — found and fixed a nasty bug hiding in the raw data, wrote two neural nets from scratch in PyTorch, tracked training with Weights & Biases, and shipped a bilingual web app that actually works.
<img src="assets/house_animation.gif" width="300">

## Results at a glance

| Rank | Model | R² | RMSE |
|:---:|---|:---:|:---:|
| 🥇 | XGBoost | 0.906 | 0.332 |
| 🥈 | Ridge Regression | 0.881 | 0.373 |
| 🥉 | Neural Net (128-64-32 + Dropout) | 0.865 | 0.398 |

10 models compared in total, from plain linear regression up to a custom neural net. Full breakdown is in the notebook.

<div align="center">
<img src="assets/model_comparison.png" width="750">
</div>

## Exploring the data in 3D

Area, number of rooms, and price, all in one plot — color shows price.

<div align="center">
<img src="assets/rotating_3d_scatter.gif" width="480">
</div>

Want to actually drag and rotate it yourself? Download [`assets/interactive_3d_chart.html`](assets/interactive_3d_chart.html) and open it in your browser (GitHub can't run interactive charts inline, but this file is fully interactive locally).

## The part that made this interesting   

I spent more time debugging than training models. Early on, plain linear regression gave an R² of basically zero — no signal at all. After a few hours of digging, I found 4 rows where the `Area` value had accidentally been copied from the `Price` column (a several-billion-square-meter "house"!). Dropping those 4 rows took R² from zero to 0.88 instantly.

Then the neural net had its own drama: the second version trained for 3000 epochs and quietly overfit after epoch 1800 — test loss jumped from 0.18 to over 10. Fixed it with a manual early-stopping loop that snapshots the model weights every time test loss improves.

## Neural net architecture (best version)

```
Input (103 features)
      │
   Linear(103 → 128) → ReLU → Dropout(0.3)
      │
   Linear(128 → 64) → ReLU → Dropout(0.3)
      │
   Linear(64 → 32) → ReLU
      │
   Linear(32 → 1)
      │
  Predicted log(Price)
```

## Web app demo

A bilingual (Persian/English) Streamlit app — enter the area, rooms, amenities and neighborhood, get a predicted price back instantly.

<div align="center">
<img src="assets/webapp_screenshot.png" width="500">
</div>

## Project structure

```
📦 house-price-prediction
├── data/
│   └── tehranhouses.csv         # raw dataset
├── sample.ipynb                 # full analysis, modeling, and training
├── app.py                       # Streamlit web app
├── house_price_model.pth        # trained neural net weights
├── scaler.pkl                   # StandardScaler used in preprocessing
├── feature_columns.pkl          # exact feature order the model expects
├── requirements.txt
├── assets/
│   ├── house_animation.gif
│   ├── model_comparison.png
│   ├── rotating_3d_scatter.gif
│   ├── interactive_3d_chart.html
│   └── webapp_screenshot.png
└── README.md                     # you're reading it
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `localhost:8501` and fill in the house details.

## Training logs (WandB)

Live train/test loss curves for both neural net versions:
🔗 [wandb.ai/zahrarahgoshaa-prohect/house-price-prediction](https://wandb.ai/zahrarahgoshaa-prohect/house-price-prediction)

---

<div align="center">
<img src="assets/Belovedhome.gif" width="200">
![footer](https://capsule-render.vercel.app/api?type=waving&color=0:00C9A7,100:6C63FF&height=100&section=footer)

</div>
</div>
