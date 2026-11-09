import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st


st.set_page_config(
    page_title="Water Quality AI Hub", page_icon="", layout="wide"
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hide Sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Elegant Light Blue Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 50%, #7dd3fc 100%);
    }

    /* Centered Content Container */
    .block-container {
        max-width: 920px;
        padding: 3rem 2rem;
        margin: auto;
    }

    /* Custom Clean Input Fields Styling */
    .stNumberInput input {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1.5px solid #bae6fd;
        color: #0369a1;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(3, 105, 161, 0.05);
        transition: all 0.3s ease;
    }

    .stNumberInput input:focus {
        border-color: #0284c7;
        box-shadow: 0 0 12px rgba(2, 132, 199, 0.25);
    }

    /* Sleek Modern Action Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        font-size: 18px;
        font-weight: 700;
        border-radius: 14px;
        height: 56px;
        border: none;
        box-shadow: 0 10px 25px rgba(2, 132, 199, 0.35);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #0c4a6e 100%);
        box-shadow: 0 14px 30px rgba(2, 132, 199, 0.5);
        transform: translateY(-2px);
    }

    /* Typography & Headings */
    h1 {
        color: #0c4a6e;
        font-weight: 800;
        text-align: center;
        letter-spacing: -0.5px;
    }
    
    p.subtitle {
        text-align: center;
        color: #0369a1;
        font-size: 17px;
        font-weight: 500;
        margin-bottom: 35px;
    }

    h3 {
        color: #0c4a6e;
        font-weight: 700;
        letter-spacing: -0.3px;
    }

    /* Professional Result Cards */
    .success-box {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 6px solid #16a34a;
        padding: 22px 28px;
        border-radius: 14px;
        color: #14532d;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 10px 25px rgba(22, 163, 74, 0.15);
    }

    .error-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 6px solid #dc2626;
        padding: 22px 28px;
        border-radius: 14px;
        color: #7f1d1d;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 10px 25px rgba(220, 38, 38, 0.15);
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_and_train_model():
  df = pd.read_csv("water_potability.csv")
  df_water = df.fillna(df.mean())

  col_list = list(df_water.columns)[0:-1]
  x = df_water[col_list]
  y = df_water["Potability"]

  x_train, x_test, y_train, y_test = train_test_split(
      x, y, test_size=0.2, random_state=42
  )

  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)

  model = RandomForestClassifier(random_state=42, n_estimators=100)
  model.fit(x_train_scaled, y_train)

  joblib.dump((model, scaler), "water_model.pkl")
  return model, scaler


try:
  model, scaler = joblib.load("water_model.pkl")
except:
  model, scaler = load_and_train_model()

# Header Section
st.title(" Water Potability Intelligence Hub")
st.markdown(
    '<p class="subtitle">Real-Time Intelligent Analysis for Safe & Potable Drinking Water</p>',
    unsafe_allow_html=True,
)

st.write("")

# Input Parameters Section
st.subheader("🧪 Enter Water Quality Parameters")
st.write("")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
  ph = st.number_input("pH Level", value=7.01, format="%.2f")
  hardness = st.number_input("Hardness", value=180.00, format="%.2f")
  solids = st.number_input("Solids (TDS)", value=18000.00, format="%.2f")

with col2:
  chloramines = st.number_input("Chloramines", value=7.00, format="%.2f")
  sulfate = st.number_input("Sulfate", value=330.00, format="%.2f")
  conductivity = st.number_input("Conductivity", value=400.00, format="%.2f")

with col3:
  organic_carbon = st.number_input("Organic Carbon", value=10.00, format="%.2f")
  trihalomethanes = st.number_input(
      "Trihalomethanes", value=60.00, format="%.2f"
  )
  turbidity = st.number_input("Turbidity", value=3.50, format="%.2f")

st.write("")
st.write("")

# Centered Action Button Container
b_col1, b_col2, b_col3 = st.columns([1, 2, 1])
with b_col2:
  predict_button = st.button("🔍 Predict Water Safety")

if predict_button:
  with st.spinner("✨ Running AI diagnostics on water parameters..."):
    features = [[
        ph,
        hardness,
        solids,
        chloramines,
        sulfate,
        conductivity,
        organic_carbon,
        trihalomethanes,
        turbidity,
    ]]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

  st.write("")
  if prediction[0] == 1:
    st.markdown(
        """
        <div class="success-box">
             Water is <b>Potable</b> and completely safe for drinking.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.balloons()
  else:
    st.markdown(
        """
        <div class="error-box">
            ⚠️ Warning! Water is <b>Not Potable</b> and unsafe for human consumption.
        </div>
        """,
        unsafe_allow_html=True,
    )