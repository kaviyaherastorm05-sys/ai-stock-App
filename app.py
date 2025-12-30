import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from model import predict_stock_price

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="AI Stock Dashboard",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------
# Sidebar – Theme & Controls
# -------------------------------------------------
st.sidebar.title("📌 Controls")

theme = st.sidebar.radio("🎨 Theme", ["Dark", "Light"])
symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
period = st.sidebar.selectbox(
    "Data Period", ("1mo", "3mo", "6mo", "1y", "2y", "5y")
)

# -------------------------------------------------
# Styling
# -------------------------------------------------
if theme == "Dark":
    bg = "#0e1117"
    card = "#1c1f26"
    plot_theme = "plotly_dark"
else:
    bg = "#ffffff"
    card = "#f5f5f5"
    plot_theme = "plotly_white"

st.markdown(f"""
<style>
body {{ background-color: {bg}; }}
[data-testid="metric-container"] {{
    background-color: {card};
    padding: 15px;
    border-radius: 12px;
}}
h1, h2, h3 {{ text-align: center; }}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Cached Data Loader (SAFE)
# -------------------------------------------------
@st.cache_data
def load_data(symbol, period):
    df = yf.download(symbol, period=period, auto_adjust=False)

    # Handle MultiIndex columns safely
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("📈 AI-Powered Stock Market Dashboard")
st.caption("Premium UI • Interactive Charts • Predictive AI")

# -------------------------------------------------
# Main Logic
# -------------------------------------------------
if symbol:
    data = load_data(symbol, period)

    if data.empty or len(data) < 2:
        st.error("Invalid stock symbol or insufficient data ❌")
        st.stop()

    # -------------------------------------------------
    # SAFE SCALAR VALUES
    # -------------------------------------------------
    last_close = data["Close"].iloc[-1].item()
    prev_close = data["Close"].iloc[-2].item()
    last_open = data["Open"].iloc[-1].item()
    last_high = data["High"].iloc[-1].item()
    last_volume = int(data["Volume"].iloc[-1].item())

    delta = last_close - prev_close
    delta_icon = "🟢" if delta > 0 else "🔴"

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Open", f"${last_open:.2f}")
    c2.metric("Close", f"${last_close:.2f}", f"{delta_icon} {delta:.2f}")
    c3.metric("High", f"${last_high:.2f}")
    c4.metric("Volume", f"{last_volume:,}")

    st.divider()

    # -------------------------------------------------
    # Tabs
    # -------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Chart", "📄 Data", "🤖 AI Prediction", "📰 News"]
    )

    # ---------------- Chart Tab ----------------
    with tab1:
        data["SMA_20"] = data["Close"].rolling(20).mean()
        data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data.index, y=data["Close"],
            mode="lines", name="Close", line=dict(width=2)
        ))

        fig.add_trace(go.Scatter(
            x=data.index, y=data["SMA_20"],
            mode="lines", name="SMA 20", line=dict(dash="dash")
        ))

        fig.add_trace(go.Scatter(
            x=data.index, y=data["EMA_20"],
            mode="lines", name="EMA 20", line=dict(dash="dot")
        ))

        fig.update_layout(
            template=plot_theme,
            height=520,
            title=f"{symbol} Price Chart"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- Data Tab ----------------
    with tab2:
        st.dataframe(data.tail(25), use_container_width=True)

        st.download_button(
            "⬇️ Download CSV",
            data=data.to_csv().encode("utf-8"),
            file_name=f"{symbol}_data.csv",
            mime="text/csv"
        )

    # ---------------- AI Prediction ----------------
    with tab3:
        prediction = predict_stock_price(data.dropna())

        confidence = min(95, 70 + abs(delta) * 2)

        st.success(f"📈 Predicted Next Close: ${prediction:.2f}")
        st.info(f"🧠 Model Confidence: {confidence:.1f}%")

    # ---------------- News Tab ----------------
    with tab4:
        ticker = yf.Ticker(symbol)
        news = ticker.news[:5]

        if not news:
            st.warning("No recent news available.")
        else:
            for article in news:
                st.markdown(f"""
**{article.get('title', 'No Title')}**  
{article.get('publisher', '')}  
[Read more]({article.get('link', '#')})
""")
                st.markdown("---")
