from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "saudi_hotel_sentiment.joblib"

st.set_page_config(page_title="SaudiStay", page_icon="🏨", layout="wide")
st.markdown("""
<style>
.stApp { direction: rtl; text-align: right; }
[data-testid="stSidebar"] { direction: rtl; text-align: right; }
h1, h2, h3, p, label { text-align: right; }
.hero { padding: 1.2rem 1.5rem; border-radius: 18px;
background:linear-gradient(115deg,#0c302d,#176b59);color:white; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>SaudiStay</h1><p>تحليل آراء النزلاء عن الفنادق في السعودية</p></div>', unsafe_allow_html=True)

@st.cache_resource
def get_bundle():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

bundle = get_bundle()
if bundle is None:
    st.warning("درّبي النموذج أولًا بالأوامر الموجودة في README، ثم ارفعي ملف النموذج إلى GitHub.")

tab1, tab2, tab3 = st.tabs(["لوحة التحليل", "تحليل تعليق", "عن المشروع"])
with tab1:
    st.subheader("تحليل ملف تقييمات")
    uploaded = st.file_uploader("ارفعي ملف CSV أو Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            df = pd.read_excel(uploaded) if uploaded.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(uploaded, encoding="utf-8-sig")
            cols = list(df.columns)
            c1, c2, c3 = st.columns(3)
            with c1:
                review_col = st.selectbox("عمود نص المراجعة", cols, index=cols.index(next((c for c in cols if str(c).lower() in {"review", "review_text", "text"}), cols[0])))
            with c2:
                loc_col = st.selectbox("عمود موقع الفندق", cols, index=cols.index(next((c for c in cols if str(c).lower() in {"location", "city", "hotel_city", "country"}), cols[0])))
            with c3:
                hotel_options = ["بدون"] + cols
                hotel_col = st.selectbox("عمود اسم الفندق", hotel_options, index=next((i + 1 for i, c in enumerate(cols) if str(c).lower() in {"hotel_name", "hotel"}), 0))
            cities = ["الرياض", "riyadh", "جدة", "jeddah", "مكة", "مكه", "makkah", "mecca",
                      "المدينة", "madinah", "الدمام", "dammam", "الخبر", "khobar", "الطائف",
                      "taif", "أبها", "ابها", "abha", "تبوك", "tabuk", "ينبع", "yanbu",
                      "جازان", "jazan", "حائل", "hail", "الأحساء", "الاحساء", "نجران", "najran"]
            pattern = "|".join(cities)
            mask = df[loc_col].fillna("").astype(str).str.contains(pattern, case=False, regex=True)
            data = df.loc[mask].copy()
            data[review_col] = data[review_col].fillna("").astype(str)
            st.info(f"وجدت {len(data):,} مراجعة في مواقع سعودية من أصل {len(df):,}. راجعي المواقع قبل اعتماد النتائج.")
            if not data.empty:
                locs = ["كل المواقع"] + sorted(data[loc_col].dropna().astype(str).unique())
                selected = st.selectbox("المدينة أو الموقع", locs)
                if selected != "كل المواقع":
                    data = data[data[loc_col].astype(str) == selected]
                if bundle:
                    data["المشاعر المتوقعة"] = bundle["pipeline"].predict(data[review_col])
                    data["ثقة النموذج"] = bundle["pipeline"].predict_proba(data[review_col]).max(axis=1)
                else:
                    label_col = next((c for c in cols if str(c).lower() in {"polarity_score", "sentiment", "label"}), None)
                    if label_col:
                        labels = {-1: "سلبي", 0: "محايد", 1: "إيجابي", "-1": "سلبي", "0": "محايد", "1": "إيجابي"}
                        data["المشاعر المتوقعة"] = data[label_col].map(labels).fillna(data[label_col].astype(str))
                if "المشاعر المتوقعة" in data:
                    counts = data["المشاعر المتوقعة"].value_counts().rename_axis("المشاعر").reset_index(name="العدد")
                    left, right = st.columns(2)
                    with left:
                        st.plotly_chart(px.pie(counts, names="المشاعر", values="العدد", hole=.5, title="توزيع المشاعر"), use_container_width=True)
                    with right:
                        if hotel_col != "بدون":
                            summary = data.groupby([hotel_col, "المشاعر المتوقعة"]).size().reset_index(name="العدد")
                            st.plotly_chart(px.bar(summary, x=hotel_col, y="العدد", color="المشاعر المتوقعة", barmode="group", title="حسب الفندق"), use_container_width=True)
                        else:
                            st.plotly_chart(px.bar(counts, x="المشاعر", y="العدد", title="عدد المراجعات"), use_container_width=True)
                    st.dataframe(data, use_container_width=True, hide_index=True)
                    st.download_button("تنزيل النتائج", data.to_csv(index=False).encode("utf-8-sig"), "saudistay_results.csv", "text/csv")
        except Exception as exc:
            st.error(f"تعذر قراءة الملف: {exc}")

with tab2:
    st.subheader("تحليل مراجعة واحدة")
    text = st.text_area("اكتبي مراجعة فندق", placeholder="الموقع ممتاز والموظفون متعاونون، لكن النظافة تحتاج تحسين.")
    if st.button("حلّل المراجعة", type="primary"):
        if not text.strip():
            st.warning("اكتبي مراجعة أولًا.")
        elif bundle is None:
            st.error("النموذج غير جاهز. اتبعي خطوات التدريب في README.")
        else:
            pipe = bundle["pipeline"]
            label = pipe.predict([text])[0]
            st.success(f"المشاعر المتوقعة: {label}")
            probs = pd.DataFrame({"المشاعر": pipe.classes_, "الاحتمال": pipe.predict_proba([text])[0]})
            st.dataframe(probs.sort_values("الاحتمال", ascending=False), hide_index=True, use_container_width=True)

with tab3:
    st.subheader("عن SaudiStay")
    st.write("مشروع NLP عربي يصنّف مراجعات الفنادق الموجودة في السعودية، ويعرض ملخصًا للمشاعر حسب المدينة والفندق.")
    st.write("النموذج الأولي يستخدم TF-IDF مع Logistic Regression، ويصنّف المشاعر إلى إيجابي أو محايد أو سلبي.")
    st.caption("موقع الفندق هو الذي يحدد انتماءه للسعودية؛ جنسية كاتب المراجعة لا تحدد ذلك.")
