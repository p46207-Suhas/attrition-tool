import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="ABC Ltd – Attrition Risk Tool", page_icon="📊")

logit = joblib.load("logit.pkl")
scaler = joblib.load("scaler.pkl")
lin = joblib.load("lin.pkl")

st.title("ABC Ltd – Employee Attrition Risk & Pay Benchmark Tool")
st.write(
    "Enter an employee's details to see their estimated attrition risk "
    "and an expected monthly income benchmark. This tool is a decision aid, "
    "not a replacement for managerial judgement."
)

st.subheader("Employee details")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)
    years_at_company = st.slider("Years at company", 0, 40, 3)
    job_satisfaction = st.selectbox(
        "Job satisfaction", [1, 2, 3, 4],
        format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x],
        index=2,
    )
    distance = st.slider("Distance from home (km)", 1, 30, 5)

with col2:
    overtime = st.radio("Works overtime?", ["No", "Yes"])
    total_working_years = st.slider("Total working years", 0, 40, 6)
    work_life_balance = st.selectbox(
        "Work-life balance", [1, 2, 3, 4],
        format_func=lambda x: {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}[x],
        index=2,
    )
    job_level = st.selectbox("Job level", [1, 2, 3, 4, 5])

if st.button("Predict", type="primary"):
    logit_features = ["Age", "YearsAtCompany", "JobSatisfaction", "DistanceFromHome",
                       "OverTime", "TotalWorkingYears", "WorkLifeBalance", "JobLevel"]

    row = pd.DataFrame(
        [[age, years_at_company, job_satisfaction, distance,
          1 if overtime == "Yes" else 0, total_working_years, work_life_balance, job_level]],
        columns=logit_features,
    )

    row_scaled = scaler.transform(row)
    risk = logit.predict_proba(row_scaled)[0][1]

    st.subheader("Results")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Attrition risk", f"{risk:.0%}")
        st.progress(float(risk))
        if risk > 0.5:
            st.error("High risk — consider a retention conversation.")
        elif risk > 0.3:
            st.warning("Moderate risk — worth monitoring.")
        else:
            st.success("Low risk.")

    with c2:
        lin_feats = ["Age", "TotalWorkingYears", "JobLevel", "YearsAtCompany"]
        income_row = row[lin_feats]
        income = lin.predict(income_row)[0]
        st.metric("Expected monthly income (benchmark)", f"{income:,.0f}")
        st.caption("Based on age, experience, job level, and tenure.")

    with st.expander("Why this prediction? (model explainability)"):
        coefs = pd.Series(logit.coef_[0], index=logit_features).sort_values()
        st.write(
            "These are the logistic regression coefficients. Positive values increase "
            "attrition risk; negative values decrease it. Larger magnitude = stronger effect."
        )
        st.bar_chart(coefs)

st.divider()
st.caption(
    "Built for an academic Business Analytics assignment using a fictional IBM HR dataset. "
    "Predictions are statistical estimates, not certainties."
)
