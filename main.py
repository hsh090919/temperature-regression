import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="기온 예측기 - 선형회귀",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 모델")
st.write(
    "과거 연평균 기온으로 선형회귀 모델을 학습하고 "
    "최근 20년(2006~2025)의 기온을 얼마나 잘 예측하는지 비교합니다."
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"

try:
    df = pd.read_csv(DATA_URL)

except Exception as e:
    st.error("데이터를 불러오는 데 문제가 발생했습니다.")
    st.stop()


# --------------------------------------------------
# 데이터 전처리
# --------------------------------------------------

df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

df = df.dropna(subset=["날짜", "평균기온"]).copy()

df["연도"] = df["날짜"].dt.year


# --------------------------------------------------
# 연평균 기온 계산
# --------------------------------------------------

annual = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index()
)

annual.columns = ["연도", "연평균기온"]

annual = annual.sort_values("연도").reset_index(drop=True)


# --------------------------------------------------
# 필요한 기간만 사용
# --------------------------------------------------

annual = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2025)
].copy()


# --------------------------------------------------
# 회귀 모델 함수
# --------------------------------------------------

def make_model(data):
    X = data[["연도"]]
    y = data["연평균기온"]

    model = LinearRegression()
    model.fit(X, y)

    return model


def evaluate_model(model, test_data):
    X_test = test_data[["연도"]]
    y_test = test_data["연평균기온"]

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, r2, y_pred


# --------------------------------------------------
# 1. 전체 데이터 회귀
# --------------------------------------------------

full_model = make_model(annual)

annual["전체회귀예측"] = full_model.predict(
    annual[["연도"]]
)

full_slope = full_model.coef_[0]
full_slope_100 = full_slope * 100
full_intercept = full_model.intercept_


# --------------------------------------------------
# 2. 훈련 / 테스트 데이터 분리
# --------------------------------------------------

train_50 = annual[
    (annual["연도"] >= 1956) &
    (annual["연도"] <= 2005)
].copy()

train_100 = annual[
    (annual["연도"] >= 1906) &
    (annual["연도"] <= 2005)
].copy()

test = annual[
    (annual["연도"] >= 2006) &
    (annual["연도"] <= 2025)
].copy()


# --------------------------------------------------
# 3. 50년 학습 모델
# --------------------------------------------------

model_50 = make_model(train_50)

slope_50 = model_50.coef_[0]
slope_50_100 = slope_50 * 100

mae_50, mse_50, r2_50, pred_50 = evaluate_model(
    model_50,
    test
)


# --------------------------------------------------
# 4. 100년 학습 모델
# --------------------------------------------------

model_100 = make_model(train_100)

slope_100 = model_100.coef_[0]
slope_100_100 = slope_100 * 100

mae_100, mse_100, r2_100, pred_100 = evaluate_model(
    model_100,
    test
)


# --------------------------------------------------
# 전체 데이터 자체 평가
# --------------------------------------------------

full_mae = mean_absolute_error(
    annual["평균기온"],
    annual["전체회귀예측"]
)

full_mse = mean_squared_error(
    annual["평균기온"],
    annual["전체회귀예측"]
)

full_r2 = r2_score(
    annual["평균기온"],
    annual["전체회귀예측"]
)


# --------------------------------------------------
# 5. 제목
# --------------------------------------------------

st.header("1. 전체 기간의 선형회귀")


st.write(
    f"1906~2025년 전체 연평균 기온을 이용해 하나의 회귀선을 만들었습니다."
)

st.latex(
    f"y = {full_slope:.4f}x + ({full_intercept:.2f})"
)

st.metric(
    "전체 기간 회귀선의 기울기",
    f"100년에 {full_slope_100:.2f} ℃"
)


# --------------------------------------------------
# 전체 기간 그래프
# --------------------------------------------------

fig_full = go.Figure()

fig_full.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(size=6)
    )
)

fig_full.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["전체회귀예측"],
        mode="lines",
        name="전체 데이터 회귀선",
        line=dict(width=3)
    )
)

fig_full.update_layout(
    title="1906~2025 연평균 기온과 전체 데이터 회귀선",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_full,
    use_container_width=True
)


# --------------------------------------------------
# 전체 데이터 평가
# --------------------------------------------------

st.subheader("전체 데이터에 대한 회귀 성능")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"{full_mae:.3f} ℃"
    )

with col2:
    st.metric(
        "MSE",
        f"{full_mse:.3f}"
    )

with col3:
    st.metric(
        "R²",
        f"{full_r2:.3f}"
    )


st.info(
    "주의: 전체 데이터 평가는 같은 데이터로 학습하고 평가한 결과입니다. "
    "따라서 실제 미래 예측 성능을 확인하려면 아래의 테스트 데이터 평가를 보는 것이 더 적절합니다."
)


# --------------------------------------------------
# 6. 50년 vs 100년 회귀선
# --------------------------------------------------

st.header("2. 50년 학습과 100년 학습 비교")

st.write(
    "두 모델 모두 2006~2025년을 테스트 데이터로 사용합니다."
)

fig_compare = go.Figure()


# 실제 데이터
fig_compare.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(size=6)
    )
)


# 50년 모델
x_50 = np.arange(1956, 2026)

y_50 = model_50.predict(
    pd.DataFrame({"연도": x_50})
)

fig_compare.add_trace(
    go.Scatter(
        x=x_50,
        y=y_50,
        mode="lines",
        name="1956~2005 학습 회귀선",
        line=dict(width=3)
    )
)


# 100년 모델
x_100 = np.arange(1906, 2026)

y_100 = model_100.predict(
    pd.DataFrame({"연도": x_100})
)

fig_compare.add_trace(
    go.Scatter(
        x=x_100,
        y=y_100,
        mode="lines",
        name="1906~2005 학습 회귀선",
        line=dict(width=3)
    )
)


# 테스트 구간 표시
fig_compare.add_vrect(
    x0=2006,
    x1=2025,
    fillcolor="gray",
    opacity=0.12,
    line_width=0,
    annotation_text="테스트 데이터 2006~2025",
    annotation_position="top left"
)


fig_compare.update_layout(
    title="50년 학습 회귀선과 100년 학습 회귀선 비교",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_compare,
    use_container_width=True
)


# --------------------------------------------------
# 7. 기울기 비교
# --------------------------------------------------

st.subheader("회귀선의 기울기 비교")

slope_table = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "훈련기간": [
        "1956~2005",
        "1906~2005"
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ],
    "100년당 기온 변화 (℃)": [
        slope_50_100,
        slope_100_100
    ]
})

st.dataframe(
    slope_table.style.format({
        "기울기 (℃/년)": "{:.5f}",
        "100년당 기온 변화 (℃)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 8. 테스트 데이터 예측 성능
# --------------------------------------------------

st.header("3. 최근 20년(2006~2025) 예측 성능 평가")

st.write(
    "1956~2005 또는 1906~2005년만 이용해 회귀선을 학습한 뒤, "
    "학습에 사용하지 않은 2006~2025년을 예측합니다."
)


performance = pd.DataFrame({
    "모델": [
        "50년 학습",
        "100년 학습"
    ],
    "훈련 기간": [
        "1956~2005",
        "1906~2005"
    ],
    "테스트 기간": [
        "2006~2025",
        "2006~2025"
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    performance.style.format({
        "MAE (℃)": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 9. 성능 비교 해석
# --------------------------------------------------

st.subheader("성능을 어떻게 비교할까?")

if mae_50 < mae_100:
    mae_result = "50년 학습 모델의 MAE가 더 작아 최근 20년의 평균적인 예측 오차가 더 작습니다."
elif mae_50 > mae_100:
    mae_result = "100년 학습 모델의 MAE가 더 작아 최근 20년의 평균적인 예측 오차가 더 작습니다."
else:
    mae_result = "두 모델의 MAE가 같습니다."


if mse_50 < mse_100:
    mse_result = "50년 학습 모델의 MSE가 더 작아 큰 오차까지 고려했을 때도 더 안정적입니다."
elif mse_50 > mse_100:
    mse_result = "100년 학습 모델의 MSE가 더 작아 큰 오차까지 고려했을 때 더 안정적입니다."
else:
    mse_result = "두 모델의 MSE가 같습니다."


if r2_50 > r2_100:
    r2_result = "50년 학습 모델의 R²가 더 높아 최근 20년의 기온 변동을 더 잘 설명합니다."
elif r2_50 < r2_100:
    r2_result = "100년 학습 모델의 R²가 더 높아 최근 20년의 기온 변동을 더 잘 설명합니다."
else:
    r2_result = "두 모델의 R²가 같습니다."


st.write("**MAE:** " + mae_result)
st.write("**MSE:** " + mse_result)
st.write("**R²:** " + r2_result)


# --------------------------------------------------
# 10. 테스트 구간 실제값 vs 예측값
# --------------------------------------------------

st.header("4. 실제 기온과 예측 기온 비교")

test_result = test.copy()

test_result["50년 예측"] = pred_50
test_result["100년 예측"] = pred_100

fig_test = go.Figure()


fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["평균기온"],
        mode="lines+markers",
        name="실제 기온",
        line=dict(width=3)
    )
)

fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["50년 예측"],
        mode="lines",
        name="50년 학습 예측",
        line=dict(dash="dash", width=3)
    )
)

fig_test.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["100년 예측"],
        mode="lines",
        name="100년 학습 예측",
        line=dict(dash="dot", width=3)
    )
)


fig_test.update_layout(
    title="2006~2025 실제 기온과 두 회귀모델의 예측 비교",
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_test,
    use_container_width=True
)


# --------------------------------------------------
# 11. 테스트 데이터 상세표
# --------------------------------------------------

st.subheader("2006~2025년 실제값과 예측값")

display_test = test_result[
    [
        "연도",
        "평균기온",
        "50년 예측",
        "100년 예측"
    ]
].copy()

display_test["50년 오차"] = (
    display_test["평균기온"] -
    display_test["50년 예측"]
)

display_test["100년 오차"] = (
    display_test["평균기온"] -
    display_test["100년 예측"]
)

display_test.columns = [
    "연도",
    "실제 연평균기온",
    "50년 학습 예측",
    "100년 학습 예측",
    "50년 오차",
    "100년 오차"
]

st.dataframe(
    display_test.style.format({
        "실제 연평균기온": "{:.2f}",
        "50년 학습 예측": "{:.2f}",
        "100년 학습 예측": "{:.2f}",
        "50년 오차": "{:.2f}",
        "100년 오차": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 12. 최종 정리
# --------------------------------------------------

st.header("5. 최종 비교")

st.success(
    f"""
**50년 학습 모델**
- 학습: 1956~2005년
- 테스트: 2006~2025년
- 100년당 기온 변화: {slope_50_100:.2f}℃
- 테스트 MAE: {mae_50:.3f}℃
- 테스트 MSE: {mse_50:.3f}
- 테스트 R²: {r2_50:.3f}

**100년 학습 모델**
- 학습: 1906~2005년
- 테스트: 2006~2025년
- 100년당 기온 변화: {slope_100_100:.2f}℃
- 테스트 MAE: {mae_100:.3f}℃
- 테스트 MSE: {mse_100:.3f}
- 테스트 R²: {r2_100:.3f}
"""
)


st.caption(
    "MAE와 MSE는 작을수록 좋고, R²는 일반적으로 1에 가까울수록 좋습니다."
)
