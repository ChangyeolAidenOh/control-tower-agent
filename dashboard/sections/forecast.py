"""Forecast section: model comparison, interval calibration, fan charts.

Metric tables come from the verified snapshot (stage 3/4 projection);
fan charts come from the pre-built forecast extract. Rendering only.
Nested per-fold/per-horizon dicts are not dumped into tables; only
scalar summary metrics are shown (ST9-06).
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import labels

ASSETS = Path(__file__).resolve().parents[1] / "assets"

MODEL_NAMES = {
    "lgbm": "LightGBM (분위수 회귀)",
    "chronos2s": "Chronos-2 small (사전학습 모델, 참고)",
    "tsmixer_a1": "TSMixer A1 (점 예측)",
    "tsmixer_a2_r3": "TSMixer A2 (비용 가중, r=3)",
    "tsmixer_a2_r5": "TSMixer A2 (비용 가중, r=5)",
    "tsmixer_a3": "TSMixer A3 (분위수)",
    "tsmixer_a3_q50": "TSMixer A3 (중앙값)",
    "lgbm_q50": "LightGBM (중앙값)",
    "chronos2s_q50": "Chronos-2 small (중앙값, 참고)",
    "lgbm_lt3": "직접 누적 모델 (3일 수요를 직접 예측)",
    "lt3_como": "일별 예측 합성 — 완전 동조 가정",
    "lt3_indep": "일별 예측 합성 — 독립 가정",
}

METRIC_NAMES = {
    "wql": "WQL (분위수 손실)",
    "wape": "WAPE",
    "mase_macro": "MASE (시리즈 평균)",
    "mase_cells_undef": "MASE 미정의 셀",
    "nonzero_mape": "MAPE (판매일만, 참고)",
    "nonzero_mape_n": "MAPE 대상 수",
    "rows": "평가 행 수",
    "coverage90": "90% 커버리지 (실측)",
    "mean_width90": "평균 구간 폭",
    "mase_macro_fold_series_cells": "MASE (시리즈 평균)",
    "mase_undefined_cells": "MASE 미정의 셀",
    "nonzero_mape_rows": "MAPE 대상 수",
    "nonzero_mape_excluded_rows": "MAPE 제외 수 (무판매일)",
}

LT_NOTE = (
    "합성 방법 간 차이에는 추정 오차와 결합 가정이 섞여 있어, 수요의 실제 "
    "시간 의존성에 대한 증거로 해석하지 않습니다. 이 비교는 발주 입력 "
    "선택과 분리된 참고 분석입니다."
)


def fmt(v):
    if isinstance(v, bool):
        return "예" if v else "아니오"
    if isinstance(v, float):
        return f"{v:,.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def scalar_table(block: dict, title: str) -> None:
    """Render one row per model using scalar metrics only.

    Scalars nested under a "primary" sub-dict are promoted to the row,
    since the stage 3 daily block keeps its headline metrics there.
    """
    rows = []
    for model, metrics in block.items():
        if not isinstance(metrics, dict):
            continue
        row = {"모델": MODEL_NAMES.get(model, model)}
        flat = dict(metrics)
        primary = flat.get("primary")
        if isinstance(primary, dict):
            for pk, pv in primary.items():
                flat.setdefault(pk, pv)
        for mk, mv in flat.items():
            if isinstance(mv, (dict, list)):
                continue
            row[METRIC_NAMES.get(mk, mk)] = fmt(mv)
        if len(row) > 1:
            rows.append(row)
    if rows:
        st.markdown(f"**{title}**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def lt_table(methods: dict) -> None:
    rows = []
    for model, metrics in methods.items():
        pooled = metrics.get("pooled", {}) if isinstance(metrics, dict) else {}
        row = {"방법": MODEL_NAMES.get(model, model)}
        for mk, mv in pooled.items():
            if isinstance(mv, (dict, list)):
                continue
            row[METRIC_NAMES.get(mk, mk)] = fmt(mv)
        if len(row) > 1:
            rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render(snapshot: dict) -> None:
    st.header("예측 — 확률 예측의 품질과 형태")
    fs = snapshot["forecast_summary"]
    s3 = fs["stage3"]
    s4 = fs["stage4"]

    st.subheader("일 단위 예측 품질")
    st.caption(labels.WQL_CLAIM)
    scalar_table(s3.get("daily", {}), "주 평가 (평가 창 내, 전 지표 공통 평가 집합)")
    st.caption(labels.CHRONOS_FOOTNOTE)
    st.caption(labels.H_CURVE_WARNING)

    st.subheader("예측구간 보정 (적응형 보정 적용)")
    h2 = s4.get("h2_citation_primary_full_period", {})
    if h2:
        c1, c2, c3 = st.columns(3)
        c1.metric("커버리지 (보정 전)", fmt(h2.get("coverage_before")))
        c2.metric("커버리지 (보정 후)", fmt(h2.get("coverage_after")))
        c3.metric("평균 구간 폭 (보정 후)", fmt(h2.get("mean_width_after")))
        st.caption(
            "목표 90% 구간의 실측 커버리지입니다. 구간은 확률적 보증이 아니라 "
            "실측 커버리지와 함께 읽습니다."
        )
    scalar_table(s4.get("decile9", {}), "분위수 예측 비교 (9분위 공통 집합)")
    scalar_table(s4.get("point_main", {}), "점 예측 비교")
    st.caption(labels.BAND_RULE)

    st.subheader("리드타임 누적 수요 구간 (발주 의사결정 입력)")
    lt = s3.get("lead_time", {})
    methods = lt.get("methods", {})
    if methods:
        st.markdown("**3일 누적 수요 — 방법 비교 (전체 평가 집합 기준)**")
        lt_table(methods)
        st.caption(LT_NOTE)

    st.subheader("예측 경로 보기 (fan chart)")
    extract_path = ASSETS / "forecast_extract.json"
    if not extract_path.exists():
        st.info("예측 경로 표시 데이터가 아직 생성되지 않았습니다.")
        return
    with open(extract_path, "r", encoding="utf-8") as f:
        extract = json.load(f)
    entries = extract["series"]
    options = {
        f"폴드 {e['fold']} · {e['cat_id']} · {e['series_id']}": i
        for i, e in enumerate(entries)
    }
    choice = st.selectbox("대상 선택 (사전 고정된 기계 규칙으로 선정)", list(options))
    e = entries[options[choice]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=e["lgbm"]["t"] + e["lgbm"]["t"][::-1],
            y=e["lgbm"]["q95"] + e["lgbm"]["q05"][::-1],
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.2)",
            line=dict(width=0),
            name=labels.BAND_LABELS["lgbm"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=e["lgbm"]["t"], y=e["lgbm"]["q50"],
            mode="lines", name="LightGBM 중앙값",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=e["chronos2s"]["t"], y=e["chronos2s"]["q50"],
            mode="lines", line=dict(dash="dash"),
            name="Chronos-2 중앙값 (참고)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=e["history"]["t"], y=e["history"]["y"],
            mode="lines", line=dict(color="gray"),
            name="과거 실측 (28일)",
        )
    )
    if e["realized"]["t"]:
        fig.add_trace(
            go.Scatter(
                x=e["realized"]["t"], y=e["realized"]["y"],
                mode="markers", marker=dict(size=5, color="white"),
                name="실현 수요 (관측 구간만)",
            )
        )
    fig.add_vline(x=e["origin_idx"], line_dash="dot", line_color="gray")
    fig.update_layout(
        xaxis_title="일 (데이터 인덱스)",
        yaxis_title="판매량 (단위)",
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "예측 시점(세로 점선) 이후 28일의 분포 예측입니다. "
        + labels.BAND_RULE
        + " 실현 수요는 관측된 날짜에만 표시합니다."
    )
    st.caption(labels.CHRONOS_FOOTNOTE)
    with st.expander("표시 대상 선정 방식"):
        st.markdown(
            "표시할 시계열과 예측 시점은 결과 곡선을 보기 전에 고정한 기계적 "
            "규칙으로 뽑았습니다: 폴드 1·7에서, 카테고리마다 이름순 첫 묶음을 "
            "고르고, 그 묶음 안에서 일평균 판매량 기준 중앙에 해당하는 SKU를 "
            "선택하며, 예측 시점은 각 폴드의 첫 시점입니다. 잘 맞은 사례를 "
            "골라낸 것이 아닙니다. 규칙 원문은 배포 manifest에 기록되어 있습니다."
        )
