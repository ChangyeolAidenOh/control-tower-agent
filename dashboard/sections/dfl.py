"""DFL ablation section: policy comparison and hypothesis results.

All figures come from the verified snapshot (s5_agg projection and the
confirmatory block). Rendering only; no recomputation beyond unit
formatting (fractions shown as percent).
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import labels

POLICY_ORDER = ["b", "a1", "apto", "a4", "a4_sl", "l0_rule_ct", "a2", "a3", "a5"]

BASELINE_NAMES = {
    "a2": "A2 (비용 가중 분위수 예측 + 최적화)",
    "a3": "A3 (분위수 시나리오 SAA)",
    "a4": "A4 (운영형 기준선, 비용 선택)",
}


def fmt_cost(v: float) -> str:
    return f"{v:,.1f}"


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def render(snapshot: dict) -> None:
    st.header("DFL ablation — 정책 비교와 가설 결과")
    st.caption(labels.DFL_BADGE_RULE)

    dfl = snapshot["dfl"]
    policies = dfl["policies"]
    agent = snapshot["approval"]["agent_summary"]

    st.subheader("정책별 성과 (480 에피소드, r=5)")
    rows = []
    for key in POLICY_ORDER:
        p = policies.get(key)
        if p is None:
            continue
        rows.append(
            {
                "정책": p.get("display_name", key),
                "평균 에피소드 비용": p["mean_episode_cost"],
                "CSL (SKU-일)": p["pooled_sku_day_csl"],
                "CSL (에피소드 평균)": p["episode_mean_csl"],
            }
        )
    rows.append(
        {
            "정책": "파이프라인 (승인 흐름)",
            "평균 에피소드 비용": agent["mean_episode_cost"],
            "CSL (SKU-일)": agent["pooled_sku_day_csl"],
            "CSL (에피소드 평균)": agent["episode_mean_csl"],
        }
    )
    df = pd.DataFrame(rows)
    df_disp = df.copy()
    df_disp["평균 에피소드 비용"] = df_disp["평균 에피소드 비용"].map(fmt_cost)
    for c in ("CSL (SKU-일)", "CSL (에피소드 평균)"):
        df_disp[c] = df_disp[c].map(fmt_pct)
    st.dataframe(df_disp, hide_index=True, use_container_width=True)
    st.caption(labels.CURRENCY_RULE)
    st.caption(labels.CSL_POOLED_RULE)
    st.caption(labels.A5_NOTE)
    st.caption("파이프라인 행 — " + labels.AGENT_SINGLE_PATH)

    st.subheader("비용–서비스 좌표")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["CSL (SKU-일)"][:-1] * 100,
            y=df["평균 에피소드 비용"][:-1],
            mode="markers+text",
            text=df["정책"][:-1],
            textposition="top center",
            name="정책",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[agent["pooled_sku_day_csl"] * 100],
            y=[agent["mean_episode_cost"]],
            mode="markers+text",
            text=["파이프라인"],
            textposition="bottom center",
            marker=dict(symbol="diamond", size=12),
            name="파이프라인 (1회 실행 경로)",
        )
    )
    fig.update_layout(
        xaxis_title="CSL, SKU-일 (%)",
        yaxis_title="평균 에피소드 비용 (시뮬레이터 단위)",
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(labels.VERDICT_HEADER + " — 상세")
    st.caption(labels.VERDICT_NOTE)
    st.caption(labels.CI_NOTE)
    res = dfl["confirmatory"]["results"]
    for r_key in sorted(res.keys(), key=lambda k: int(k[1:])):
        r_val = res[r_key]
        with st.expander(f"r = {r_key[1:]} (주문 후보 수)", expanded=(r_key == "r5")):
            st.markdown("**비용 (H1)** — 기준선 대비 B의 비용 차이, 양수가 B 우위")
            h1_rows = []
            for base in ("a2", "a3", "a4"):
                e = r_val["h1"][base]
                h1_rows.append(
                    {
                        "기준선": BASELINE_NAMES.get(base, base),
                        "차이 추정": fmt_cost(e["point_estimate"]),
                        "95% 하한": fmt_cost(e["primary"]["lower_5pct"]),
                        "결과": labels.PASS_LABEL if e["pass"] else labels.FAIL_LABEL,
                    }
                )
            st.dataframe(pd.DataFrame(h1_rows), hide_index=True, use_container_width=True)
            st.markdown(
                f"종합: **{labels.PASS_LABEL if r_val['h1_pass_intersection'] else labels.FAIL_LABEL}** "
                "(세 기준선 모두에서 우위일 때만 충족)"
            )

            st.markdown("**서비스 (H1-SL)** — 기준선 대비 B의 CSL 차이 (%p), 음수가 서비스 하락")
            sl_rows = []
            for base in ("a2", "a3", "a4"):
                e = r_val["h1_sl"][base]
                sl_rows.append(
                    {
                        "기준선": BASELINE_NAMES.get(base, base),
                        "CSL 차이 (%p)": f"{e['point_estimate'] * 100:+.2f}",
                        "결과": labels.PASS_LABEL if e["pass"] else labels.FAIL_LABEL,
                    }
                )
            st.dataframe(pd.DataFrame(sl_rows), hide_index=True, use_container_width=True)
            st.markdown(
                f"종합: **{labels.PASS_LABEL if r_val['h1_sl_pass_all'] else labels.FAIL_LABEL}**"
            )

            hb = r_val["h1_b"]
            st.markdown(
                "**보조 비교 (H1-b)** — 외부 대회 우승 접근을 참고한 강한 기준선(A-PTO) 대비: "
                f"차이 {fmt_cost(hb['point_estimate'])}, 95% 하한 {fmt_cost(hb['primary']['lower_5pct'])} → "
                f"**{labels.PASS_LABEL if hb['pass'] else labels.FAIL_LABEL}** "
                "(예측 모델 구조가 달라 H1과 분리해 보조로만 제시)"
            )
            a5 = r_val["a5_reference"]
            gap_all = a5.get("mean_gap_b_minus_a5_all")
            gap_1_11 = a5.get("mean_gap_b_minus_a5_folds_1_11")
            if gap_all is not None and gap_1_11 is not None:
                st.caption(
                    f"수요를 미리 아는 참고 기준(A5) 대비 B의 평균 비용 격차: "
                    f"{gap_all:,.1f} (마지막 폴드 제외 {gap_1_11:,.1f}) — "
                    "참고용이며 달성 가능한 하한이 아닙니다. "
                    "마지막 폴드는 평가 구간이 짧아 함께 해석하지 않습니다."
                )

    st.info(labels.DFL_NARRATIVE)
