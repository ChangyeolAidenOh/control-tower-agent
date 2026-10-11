"""Evaluation section: lagging metrics, financial bridge, gamma-k scenarios.

All base figures come from the verified snapshot (stage 8 projection).
The only computation here is the frozen linear bridge formula
(ST9-02/ST9-05): dWC(gamma) = gamma * delta_V_retail, annual = dWC * k.
Metric blocks are nested dicts; unknown sub-keys render as-is rather
than being guessed.
"""

import pandas as pd
import streamlit as st

import labels

N_BUNDLES = 10

GROUP_LABELS = {
    "total_cost": "총비용",
    "csl": "서비스 수준 (CSL)",
    "service_shortfall": "미충족",
    "days_of_inventory": "재고일수",
}

SUB_LABELS = {
    "pooled": "SKU-일",
    "episode_mean": "에피소드 평균",
    "mean_episode": "에피소드 평균",
    "onhand": "보유",
    "on_hand": "보유",
    "intransit": "운송 중",
    "in_transit": "운송 중",
    "total": "합계",
    "total_position": "총 포지션",
    "unmet_sku_day_share": "미충족 SKU-일 비율",
    "stockout_rate_quantity": "수량 결품률",
    "stockout_rate_quantity_pooled": "수량 결품률 (SKU-일)",
    "fill_rate": "수량 충족률",
}

COMPARISON_LABELS = {
    "cost_diff_mean_episode": "평균 에피소드 비용 차이",
    "cost_diff_rel_pct": "비용 차이 (상대 %)",
    "csl_diff_pp": "CSL 차이 (%p)",
    "csl_diff_rel_pct": "CSL 차이 (상대 %)",
    "doi_onhand_diff_days": "재고일수 차이 — 보유 (일)",
    "doi_onhand_diff_rel_pct": "재고일수 차이 — 보유 (상대 %)",
    "fill_qty_diff_pp": "수량 충족률 차이 (%p)",
    "total_position_diff_days": "총 재고 포지션 차이 (일)",
    "unmet_share_diff_pp": "미충족 SKU-일 비율 차이 (%p)",
}

KEY_POLICIES = ["l0_rule_ct", "agent_llm", "a4", "a4_sl", "b"]

POLICY_LABELS = {
    "l0_rule_ct": "L0 (규칙 기반 운영)",
    "agent_llm": "파이프라인 (승인 흐름)",
    "a4": "A4 (비용 선택)",
    "a4_sl": "A4-SL (서비스 지향)",
    "b": "B (결정 중심 학습)",
    "a1": "A1",
    "a2": "A2",
    "a3": "A3",
    "a5": "A5 (완전정보 참고)",
    "apto": "A-PTO",
}


def fmt(v):
    if isinstance(v, bool):
        return "예" if v else "아니오"
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def metric_row(policy_key: str, block: dict) -> dict:
    row = {"정책": POLICY_LABELS.get(policy_key, policy_key)}
    for group in ("total_cost", "csl", "service_shortfall", "days_of_inventory"):
        sub = block.get(group)
        if sub is None:
            continue
        if isinstance(sub, dict):
            for sk, sv in sub.items():
                col = f"{GROUP_LABELS[group]} · {SUB_LABELS.get(sk, sk)}"
                row[col] = fmt(sv)
        else:
            row[GROUP_LABELS[group]] = fmt(sub)
    for group, sub in block.items():
        if group not in GROUP_LABELS:
            row[group] = fmt(sub)
    return row


def render(snapshot: dict) -> None:
    st.header("평가 — 운영 지표와 조건부 재무 시나리오")
    ev = snapshot["evaluation"]

    st.caption(labels.JOINT_REPORT_RULE)

    st.subheader("1. 관찰 — 운영 결과 지표")
    lag = ev["lagging_metrics"]
    rows = [metric_row(k, lag[k]) for k in KEY_POLICIES if k in lag]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(labels.UNMET_SHARE_RULE)
    st.caption(labels.POOLED_SCOPE_RULE)
    st.caption(labels.DOI_RULE)

    other = {k: v for k, v in lag.items() if k not in KEY_POLICIES}
    if other:
        with st.expander("그 외 정책 지표 (에피소드 평균 기준)"):
            o_rows = [metric_row(k, v) for k, v in other.items()]
            st.dataframe(pd.DataFrame(o_rows), hide_index=True, use_container_width=True)

    comp = ev.get("comparisons", {}).get("l0_rule_ct_vs_agent_llm")
    if comp:
        st.subheader("핵심 비교 — L0 (규칙 기반) 대비 파이프라인")
        st.caption(labels.H3_JOINT_CAVEAT)
        c_rows = [
            {"항목": COMPARISON_LABELS.get(k, k), "값": fmt(v)}
            for k, v in comp.items()
        ]
        st.dataframe(pd.DataFrame(c_rows), hide_index=True, use_container_width=True)
        st.caption(labels.DOI_RULE)

    li = ev.get("leading_indicators", {})
    if li:
        st.subheader("운영 특성 관찰값")
        st.caption(li.get("label", "") or labels.LEADING_UNVERIFIED)
        li_rows = []
        lat = li.get("selection_latency_s", {})
        if lat:
            li_rows.append(
                {
                    "항목": "주별 조치 선정 소요 시간 (초)",
                    "값": f"중앙값 {fmt(lat.get('median'))} "
                    f"(p05 {fmt(lat.get('p05'))} / p95 {fmt(lat.get('p95'))})",
                }
            )
        fb = li.get("rollback_fallback_rate", {})
        if fb:
            li_rows.append(
                {
                    "항목": "안전 장치 발동 (조치 없이 기본 정책 적용)",
                    "값": f"{fmt(fb.get('fallback_weeks'))} / {fmt(fb.get('total_weeks'))}주 "
                    f"= {fb.get('rate', 0) * 100:.1f}%",
                }
            )
        ff = li.get("forecast_freshness", {})
        if ff:
            li_rows.append(
                {
                    "항목": "예측 갱신 주기 (일)",
                    "값": fmt(ff.get("selection_forecast_cadence_days")),
                }
            )
        st.dataframe(pd.DataFrame(li_rows), hide_index=True, use_container_width=True)

    st.subheader("2. 재무 브리지 — 보유비용에서 재고자산 평가액으로")
    fb_block = ev["financial_bridge"]
    delta = fb_block["delta"]
    delta_v = delta["l0_minus_agent_sale_price_valued_usd"]
    st.markdown(
        "재고 보유비용 계수가 판매가에 비례하도록 설계되어 있어, 일평균 보유비용을 "
        "그 비례계수로 나누면 판매가 기준 재고자산 평가액이 그대로 유도됩니다. "
        "이 단계까지는 가정(γ·k)이 적용되지 않은 값입니다."
    )
    st.metric(
        "재고자산 평가액 차이 — L0 대비 파이프라인 (묶음 1개, 판매가 기준 USD)",
        f"{delta_v:,.2f}",
        help="판매가 기준 평가액이며, 매입원가 기준 운전자본으로 바꾸려면 아래 γ 가정이 필요합니다.",
    )
    with st.expander("정책별 재고자산 평가액 (참고)"):
        pp_rows = [
            {
                "정책": POLICY_LABELS.get(k, k),
                "일평균 보유비용": fmt(v.get("mean_daily_holding_cost")),
                "재고자산 평가액 (판매가 USD)": fmt(v.get("sale_price_valued_onhand_usd")),
            }
            for k, v in fb_block["per_policy"].items()
        ]
        st.dataframe(pd.DataFrame(pp_rows), hide_index=True, use_container_width=True)
    st.caption(delta.get("note", ""))

    st.subheader("3. 가정 — γ·k 조건부 시나리오")
    st.warning(labels.GAMMA_K_ASSUMPTION)
    col1, col2 = st.columns(2)
    with col1:
        gamma = st.slider(
            "γ — 재고 평가원가율 (매입원가 ÷ 판매가)", 0.50, 0.70, 0.60, 0.01
        )
    with col2:
        k = st.slider("k — 연간 자본비용률", 0.08, 0.12, 0.10, 0.005)

    dwc_bundle = gamma * delta_v
    dwc_subset = dwc_bundle * N_BUNDLES
    annual_subset = dwc_subset * k

    st.markdown("**일회성 운전자본 효과** (재고자산 감소분)")
    c1, c2 = st.columns(2)
    c1.metric("묶음 1개 (USD, 일회성)", f"{dwc_bundle:,.1f}")
    c2.metric(f"서브셋 {N_BUNDLES}묶음 (USD, 일회성)", f"{dwc_subset:,.1f}")

    st.markdown("**연간 자본비용 효과** (운전자본 효과 × k)")
    st.metric(f"서브셋 {N_BUNDLES}묶음 (USD/년)", f"{annual_subset:,.1f}")
    st.caption(
        "두 수치는 성격이 달라(일회성 저량 vs 연간 유량) 합산하지 않으며, "
        "화면에서도 분리해 제시합니다."
    )

    grid_rows = []
    for g in (0.5, 0.6, 0.7):
        row = {"γ": g, "일회성 ΔWC (서브셋, USD)": f"{g * delta_v * N_BUNDLES:,.1f}"}
        for kk in (0.08, 0.10, 0.12):
            row[f"연간, k={kk:.2f}"] = f"{g * delta_v * N_BUNDLES * kk:,.1f}"
        grid_rows.append(row)
    st.markdown("기준 격자 (사전 정의된 가정 조합)")
    st.dataframe(pd.DataFrame(grid_rows), hide_index=True, use_container_width=True)

    st.markdown(
        "절대 금액이 작은 것은 평가 범위(서브셋 10묶음·153개 SKU)의 규모 때문이며, "
        "이 화면의 목적은 금액의 크기가 아니라 환산 구조의 타당성을 보여주는 것입니다. "
        "전사 규모 확장은 별도의 규모 가정을 명시한 뒤에만 가능합니다."
    )
    st.caption(labels.CURRENCY_RULE)
    with st.expander("재무 수치 제시 원칙"):
        for rule in labels.FINANCE_PROHIBITIONS:
            st.markdown(f"- {rule}")
        st.markdown(
            "- 운송 중 재고 감소분은 평가액으로 환산하지 않고 수량·일수 기준으로만 제시합니다."
        )
