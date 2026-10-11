"""Approval section: human-in-the-loop flow, verdicts, stored cases.

All figures come from the verified snapshot. The case panel shows
reviewed stored records only; no live model calls (ST9-05). Internal
English marker strings are replaced by fixed Korean labels (ST9-06).
"""

import pandas as pd
import streamlit as st

import labels

GATE_LABELS = {
    "evaluated_candidates": "검증한 후보 수",
    "gate_passed_candidates": "통과한 후보 수",
    "capacity_rejected": "용량 기준 기각",
    "csl_rejected": "서비스 기준 기각",
    "candidate_gate_pass_rate": "통과율",
    "all_rejected": "전원 기각 여부",
}

CASE_TITLES = {
    "C1_normal": "사례 1 — 일부 후보 통과, 조치 승인",
    "C2_fallback": "사례 2 — 전원 기각, 안전 장치 발동",
    "C3_partial_gate": "사례 3 — 부분 통과 (다른 묶음)",
    "C4_full_pass": "사례 4 — 전원 통과",
}


def fmt(v):
    if isinstance(v, bool):
        return "예" if v else "아니오"
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def render(snapshot: dict) -> None:
    st.header("승인 — 사람이 승인하는 의사결정 흐름")
    ap = snapshot["approval"]

    st.subheader("운영 흐름에서 파이프라인이 맡는 부분")
    st.markdown(
        "- **파이프라인이 대신하는 단계**: 근거 수집 → 조치 후보 생성·검증 → "
        "승인 요청 초안과 설명 작성 → 적용 후 추적\n"
        "- **사람이 유지하는 단계**: 조치 범위의 정의, 승인(① 수급 계획 담당, "
        "② 물류센터 관리자), 실행과 되돌리기\n"
        "- 모든 후보 조치는 용량·서비스 기준의 검증 게이트를 거치며, 전부 "
        "기각되면 조치 없이 기본 정책을 적용합니다(안전 장치)."
    )

    st.subheader(labels.VERDICT_HEADER)
    st.caption(labels.VERDICT_NOTE)
    h = ap["h3_h5"]
    h3 = h["h3"]
    st.markdown(
        f"**{labels.HYPOTHESES['h3']}** — "
        f"**{labels.PASS_LABEL if h3['pass'] else labels.FAIL_LABEL}**"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("비용 차이 (L0 − 파이프라인, 평균)", f"{h3['point_estimate']:,.1f}")
    c2.metric("95% 하한 (단측)", f"{h3['primary_lower_95_one_sided']:,.1f}")
    csl = h3["csl_delta_descriptive"]
    c3.metric(
        "CSL 변화 (파이프라인 − L0)",
        f"{csl['point'] * 100:+.2f}%p",
        delta=f"{csl['point'] * 100:+.2f}%p",
        delta_color="inverse",
    )
    st.caption(labels.H3_JOINT_CAVEAT)

    h5 = h["h5"]
    st.markdown(
        f"**{labels.HYPOTHESES['h5']}** — "
        f"**{labels.PASS_LABEL if h5['pass'] else labels.FAIL_LABEL}**"
    )
    st.markdown(
        f"480주 중 승리 {h5['strict_win']}건({h5['p_win']:.2%}), "
        f"동률 {h5['tie']}건, 패배 {h5['loss']}건 — 판정 기준인 과반 승리"
        f"({h5['threshold_min_wins']}건)에 미달했습니다. 동률 중 {h5['tie_both_fallback']}건은 "
        "양쪽 모두 안전 장치가 발동해 같은 기본 정책으로 귀결된 주입니다."
    )
    st.caption(
        f"동률을 제외한 승률은 {h5['tie_excluded_rate_auxiliary']:.1%}이지만, "
        "판정에는 사용하지 않는 보조 수치입니다. 이 결과에 따라 시스템을 "
        "자율 '에이전트'가 아니라 사람의 승인을 보조하는 의사결정 지원 "
        "파이프라인으로 규정합니다."
    )

    rm = ap["run_meta"]
    st.subheader("실행 개요 (저장된 공식 실행 1회)")
    st.caption(labels.AGENT_SINGLE_PATH)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("평가 주 수", f"{rm['n_weeks']:,}")
    m2.metric("모델 호출 비용 (USD)", f"{rm['ledger']['spent_usd']:,.2f}")
    m3.metric("모델 호출 수", f"{rm['ledger']['calls']:,}")
    m4.metric("조치 선정 소요 중앙값 (초)", f"{rm['selection_time_median_s']:.2f}")
    st.caption(labels.LEADING_UNVERIFIED)

    st.subheader("저장된 실행 기록 — 대표 사례")
    st.caption(labels.CASE_PANEL_NOTE)
    st.caption(labels.CASE_CAPTION)
    for case in ap["cases"]:
        title = CASE_TITLES.get(case["case_label"], case["case_label"])
        with st.expander(
            f"{title} · 폴드 {case['fold']} · {case['week']}주차 · {case['group_id']}"
        ):
            a1, a2, a3 = st.columns(3)
            a1.metric("승인된 목표 수준 (τ)", fmt(case["selected_tau"]))
            a2.metric(
                "SKU별 승수 조정",
                "없음" if not case["selected_multipliers"] else
                f"{len(case['selected_multipliers'])}건",
            )
            a3.metric("안전 장치 발동", "예" if case["fallback"] else "아니오")
            g_rows = [
                {"항목": GATE_LABELS.get(k, k), "값": fmt(v)}
                for k, v in case["gate"].items()
            ]
            st.dataframe(pd.DataFrame(g_rows), hide_index=True, use_container_width=True)
            st.markdown("**저장된 설명 (모델 출력 원문)**")
            st.text(case["explanation"]["text"])
    with st.expander("사례 선정 방식"):
        st.text(ap["case_selection_rule"])
