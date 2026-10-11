"""Stage 9 read-only dashboard entry point.

Four sections per plan 4.10: forecast, DFL ablation, approval,
evaluation. All figures come from the verified display snapshot.
External wording follows the two-layer principle (ST9-06).
"""

import streamlit as st

import labels
from data_loader import SnapshotIntegrityError, load_snapshot

st.set_page_config(
    page_title="SCM Control Tower — 의사결정 지원 파이프라인",
    layout="wide",
)


@st.cache_data
def get_data():
    return load_snapshot()


def sorted_r_keys(results: dict) -> list:
    return sorted(results.keys(), key=lambda k: int(k[1:]))


def verdict(flag: bool) -> str:
    return labels.PASS_LABEL if flag else labels.FAIL_LABEL


def main() -> None:
    try:
        snapshot, manifest = get_data()
    except SnapshotIntegrityError as e:
        st.error(f"배포 데이터 무결성 확인 실패 — 수치를 표시하지 않습니다. ({e})")
        st.stop()
        return
    except FileNotFoundError as e:
        st.error(f"배포 데이터 파일 없음 — 배포 구성을 확인하세요. ({e})")
        st.stop()
        return

    st.title("SCM Control Tower — " + labels.NAMING)
    st.warning(labels.SIMULATION_BANNER)
    st.caption(labels.READ_ONLY_NOTE)

    section = st.sidebar.radio(
        "섹션",
        ["개요", "예측", "DFL ablation", "승인", "평가"],
    )
    st.sidebar.caption(labels.INTEGRITY_NOTE)
    st.sidebar.caption(
        f"배포 데이터 식별값: {manifest['snapshot_sha256'][:16]}... / "
        f"생성 {manifest['generated_at']}"
    )

    if section == "개요":
        render_overview(snapshot, manifest)
    elif section == "예측":
        from sections import forecast
        forecast.render(snapshot)
    elif section == "DFL ablation":
        from sections import dfl
        dfl.render(snapshot)
    elif section == "승인":
        from sections import approval
        approval.render(snapshot)
    elif section == "평가":
        from sections import evaluation
        evaluation.render(snapshot)


def render_overview(snapshot: dict, manifest: dict) -> None:
    st.header("개요")
    st.write(
        "물류센터 공유 용량 제약 아래에서 일일 발주를 지원하는 파이프라인의 "
        "평가 결과를 조회합니다. 섹션: 예측 / DFL ablation / 승인 / 평가."
    )
    st.subheader(labels.VERDICT_HEADER)
    st.caption(labels.VERDICT_NOTE)
    res = snapshot["dfl"]["confirmatory"]["results"]
    h = snapshot["approval"]["h3_h5"]
    r_keys = sorted_r_keys(res)
    rows = [
        (labels.HYPOTHESES["h1"],
         " / ".join(
             f"r={rk[1:]}: {verdict(res[rk]['h1_pass_intersection'])}"
             for rk in r_keys
         )),
        (labels.HYPOTHESES["h1_sl"],
         " / ".join(
             f"r={rk[1:]}: {verdict(res[rk]['h1_sl_pass_all'])}"
             for rk in r_keys
         )),
        (labels.HYPOTHESES["h3"],
         f"{verdict(h['h3']['pass'])} "
         f"(비용차 95% 하한 {h['h3']['primary_lower_95_one_sided']:.1f}) — "
         + labels.H3_JOINT_CAVEAT),
        (labels.HYPOTHESES["h5"],
         f"{verdict(h['h5']['pass'])} "
         f"({h['h5']['strict_win']}/480 = {h['h5']['p_win']:.2%} — 판정 기준: 승률 50% 초과 필요)"),
    ]
    for name, result in rows:
        st.markdown(f"- **{name}**: {result}")
    st.caption(labels.DFL_NARRATIVE)
    st.subheader("데이터 출처")
    st.caption(labels.INTEGRITY_NOTE)
    with st.expander("원본 산출물 식별값 (sha256)"):
        st.json(manifest["sources"])


if __name__ == "__main__":
    main()
