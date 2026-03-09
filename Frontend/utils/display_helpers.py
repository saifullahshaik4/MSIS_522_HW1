"""Reusable Streamlit display helpers."""
from __future__ import annotations

import streamlit as st
import pandas as pd
from pathlib import Path


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label=label, value=value, help=help_text or None)


def render_metric_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            metric_card(label, value, help_text)


def render_image_section(
    title: str,
    image_path: Path,
    interpretation: str = "",
    use_expander: bool = False,
) -> None:
    if not image_path.exists():
        st.warning(f"Plot not found: `{image_path.name}`")
        return
    if use_expander:
        with st.expander(title, expanded=True):
            st.image(str(image_path), use_container_width=True)
            if interpretation:
                st.caption(interpretation)
    else:
        st.subheader(title)
        st.image(str(image_path), use_container_width=True)
        if interpretation:
            st.info(interpretation)


def divider() -> None:
    st.markdown("---")


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    divider()
