from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
CHAPTERS_DIR = ROOT / "chapters"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

NAV_BLUEPRINT: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Chapter 0: Normal Distributions",
        [
            ("0.1", "01_standard_normal"),
            ("0.2", "02_general_normal_with_sliders"),
            ("0.3", "03_multivariate_normal_and_cholesky"),
        ],
    ),
    (
        "Chapter 1: GP Foundations",
        [
            ("1.1", "04_functions_as_vectors"),
            ("1.2", "05_kernels_and_covariance"),
            ("1.3", "06_formal_gp_definition"),
        ],
    ),
    (
        "Chapter 2: GP Regression",
        [
            ("2.1", "07_interactive_gp_regressor"),
            ("2.2", "08_sklearn_gp_autotuning"),
        ],
    ),
]


def _clean_title(title: str) -> str:
    return re.sub(r"^\s*\d+\)\s*", "", title).strip()


def _build_navigation(
    chapters: list[tuple[str, object, str]],
) -> list[dict[str, list[dict[str, object]] | str]]:
    chapters_by_slug = {slug: (title, module) for title, module, slug in chapters}
    used_slugs: set[str] = set()
    sections: list[dict[str, list[dict[str, object]] | str]] = []

    for section_label, items in NAV_BLUEPRINT:
        section_items: list[dict[str, object]] = []
        for number, slug in items:
            chapter_entry = chapters_by_slug.get(slug)
            if chapter_entry is None:
                continue
            title, module = chapter_entry
            used_slugs.add(slug)
            section_items.append(
                {
                    "slug": slug,
                    "title": title,
                    "module": module,
                    "nav_label": f"{number} {_clean_title(title)}",
                }
            )
        if section_items:
            sections.append({"section_label": section_label, "items": section_items})

    remaining = sorted(slug for slug in chapters_by_slug if slug not in used_slugs)
    if remaining:
        extra_items: list[dict[str, object]] = []
        for idx, slug in enumerate(remaining, start=1):
            title, module = chapters_by_slug[slug]
            extra_items.append(
                {
                    "slug": slug,
                    "title": title,
                    "module": module,
                    "nav_label": f"X.{idx} {_clean_title(title)}",
                }
            )
        sections.append({"section_label": "Extra Chapters", "items": extra_items})

    return sections


def load_chapters() -> list[tuple[str, object, str]]:
    chapters: list[tuple[str, object, str]] = []
    if not CHAPTERS_DIR.exists():
        return chapters

    for chapter_dir in sorted(p for p in CHAPTERS_DIR.iterdir() if p.is_dir()):
        page_file = chapter_dir / "page.py"
        if not page_file.exists():
            continue

        module_name = f"chapter_{chapter_dir.name}"
        spec = importlib.util.spec_from_file_location(module_name, page_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        title = getattr(module, "TITLE", chapter_dir.name.replace("_", " "))
        chapters.append((title, module, chapter_dir.name))
    return chapters


def main() -> None:
    st.set_page_config(page_title="Gaussian Process Explainer", layout="wide")
    st.title("Gaussian Process Explainer Book")
    st.caption("Standalone interactive chapters: markdown + LaTeX + code + plots")

    chapters = load_chapters()
    if not chapters:
        st.error("No chapter files found under `chapters/`.")
        return

    nav_sections = _build_navigation(chapters)
    flat_pages = [item for section in nav_sections for item in section["items"]]
    if not flat_pages:
        st.error("No navigable chapters found.")
        return

    all_slugs = [page["slug"] for page in flat_pages]
    if "selected_slug" not in st.session_state or st.session_state["selected_slug"] not in all_slugs:
        st.session_state["selected_slug"] = all_slugs[0]

    current_slug = st.session_state["selected_slug"]
    current_flat_index = all_slugs.index(current_slug)

    st.sidebar.header("Navigation")
    section_labels = [section["section_label"] for section in nav_sections]
    current_section_index = next(
        idx
        for idx, section in enumerate(nav_sections)
        if any(item["slug"] == current_slug for item in section["items"])
    )
    selected_section_label = st.sidebar.selectbox(
        "Chapter",
        section_labels,
        index=current_section_index,
    )
    selected_section = next(
        section for section in nav_sections if section["section_label"] == selected_section_label
    )
    section_pages = selected_section["items"]
    section_page_labels = [page["nav_label"] for page in section_pages]
    default_page_index = next(
        (idx for idx, page in enumerate(section_pages) if page["slug"] == current_slug),
        0,
    )
    selected_page_label = st.sidebar.radio(
        "Subchapter",
        section_page_labels,
        index=default_page_index,
    )
    selected_page = next(page for page in section_pages if page["nav_label"] == selected_page_label)
    if selected_page["slug"] != current_slug:
        st.session_state["selected_slug"] = selected_page["slug"]
        st.rerun()

    current_slug = st.session_state["selected_slug"]
    current_flat_index = all_slugs.index(current_slug)
    current_page = flat_pages[current_flat_index]

    prev_col, info_col, next_col = st.columns([1, 2, 1])
    with prev_col:
        if st.button("Previous Page", disabled=current_flat_index == 0):
            st.session_state["selected_slug"] = all_slugs[current_flat_index - 1]
            st.rerun()
    with info_col:
        st.markdown(
            f"**{selected_section_label}**  |  **{current_page['nav_label']}**"
        )
    with next_col:
        if st.button("Next Page", disabled=current_flat_index == len(flat_pages) - 1):
            st.session_state["selected_slug"] = all_slugs[current_flat_index + 1]
            st.rerun()

    selected_module = current_page["module"]
    render_fn = getattr(selected_module, "render", None)
    if render_fn is None:
        st.error(f"Chapter `{current_page['title']}` does not define `render()`.")
        return
    render_fn()


if __name__ == "__main__":
    main()
