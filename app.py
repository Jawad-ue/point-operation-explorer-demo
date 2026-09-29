"""
Point Operation Explorer
=========================
A chat-style interface for learning point operations in Digital Image
Processing. Upload an image and pick an operation from the compose bar
at the bottom (like picking a model), hit Apply, and the result appears
as a message in the conversation — before/after images, a plain-English
explanation, and the transformation curve. Keep chatting below for
follow-up questions (real AI, your own API key) or type something like
"generate a report" for a plain-text summary — no AI call needed for that.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ai_assistant import build_image_context, get_chat_response
from image_utils import get_basic_stats, get_image_info, load_image, validate_image
from operations import EXAMPLE_INPUTS, OPERATIONS
from report_utils import generate_report_markdown

st.set_page_config(page_title="Point Operation Explorer", page_icon="🖼️", layout="wide")

WELCOME_MSG = (
    "👋 Upload an image and pick an operation from the bar below, then hit "
    "**Apply**. I'll process it and explain what happened. Keep chatting for "
    "follow-up questions, or type something like *\"generate a report\"* for "
    "a summary."
)

SECRET_KEY_NAMES = {
    "Anthropic (Claude)": "ANTHROPIC_API_KEY",
    "OpenAI (GPT)": "OPENAI_API_KEY",
    "Groq (Free)": "GROQ_API_KEY",
}

REPORT_KEYWORDS = ("report",)

# ----------------------------------------------------------------------
# Styling: minimal chat look + a real fixed compose bar at the bottom
# ----------------------------------------------------------------------
st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    .block-container { max-width: 880px; padding-top: 1rem; padding-bottom: 240px; }
    div[data-testid="stMetric"] {
        background: #F7F9FC; border-radius: 8px; padding: 0.4rem 0.6rem;
        border: 1px solid #E5E9F0;
    }
    div[data-testid="stChatMessage"] { padding: 0.3rem 0; }

    div[class*="st-key-footer_bar"] {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: min(880px, calc(100vw - 2rem));
        background: var(--background-color, #FFFFFF);
        border: 1px solid #E5E9F0;
        border-radius: 14px 14px 0 0;
        box-shadow: 0 -2px 12px rgba(0,0,0,0.06);
        padding: 0.6rem 0.9rem 0.8rem 0.9rem;
        z-index: 999;
    }

    @media (max-width: 768px) {
        div[data-testid="stDeployButton"] { display: none; }

        div[class*="st-key-footer_bar"] {
            width: calc(100vw - 1rem);
            padding-bottom: calc(0.8rem + env(safe-area-inset-bottom));
        }
        div[data-testid="stAppDeployButton"],
            div[data-testid="stStatusWidget"] {
            display: none !important;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Sidebar — AI provider/model only; the key itself lives in secrets.toml
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 AI Assistant")
    provider = st.selectbox(
        "Provider", ["Anthropic (Claude)", "OpenAI (GPT)", "Groq (Free)"], key="ai_provider"
    )
    default_models = {
        "Anthropic (Claude)": "claude-sonnet-5",
        "OpenAI (GPT)": "gpt-6-astra",
        "Groq (Free)": "llama-3.3-70b-versatile",
    }
    if "ai_model" not in st.session_state:
        st.session_state["ai_model"] = default_models[provider]
    st.text_input("Model", key="ai_model")

    secret_name = SECRET_KEY_NAMES[provider]
    try:
        api_key = st.secrets.get(secret_name, "")
    except Exception:
        api_key = ""

    if api_key:
        st.success(f"✅ {secret_name} loaded from secrets.toml")
    else:
        st.warning(f"⚠️ {secret_name} not found")
        st.caption(
            "Add it to `.streamlit/secrets.toml` in your project folder — see README.md."
        )

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
st.session_state.setdefault("conversation", [])
st.session_state.setdefault("api_chat_history", [])
st.session_state.setdefault("current_image", None)
st.session_state.setdefault("latest_operation", None)
st.session_state.setdefault("last_upload_signature", None)

st.markdown("## 🖼️ Point Operation Explorer")

# ----------------------------------------------------------------------
# Bottom compose bar: attach, operation, parameters, apply, chat input
# ----------------------------------------------------------------------
with st.container(key="footer_bar"):
    c1, c2, c3, c4 = st.columns([1, 3, 1.4, 1])

    with c1:
        with st.popover("📎", use_container_width=True):
            uploaded_file = st.file_uploader(
                "Image",
                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                label_visibility="collapsed",
            )
            if uploaded_file is not None:
                signature = (uploaded_file.name, uploaded_file.size)
                if signature != st.session_state["last_upload_signature"]:
                    is_valid, message = validate_image(uploaded_file)
                    if not is_valid:
                        st.error(f"❌ {message}")
                    else:
                        image_data = load_image(uploaded_file)
                        info = get_image_info(image_data)
                        original_array = image_data["array_gray"]
                        original_stats = get_basic_stats(original_array)
                        st.session_state["current_image"] = {
                            "info": info,
                            "pil_rgb": image_data["pil_rgb"],
                            "original_array": original_array,
                            "original_stats": original_stats,
                        }
                        st.session_state["latest_operation"] = None
                        st.session_state["last_upload_signature"] = signature
                        st.session_state["conversation"].append(
                            {
                                "kind": "image_upload",
                                "role": "user",
                                "info": info,
                                "stats": original_stats,
                                "pil_rgb": image_data["pil_rgb"],
                            }
                        )

    current = st.session_state["current_image"]

    with c2:
        operation_name = st.selectbox(
            "Operation",
            list(OPERATIONS.keys()),
            key="selected_operation",
            label_visibility="collapsed",
        )
    op = OPERATIONS[operation_name]

    params = {}
    with c3:
        if current is not None and op["params_spec"]:
            with st.popover("⚙️ Params", use_container_width=True):
                for spec in op["params_spec"]:
                    default_val = (
                        spec["default_fn"](current["original_array"])
                        if "default_fn" in spec
                        else spec["default"]
                    )
                    widget_key = f"param_{operation_name}_{spec['key']}"
                    if spec["type"] == "slider":
                        params[spec["key"]] = st.slider(
                            spec["label"],
                            float(spec["min"]),
                            float(spec["max"]),
                            value=float(default_val),
                            step=float(spec.get("step", 1.0)),
                            key=widget_key,
                        )
                    elif spec["type"] == "select":
                        options = spec["options"]
                        idx = options.index(default_val) if default_val in options else 0
                        params[spec["key"]] = st.selectbox(
                            spec["label"], options, index=idx, key=widget_key
                        )
        elif current is None:
            st.caption("Upload first")
        else:
            st.caption("No parameters")

    with c4:
        apply_clicked = st.button("▶ Apply", use_container_width=True, disabled=current is None)

    user_msg = st.chat_input("Ask about your image, the operation, or DIP...")

# Backfill any params not rendered above (e.g. popover never opened, or no image yet)
if current is not None:
    for spec in op["params_spec"]:
        if spec["key"] not in params:
            params[spec["key"]] = (
                spec["default_fn"](current["original_array"]) if "default_fn" in spec else spec["default"]
            )

# ----------------------------------------------------------------------
# Handle Apply
# ----------------------------------------------------------------------
if apply_clicked and current is not None:
    original_array = current["original_array"]
    original_stats = current["original_stats"]
    processed_array = op["function"](original_array, params)
    processed_stats = get_basic_stats(processed_array)
    change_summary = op["describe_change"](original_stats, processed_stats, params)

    example_arr = np.array(EXAMPLE_INPUTS, dtype="uint8")
    example_out = op["function"](example_arr, params).tolist()

    result_entry = {
        "kind": "operation_result",
        "role": "assistant",
        "operation_name": operation_name,
        "op": op,
        "params": params,
        "original_array": original_array,
        "processed_array": processed_array,
        "original_stats": original_stats,
        "processed_stats": processed_stats,
        "change_summary": change_summary,
        "example_pairs": list(zip(EXAMPLE_INPUTS, example_out)),
    }
    st.session_state["conversation"].append(result_entry)
    st.session_state["latest_operation"] = result_entry
    st.rerun()


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------
def render_stats_row(stats: dict):
    cols = st.columns(5)
    cols[0].metric("Min", stats["min"])
    cols[1].metric("Max", stats["max"])
    cols[2].metric("Mean", stats["mean"])
    cols[3].metric("Std Dev", stats["std"])
    cols[4].metric("Median", stats["median"])


def render_histogram(array: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 2))
    ax.hist(array.ravel(), bins=256, range=(0, 255), color="#4C72B0")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Intensity", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_xlim(0, 255)
    st.pyplot(fig, use_container_width=True)


def render_transform_curve(op_: dict, params_: dict):
    """Plot s = T(r) for r in [0, 255] — the actual shape of the operation."""
    r = np.arange(256, dtype=np.uint8)
    s = op_["function"](r, params_)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(range(256), s, color="#4C72B0", linewidth=2, label="s = T(r)")
    ax.plot([0, 255], [0, 255], linestyle="--", color="#bbb", linewidth=1, label="identity (no change)")
    ax.set_xlabel("Input intensity (r)")
    ax.set_ylabel("Output intensity (s)")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.legend(fontsize=7, loc="lower right")
    st.pyplot(fig, use_container_width=True)


def render_image_upload(entry: dict):
    cols = st.columns([1, 3])
    with cols[0]:
        st.image(entry["pil_rgb"], width=120)
    with cols[1]:
        info = entry["info"]
        st.markdown(f"**{info['filename']}** — {info['width']}×{info['height']}px, {info['original_mode']}")


def render_operation_result(entry: dict, idx: int):
    op_ = entry["op"]
    st.markdown(f"**{entry['operation_name']}**  \n{op_['what']}")
    if entry["params"]:
        param_str = ", ".join(f"{k}={v}" for k, v in entry["params"].items())
        st.caption(f"Parameters — {param_str}")
    st.latex(op_["formula_latex"])

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.image(entry["original_array"], caption="Before", use_container_width=True, clamp=True)
    with bcol2:
        st.image(entry["processed_array"], caption="After", use_container_width=True, clamp=True)

    st.info(entry["change_summary"])

    st.markdown("**How it works — the transformation curve:**")
    render_transform_curve(op_, entry["params"])

    with st.expander("🔍 Detailed stats & pixel values"):
        st.markdown("*Before*")
        render_stats_row(entry["original_stats"])
        st.markdown("*After*")
        render_stats_row(entry["processed_stats"])
        hcol1, hcol2 = st.columns(2)
        with hcol1:
            render_histogram(entry["original_array"], "Before")
        with hcol2:
            render_histogram(entry["processed_array"], "After")
        st.table(
            {
                "Original (r)": [p[0] for p in entry["example_pairs"]],
                "Transformed (s)": [p[1] for p in entry["example_pairs"]],
            }
        )


# ----------------------------------------------------------------------
# Chat feed
# ----------------------------------------------------------------------
if not st.session_state["conversation"]:
    with st.chat_message("assistant"):
        st.markdown(WELCOME_MSG)
else:
    for idx, entry in enumerate(st.session_state["conversation"]):
        with st.chat_message(entry["role"]):
            if entry["kind"] == "image_upload":
                render_image_upload(entry)
            elif entry["kind"] == "operation_result":
                render_operation_result(entry, idx)
            elif entry["kind"] == "text":
                st.markdown(entry["content"])

# ----------------------------------------------------------------------
# Handle chat input
# ----------------------------------------------------------------------
if user_msg:
    st.session_state["conversation"].append({"kind": "text", "role": "user", "content": user_msg})

    is_report_request = any(word in user_msg.lower() for word in REPORT_KEYWORDS)
    latest_op = st.session_state["latest_operation"]

    if is_report_request:
        if latest_op is None:
            reply = "Apply an operation first, then ask me for a report."
        else:
            reply = generate_report_markdown(
                filename=st.session_state["current_image"]["info"]["filename"],
                image_info=st.session_state["current_image"]["info"],
                operation_name=latest_op["operation_name"],
                operation_meta=latest_op["op"],
                params_used=latest_op["params"],
                original_array=latest_op["original_array"],
                processed_array=latest_op["processed_array"],
                original_stats=latest_op["original_stats"],
                processed_stats=latest_op["processed_stats"],
                change_summary=latest_op["change_summary"],
            )
        # Deterministic, local — no AI call, so it doesn't touch the
        # provider's chat history either.
        st.session_state["conversation"].append({"kind": "text", "role": "assistant", "content": reply})
    else:
        st.session_state["api_chat_history"].append({"role": "user", "content": user_msg})
        if not api_key:
            reply = f"No API key found for {provider} — add {secret_name} to `.streamlit/secrets.toml`."
        else:
            current_img = st.session_state["current_image"]
            if current_img is not None:
                context = build_image_context(
                    image_info=current_img["info"],
                    original_stats=current_img["original_stats"],
                    operation_name=latest_op["operation_name"] if latest_op else None,
                    operation_meta=latest_op["op"] if latest_op else None,
                    params_used=latest_op["params"] if latest_op else None,
                    processed_stats=latest_op["processed_stats"] if latest_op else None,
                    change_summary=latest_op["change_summary"] if latest_op else None,
                )
            else:
                context = "No image has been uploaded yet."
            try:
                reply = get_chat_response(
                    provider=provider,
                    api_key=api_key,
                    model=st.session_state["ai_model"],
                    context=context,
                    history=st.session_state["api_chat_history"],
                )
            except Exception as e:
                reply = f"⚠️ AI request failed: {e}"
        st.session_state["conversation"].append({"kind": "text", "role": "assistant", "content": reply})
        st.session_state["api_chat_history"].append({"role": "assistant", "content": reply})

    st.rerun()
