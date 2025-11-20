import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from utils import preprocess, get_glyph_info, find_nearest_label


st.set_page_config(page_title="Hieroglyph Learner", layout="centered")
st.title("🪶 Egyptian Hieroglyph Classifier")
st.markdown("""
Draw a hieroglyph below and click **Identify** to classify it using your training samples.  
You can also **save** your drawing as a new training sample below.
""")


# ============================================================
# Canvas (必须透明 fill，image_data 才能返回)
# ============================================================

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # transparent fill
    stroke_width=5,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas"
)

st.write("Has image_data:", canvas_result.image_data is not None)  # debug


# ============================================================
# Identify using KNN
# ============================================================

if st.button("🔍 Identify Sign"):
    st.write("Button clicked!")

    if canvas_result.image_data is not None:
        st.write("Image data detected!")

        # Convert canvas to PIL image
        img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))

        # preprocess for model
        # preprocess
        img_arr = preprocess(img).flatten()

# 使用 1-NN 找最近的标签
        pred_id = find_nearest_label(img_arr)
        if pred_id is None:
            pred_id = "NoModel"


        st.write("Predicted ID (debug):", pred_id)

        # Lookup JSON info
        info = get_glyph_info(pred_id)
        st.write("Lookup result:", info)

        # UI result
        if info:
            st.success(f"Predicted Sign: **{pred_id}**")
            st.markdown(f"""
**Glyph**: {info.get("glyph", "")}  
**Phonetic value**: `{info.get("phonetic", "")}`  
**Meaning**: *{info.get("meaning", "")}*  
**Unicode**: `{info.get("unicode", "")}`
""")
        else:
            if pred_id == "NoModel":
                st.warning("No training data found. Save samples below first!")
            else:
                st.warning(f"I couldn't find info for sign '{pred_id}'. Add it to glyph_data.json if needed.")

    else:
        st.info("Please draw a sign on the canvas first 🙂")



# ============================================================
# Save sample
# ============================================================

st.subheader("💾 Save Training Sample")

label = st.text_input("Label this drawing (e.g., A1, G17, M17):")

if st.button("📁 Save As Training Sample"):
    if canvas_result.image_data is not None and label.strip():
        img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))

        import os
        save_dir = f"data/{label}"
        os.makedirs(save_dir, exist_ok=True)

        count = len(os.listdir(save_dir))
        img.save(f"{save_dir}/{count+1}.png")

        st.success(f"Saved sample #{count+1} for label {label}!")
    else:
        st.warning("Please draw something AND enter a label before saving!")





