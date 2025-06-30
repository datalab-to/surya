import os
import re
from typing import List

from surya.recognition import RecognitionPredictor
from surya.foundation import FoundationPredictor
from surya.common.surya.schema import TaskNames

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS
)

import io

import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import hashlib
import pypdfium2

from surya.settings import settings
from PIL import Image

MAX_WIDTH = 800
MAX_HEIGHT = 1000


def replace_fences(text):
    text = re.sub(r'<math display="block">(.*?)</math>', r"$$\1$$", text)
    text = re.sub(r"<math>(.*?)</math>", r"$\1$", text)
    text = re.sub(r'<math display="inline">(.*?)</math>', r"$\1$", text)
    return text


@st.cache_resource()
def load_predictor():
    foundation_predictor = FoundationPredictor()
    return RecognitionPredictor(foundation_predictor)


@st.cache_data()
def inference(pil_image: Image.Image, bbox: List[float]):
    input_img = pil_image.crop(bbox)
    bbox = [0, 0, input_img.width, input_img.height]
    model_output = predictor(
        [input_img], [TaskNames.block_without_boxes], bboxes=[[bbox]]
    )
    return model_output[0].text_lines[0].text


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=settings.IMAGE_DPI_HIGHRES):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    doc.close()
    return png_image


@st.cache_data()
def page_counter(pdf_file):
    doc = open_pdf(pdf_file)
    doc_len = len(doc)
    doc.close()
    return doc_len


def resize_image(pil_image):
    if pil_image is None:
        return
    pil_image.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)


def get_canvas_hash(pil_image):
    return hashlib.md5(pil_image.tobytes()).hexdigest()


st.set_page_config(layout="wide")

top_message = """### LaTeX OCR

After the model loads, upload an image or a pdf, then draw a box around the equation or text you want to OCR by clicking and dragging. Surya will convert it to Markdown with LaTeX math on the right.
"""

st.markdown(top_message)
col1, col2 = st.columns([0.7, 0.3])

predictor = load_predictor()

in_file = st.sidebar.file_uploader(
    "PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"]
)
if in_file is None:
    st.stop()

if in_file is None:
    st.stop()

filetype = in_file.type
page_count = None
if "pdf" in filetype:
    page_count = page_counter(in_file)
    page_number = st.sidebar.number_input(
        f"Page number out of {page_count}:", min_value=1, value=1, max_value=page_count
    )
    pil_image = get_page_image(in_file, page_number, dpi=settings.IMAGE_DPI_HIGHRES)
else:
    pil_image = Image.open(in_file).convert("RGB")
    page_number = None

if pil_image is None:
    st.stop()

pil_image.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)
canvas_hash = get_canvas_hash(pil_image)

with col1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.1)",  # Fixed fill color with some opacity
        stroke_width=1,
        stroke_color="#FFAA00",
        background_color="#FFF",
        background_image=pil_image,
        update_streamlit=True,
        height=pil_image.height,
        width=pil_image.width,
        drawing_mode="rect",
        point_display_radius=0,
        key=canvas_hash,
    )

if not canvas_result.json_data:
    st.stop()

objects = pd.json_normalize(
    canvas_result.json_data["objects"]
)  # need to convert obj to str because PyArrow
bbox_list = None
if objects.shape[0] > 0:
    boxes = objects[objects["type"] == "rect"][["left", "top", "width", "height"]]
    boxes["right"] = boxes["left"] + boxes["width"]
    boxes["bottom"] = boxes["top"] + boxes["height"]
    bbox_list = boxes[["left", "top", "right", "bottom"]].values.tolist()

if bbox_list:
    with col2:
        texts = [inference(pil_image, bbox) for bbox in bbox_list]
        for idx, latex in enumerate(reversed(texts)):
            st.markdown(f"### {len(texts) - idx}")
            st.markdown(replace_fences(latex), unsafe_allow_html=True)
            st.code(latex)
            st.divider()

with col2:
    tips = """
    ### Usage tips
    - Texify is sensitive to how you draw the box around the text you want to OCR. If you get bad results, try selecting a slightly different box, or splitting the box into multiple.
    """
    st.markdown(tips)
