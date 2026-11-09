"""Photo Lab: brightness/contrast (with a clip on/off switch), thresholding,
annotation and alpha channels in one small Streamlit app.

Concepts used (CAI2840C, Mod 1-2):
  1. Image arithmetic: brightness (add) + contrast (multiply), with np.clip on/off
  2. Thresholding: global, Otsu, adaptive (mean/gaussian), inverted or not
  3. Drawing / annotation: cv2.putText caption + cv2.rectangle border
  4. Alpha channel: use the threshold mask as transparency, download a PNG
"""
import cv2
import numpy as np
import streamlit as st


# ---------- image processing (pure functions, no Streamlit) ----------

def adjust(img_bgr, brightness, contrast, clip):
    """out = img * contrast + brightness.

    clip=True  -> values are capped to 0..255 (np.clip), the right way.
    clip=False -> values wrap around modulo 256, the overflow bug from class
                  (e.g. 276 -> 20), so bright areas turn dark and speckled.
    """
    out = img_bgr.astype(np.float64) * contrast + brightness
    if clip:
        out = np.clip(out, 0, 255)
    else:
        out = np.mod(np.round(out), 256)
    return out.astype(np.uint8)


def make_mask(img_bgr, method, thresh, invert, block_size=31, c=10):
    """Return a binary (0/255) single-channel mask from the image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kind = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if method == "Global":
        _, mask = cv2.threshold(gray, thresh, 255, kind)
    elif method == "Otsu":
        _, mask = cv2.threshold(gray, 0, 255, kind + cv2.THRESH_OTSU)
    else:  # Adaptive (Gaussian)
        block_size = block_size | 1  # must be odd
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, kind, block_size, c
        )
    return mask


def annotate(img_bgr, text, box, position):
    """Draw a magenta rectangle border and a centred caption (copy, never in place)."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    if box:
        m = max(2, int(0.02 * min(h, w)))
        cv2.rectangle(out, (m, m), (w - m, h - m), (255, 0, 255), max(2, m // 3))
    if text:
        scale = max(0.6, w / 900)
        thick = max(1, int(scale * 2))
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x = max(0, (w - tw) // 2)
        y = {"Top": th + 30, "Center": (h + th) // 2, "Bottom": h - 30 - base}[position]
        # dark outline first so the white text stays readable on any photo
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def with_alpha(img_bgr, mask):
    """Attach the mask as the alpha channel -> BGRA (255 keeps the pixel, 0 is transparent)."""
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])


def process(img_bgr, brightness, contrast, clip, effect, thresh, invert,
            block_size, c, caption, box, position, transparent):
    """Full pipeline. Returns (display_rgb_or_rgba, png_bgr_or_bgra)."""
    out = adjust(img_bgr, brightness, contrast, clip)

    mask = None
    if effect != "None" or transparent:
        method = effect if effect != "None" else "Otsu"
        mask = make_mask(out, method, thresh, invert, block_size, c)

    if effect != "None" and not transparent:
        out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # show the binary result

    out = annotate(out, caption, box, position)

    if transparent:
        out = with_alpha(out, mask)
        return cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA), out
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), out


# ---------- Streamlit UI ----------

def load_upload(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # drops any alpha; always 3-channel BGR
    if img is None:
        return None
    # keep the app fast on big phone photos
    h, w = img.shape[:2]
    if max(h, w) > 1200:
        s = 1200 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def sample_image():
    """Built-in demo image so the app works before anything is uploaded."""
    h, w = 360, 640
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = np.linspace(40, 235, w, dtype=np.uint8)[None, :, None]  # horizontal gradient
    cv2.circle(img, (160, 180), 90, (40, 60, 220), -1)
    cv2.rectangle(img, (300, 80), (560, 280), (230, 230, 230), -1)
    cv2.putText(img, "Upload your own photo", (150, 335), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def main():
    st.set_page_config(page_title="Photo Lab", page_icon="🖼️", layout="wide")
    st.title("🖼️ Photo Lab")
    st.caption(
        "Brightness & contrast with a clipping switch, thresholding, annotation and "
        "transparent backgrounds, all with OpenCV. Upload a photo and move the controls."
    )

    with st.sidebar:
        st.header("1. Image")
        up = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])

        st.header("2. Brightness & contrast")
        brightness = st.slider("Brightness (add)", -100, 100, 0)
        contrast = st.slider("Contrast (multiply)", 0.5, 3.0, 1.0, 0.1)
        clip = st.checkbox("Clip to 0-255 (np.clip)", value=True,
                           help="Turn OFF and push contrast up: bright areas wrap around to dark.")

        st.header("3. Threshold")
        effect = st.selectbox("Mode", ["None", "Global", "Otsu", "Adaptive"])
        thresh, block_size, c = 127, 31, 10
        if effect == "Global":
            thresh = st.slider("Threshold T", 0, 255, 127)
        if effect == "Adaptive":
            block_size = st.slider("Neighbourhood size", 3, 101, 31, 2)
            c = st.slider("C (subtracted from the local mean)", 0, 30, 10)
        invert = st.checkbox("Invert (THRESH_BINARY_INV)", value=False)

        st.header("4. Annotate")
        caption = st.text_input("Caption", "")
        position = st.radio("Caption position", ["Top", "Center", "Bottom"], index=2, horizontal=True)
        box = st.checkbox("Magenta border box", value=False)

        st.header("5. Alpha channel")
        transparent = st.checkbox(
            "Make the mask transparent (PNG)",
            help="Uses the threshold mask (Otsu if Mode is None) as the alpha channel: "
                 "white = keep, black = transparent.",
        )

    if up is not None:
        img = load_upload(up)
        if img is None:
            st.error("Could not read that file as an image. Try a JPG or PNG.")
            st.stop()
    else:
        img = sample_image()
        st.info("Showing a demo image. Upload your own in the sidebar.")

    shown, out = process(img, brightness, contrast, clip, effect, thresh, invert,
                         block_size, c, caption, box, position, transparent)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width='stretch')
    with col2:
        st.subheader("Processed")
        st.image(shown, width='stretch')

    ok, png = cv2.imencode(".png", out)  # PNG keeps the alpha channel
    if ok:
        st.download_button("⬇️ Download result (PNG)", png.tobytes(), "photo_lab.png", "image/png")

    with st.expander("What's happening?"):
        st.markdown(
            "- **Brightness/contrast:** `out = img × contrast + brightness`.\n"
            "- **Clip off:** values above 255 wrap around (276 → 20), so bright areas turn dark. "
            "Clip on caps them at 255 with `np.clip`.\n"
            "- **Global vs Otsu vs Adaptive:** one cutoff for the whole image, a cutoff picked "
            "from the histogram, or a different cutoff for each neighbourhood.\n"
            "- **Alpha:** a 4th channel; 255 = opaque, 0 = transparent."
        )


if __name__ == "__main__":
    main()
