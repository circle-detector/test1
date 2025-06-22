
import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("تحليل صور Sentinel-2 وعدّ دوائر الري المحوري")

st.markdown("""
📸 قم برفع صورة جوية من الأقمار الصناعية (مثل Sentinel-2).
🛰️ سيتم اكتشاف وعدّ **دوائر الري المحوري فقط** باستخدام تحليل الألوان والأحجام.
""")

uploaded_file = st.file_uploader("📥 اختر صورة (JPG / PNG)", type=["jpg", "jpeg", "png"])

def is_irrigation_circle(image, circle, min_radius=80, max_radius=300):
    x, y, r = circle
    if not (min_radius < r < max_radius):
        return False

    # mask لتحديد داخل الدائرة
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r - 5, 255, -1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_val = cv2.mean(hsv, mask=mask)

    hue, sat, val = mean_val[:3]
    green = (35 <= hue <= 90 and sat > 40 and val > 40)
    brown = (10 <= hue <= 30 and sat > 20 and val > 40)
    return green or brown

def detect_irrigation_circles(image_pil):
    image = np.array(image_pil.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blurred, 50, 150)

    h, w = gray.shape
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
                               param1=50, param2=30,
                               minRadius=int(min(h, w) * 0.05),
                               maxRadius=int(min(h, w) * 0.2))

    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        irrigation_circles = [c for c in circles if is_irrigation_circle(image, c)]
        count = len(irrigation_circles)
        for i, c in enumerate(irrigation_circles, 1):
            x, y, r = c
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.putText(output, str(i), (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    output_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    return count, output_image

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 الصورة الأصلية", use_column_width=True)
        count, result_image = detect_irrigation_circles(image)
        st.image(result_image, caption=f"🟢 عدد دوائر الري المحوري المكتشفة: {count}", use_column_width=True)
        st.success(f"✅ تم اكتشاف {count} دائرة ري محوري.")
    except Exception as e:
        st.error(f"حدث خطأ أثناء المعالجة: {e}")
