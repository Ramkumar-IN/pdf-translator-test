import streamlit as st
import fitz
#import cv2
import numpy as np
import os
import markdown
import asyncio
import re
import tempfile
import time
import pandas as pd
from PIL import Image
from google import genai
from doclayout_yolo import YOLOv10
from dotenv import load_dotenv
import streamlit_gsheets
from streamlit_gsheets import GSheetsConnection

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="AI PDF Document Translator - Linkinlegal", layout="wide")
st.title("🌍 AI PDF Document Translator - Linkinlegal")

# --- CONFIGURATION ---
GEMINI_MODEL = "gemini-2.5-flash"  # Using 2.0 Flash for stable metadata access

# Initialize Connection for Permanent Stats
conn = st.connection("gsheets", type=GSheetsConnection)

# --- API KEY LOADING ---
api_key = None

# 1. Try Streamlit Secrets (for Cloud or local .streamlit/secrets.toml)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]



# 3. Final Check
if not api_key:
    st.error("❌ API Key Missing! Please add GOOGLE_API_KEY to your .env or secrets.toml file.")
    st.stop() # Stops the app here so you don't get that long Traceback error

client = genai.Client(api_key=api_key)

# Model Loading - Ensure path is accessible for Streamlit
model_path = os.path.join("models", "doclayout_yolo_docstructbench_imgsz1024.pt")
if not os.path.exists(model_path):
    # Fallback to your local absolute path for Cursor testing
    model_path = r"C:\Users\mailr\chatbot-rag-main\models\doclayout_yolo_docstructbench_imgsz1024.pt"

model_detect = YOLOv10(model_path)

# --- YOUR ORIGINAL FUNCTIONS (EXACTLY AS PROVIDED) ---

def check_usage_limit(limit=20):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df = conn.read(worksheet="Sheet1", ttl=0)
        if df.empty:
            return 0, True
        count = len(df)
        return count, count < limit
    except:
        # If the sheet is empty/missing, assume 0 used
        return 0, True

def get_smart_bg_color(page, rect):
    try:
        pix = page.get_pixmap(clip=rect)
        if pix.width < 2 or pix.height < 2: return (1, 1, 1)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        median_rgb = np.median(arr[:, :, :3], axis=(0, 1)) 
        return tuple(c / 255 for c in median_rgb)
    except: return (1, 1, 1)

def get_contrast_color(bg_rgb):
    luminance = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
    return (1, 1, 1) if luminance < 0.5 else (0, 0, 0)

def get_font_for_text(text, bold=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    f_dir = os.path.join(base_dir, "assets", "fonts")
    suffix = "Bold.ttf" if bold else "Regular.ttf"
    if re.search(r'[\u0B80-\u0BFF]', text): return f"noto-tamil{'-bold' if bold else ''}", os.path.join(f_dir, f"NotoSansTamil-{suffix}")
    if re.search(r'[\u0900-\u097F]', text): return f"noto-hindi{'-bold' if bold else ''}", os.path.join(f_dir, f"NotoSansDevanagari-{suffix}")
    if re.search(r'[\u0600-\u06FF]', text): return f"noto-arabic{'-bold' if bold else ''}", os.path.join(f_dir, f"NotoSansArabic-{suffix}")
    return f"noto-latin{'-bold' if bold else ''}", os.path.join(f_dir, f"NotoSans-{suffix}")

def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def filter_overlapping_boxes(boxes, overlap_thresh=0.7):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), reverse=True)
    keep = []
    for box_a in boxes:
        is_redundant = False
        for box_b in keep:
            ax1, ay1, ax2, ay2 = box_a['bbox']; bx1, by1, bx2, by2 = box_b['bbox']
            inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
            inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area_a = (ax2 - ax1) * (ay2 - ay1)
                if inter_area / area_a > overlap_thresh: is_redundant = True; break
        if not is_redundant: keep.append(box_a)
    return keep

def render_table_html(page, rect, md_text):
    lines = [l.strip() for l in md_text.split('\n') if '|' in l]
    if len(lines) < 2: return False
    if not any('---' in line for line in lines[:3]):
        col_count = lines[0].count('|') - 1
        lines.insert(1, "|" + "---|" * max(1, col_count))
    clean_md = '\n'.join(lines)
    target_width, target_height = rect.width, rect.height
    char_count = len(md_text)
    area = target_width * target_height
    font_size = "6pt" if (char_count / area) > 0.05 else "7pt" if (char_count / area) > 0.03 else "8pt"
    try:
        clean_html = markdown.markdown(clean_md, extensions=['tables'])
        user_css = f"table {{ border-collapse: collapse; width: {target_width}pt; height: {target_height}pt; table-layout: fixed; font-family: sans-serif; font-size: {font_size}; line-height: 1.0; background-color: transparent; }} th, td {{ border: 0.5pt solid black; padding: 2pt; text-align: left; vertical-align: top; word-wrap: break-word; overflow: hidden; }} th {{ background-color: #f2f2f2; font-weight: bold; }}"
        page.insert_htmlbox(rect, clean_html, css=user_css)
        return True
    except: return False

def save_markdown_audit(audit_log, report_path):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Translation Audit Report\n\n| Page | Label | Original | Translated |\n| :--- | :--- | :--- | :--- |\n")
        for item in audit_log:
            orig = str(item['original']).replace('\n', ' ').replace('|', 'I')
            trans = str(item['translated']).replace('\n', ' ').replace('|', 'I')
            f.write(f"| {item['page']} | {item['label']} | {orig} | {trans} |\n")

# --- CORE AI LOGIC (WITH ACTUAL TOKEN CAPTURE) ---

async def get_gemini_translation_async(crop_bgr, label, target_lang):
    pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    prompt = (
        f"ACT AS AN EXPERT DOCUMENT TRANSLATOR. Target Language: {target_lang}.\n\n"
        "TASK: Translate the content of the image.\n"
        "CONTEXT: You are looking at a cropped image of a technical document. "    
        "If the source is in Arabic (Right-to-Left script).\n\n"
        "CRITICAL INSTRUCTION FOR SECTION NUMBERS for ARABIC source:\n"
        "1. Identify any hierarchical section numbers (e.g., 6.4.5).\n"
        "2. In Arabic layout, the primary/parent section is on the right.\n"
        "3. You MUST reverse this for the English translation. \n"
        "   - Example: 'إجراءات إيقاف 6.4.5' -> '5.4.6 Procedures for Suspending'.\n"
        "   - Example: '1.9' -> '9.1'.\n"
        "   - Example: '2.4.5.1' -> '1.5.4.2'.\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Output ONLY the translated text. No explanations.\n"
        "- Use transliteration for specific technical terms, names, and acronyms that don't have a direct translation.\n"
        "-  Dates and date period can be numerically maintained as is.\n"
        "- If it is a Table, return ONLY Markdown format, No preamble .\n"
        "- If it is a LOGO, ICON, or SEAL, mailid, phone number, url, return the word 'SKIP'.\n"
        "- NO original source script characters allowed in the final translated text, even if the source in capital letter."
        "- Every single word must be translated or transliterated into the target script. Absolutely zero Latin/English characters are allowed if they are the source language."
    )
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(model=GEMINI_MODEL, contents=[prompt, pil_img])
            # EXTRACTING ACTUAL TOKENS
            actual_tokens = response.usage_metadata.total_token_count
            return (response.text.strip() if response.text else "NONE"), actual_tokens
        except: await asyncio.sleep(2)
    return "ERROR", 0

async def process_page(doc, page_idx, semaphore, target_lang):
    async with semaphore:
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = model_detect(img, imgsz=1024, conf=0.20, verbose=False)
        raw_dets = [{'label': model_detect.names[int(box.cls[0])], 'bbox': box.xyxy[0].cpu().numpy().astype(int).tolist()} for r in results for box in r.boxes]
        detections = filter_overlapping_boxes(raw_dets)
        tasks, valid_dets = [], []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            rect = fitz.Rect(x1/2, y1/2, x2/2, y2/2)
            orig_txt = page.get_text("text", clip=rect).strip()
            if orig_txt.isdigit() and len(orig_txt) < 4: continue
            tasks.append(get_gemini_translation_async(img[y1:y2, x1:x2], det['label'], target_lang))
            valid_dets.append((det, rect, orig_txt))
        
        results_gathering = await asyncio.gather(*tasks)
        translations = [r[0] for r in results_gathering]
        page_actual_tokens = sum(r[1] for r in results_gathering)
        return page_idx, (valid_dets, translations, page_actual_tokens)

# --- MAIN ENGINE ---

async def main_engine(pdf_input_path, target_lang, file_name):
    start_time = time.time()
    doc = fitz.open(pdf_input_path)
    output_path = f"mirrored_{file_name}"
    audit_path = "translation_notes.md"
    audit_log, semaphore = [], asyncio.Semaphore(10) # Using your Semaphore(10)
    all_results = [None] * len(doc)
    total_actual_tokens = 0

    prog_bar = st.progress(0)
    status_msg = st.empty()

    tasks = [process_page(doc, i, semaphore, target_lang) for i in range(len(doc))]
    pages_completed = 0
    for i, task in enumerate(asyncio.as_completed(tasks)):
        idx, (valid_dets, translations, page_tokens) = await task
        all_results[idx] = (valid_dets, translations)
        total_actual_tokens += page_tokens
        pages_completed += 1
        prog_bar.progress(pages_completed / len(doc))
        status_msg.info(f"⚡ Finished {pages_completed}/{len(doc)} pages... (Processed Page {idx+1})")

    for page_idx, page_data in enumerate(all_results):
        if not page_data: continue
        valid_dets, translations = page_data
        page = doc[page_idx]
        pdf_w, pdf_h = page.rect.width, page.rect.height
        pix_for_scale = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        scale_x, scale_y = pdf_w / pix_for_scale.width, pdf_h / pix_for_scale.height
        is_rtl_page = any(is_arabic(orig) for _, _, orig in valid_dets)

        captured_logos = {}
        for idx, ((det, _, _), translation) in enumerate(zip(valid_dets, translations)):
            x1, y1, x2, y2 = det['bbox']
            orig_rect = fitz.Rect(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            is_logo = any(word in translation.upper() for word in ["SKIP", "NONE", "LOGO", "ICON", "SEAL"])
            if is_logo:
                try:
                    logo_pix = page.get_pixmap(clip=orig_rect, matrix=fitz.Matrix(3, 3))
                    samp_x = orig_rect.x0 - 2 if orig_rect.x0 > 2 else orig_rect.x1 + 2
                    bg_pix_sample = page.get_pixmap(clip=fitz.Rect(samp_x, orig_rect.y0, samp_x+1, orig_rect.y0+1))
                    outside_color = [c/255 for c in bg_pix_sample.pixel(0, 0)]
                    captured_logos[idx] = {"pixmap": logo_pix, "bg_color": outside_color}
                    page.draw_rect(orig_rect, color=outside_color, fill=outside_color, overlay=True)
                except: pass
            else:
                bg_color = get_smart_bg_color(page, orig_rect)
                page.draw_rect(orig_rect, color=bg_color, fill=bg_color, overlay=True)

        for idx, ((det, rect_raw, orig_txt), translation) in enumerate(zip(valid_dets, translations)):
            x1, y1, x2, y2 = det['bbox']
            orig_rect = fitz.Rect(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            rect = fitz.Rect(pdf_w - orig_rect.x1, orig_rect.y0, pdf_w - orig_rect.x0, orig_rect.y1) if is_rtl_page else orig_rect
            audit_log.append({"page": page_idx + 1, "label": det['label'], "original": orig_txt, "translated": translation})
            
            if idx in captured_logos:
                final_bg = captured_logos[idx]["bg_color"]
                page.draw_rect(rect, color=final_bg, fill=final_bg, overlay=True)
                page.insert_image(rect, pixmap=captured_logos[idx]["pixmap"])
                continue

            if any(word in translation.upper() for word in ["SKIP", "NONE"]): continue
            bg_color = get_smart_bg_color(page, rect)
            text_color = (0, 0, 0) if (0.299*bg_color[0] + 0.587*bg_color[1] + 0.114*bg_color[2]) > 0.5 else (1, 1, 1)
            page.draw_rect(rect, color=bg_color, fill=bg_color, overlay=True)

            detected_align = 0; is_bold = False; start_fs = 10
            is_table = "table" in det['label'].lower()
            if not is_table:
                words = page.get_text("words", clip=orig_rect)
                if words:
                    # 1. Group words into lines
                    lines_dict = {}
                    for w in words:
                        y_val = round(w[1], 1)
                        if y_val not in lines_dict: lines_dict[y_val] = []
                        lines_dict[y_val].append(w)
                    
                    # 2. Apply your strict difference logic for multi-line blocks
                    if len(lines_dict) > 1:
                        is_centered_block = True
                        for y in lines_dict:
                            line_x0 = min(w[0] for w in lines_dict[y])
                            line_x2 = max(w[2] for w in lines_dict[y])
                            
                            # Calculate space between text and YOLO box
                            left_space = line_x0 - orig_rect.x0
                            right_space = orig_rect.x1 - line_x2
                            
                            # If ANY line is not balanced (within 3px tolerance), it's LEFT
                            if abs(left_space - right_space) > 3:
                                is_centered_block = False
                                break
                        detected_align = 1 if is_centered_block else 0
                    else:
                        # For 1-line boxes, default to Left
                        detected_align = 0
                    

            text_dict = page.get_text("dict", clip=orig_rect)
            all_spans = [span for b in text_dict["blocks"] if "lines" in b for l in b["lines"] for span in l["spans"]]
            if all_spans:
                if not is_table: is_bold = sum(1 for s in all_spans if s["flags"] & 2**4) > (len(all_spans)/2)
                sizes = [s["size"] for s in all_spans if "size" in s]
                if sizes: start_fs = max(sizes)
            
            font_id, font_path = get_font_for_text(translation, bold=is_bold)
            page.insert_font(fontname=font_id, fontfile=font_path)

            if is_table:
                if not render_table_html(page, rect, translation):
                    page.insert_textbox(rect, translation, fontsize=7, fontname=font_id, color=text_color, align=0)
            else:
                inserted = False
                for fs in range(int(start_fs), 3, -1):
                    if page.insert_textbox(rect, translation, fontsize=fs, fontname=font_id, color=text_color, align=detected_align) >= 0:
                        inserted = True; break
                if not inserted: page.insert_textbox(rect, translation, fontsize=5, fontname=font_id, color=text_color, align=detected_align)

    doc.save(output_path, garbage=4, deflate=True)
    save_markdown_audit(audit_log, audit_path)

    # PERMANENT LOGGING: Google Sheets with ACTUAL TOKENS
    # Calculate processing time (add 'start_time = time.time()' at start of main_engine)
    duration = round(time.time() - start_time, 2) 

    # PERMANENT LOGGING: Enhanced Statistics
    new_stats = pd.DataFrame([{
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": file_name,
        "Pages": len(doc),
        "Act_tokens": int(total_actual_tokens), 
        "Language": target_lang,
        "Processing_Sec": duration,
        "Avg_Sec_Per_Page": round(duration / len(doc), 2)
    }])

    try:
        # Re-initialize for fresh state
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # 1. READ: Get current data or empty DF
        try:
            existing = conn.read(worksheet="Sheet1", ttl=0)
        except:
            existing = pd.DataFrame()
        
        # 2. MERGE: Add new row
        if not existing.empty:
            updated = pd.concat([existing, new_stats], ignore_index=True)
        else:
            updated = new_stats
            
        # 3. WRITE: index=False prevents "extra column" errors
        conn.update(worksheet="Sheet1", data=updated)
        st.sidebar.success("📊 Stats Synced to Google Sheets!")

    except Exception as e:
        # If this still fails, PLEASE tell me the exact text that appears after 'Failed:'
        st.sidebar.error(f"Sheet Sync Failed: {e}")

    status_msg.success("🎉 Process Complete!")
    return output_path, audit_path

# --- UI FLOW ---
st.sidebar.markdown("### 🌐 Language Settings")

# Fixed Box for Source Language (Now at the top)
st.sidebar.write("🔍 **Source Language**")
st.sidebar.code("Auto-Detect", language=None) 
st.sidebar.info("The AI automatically identifies the source language")

st.sidebar.markdown("---")

# Target Language Dropdown (Now below)
sel_lang = st.sidebar.selectbox(
    "🎯 Target Language", 
    ["English", "Tamil", "Hindi", "Japanese", "Chinese", "German", "Russian"]
)
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Reset session if a brand new file is uploaded
    if "file_id" not in st.session_state or st.session_state.file_id != uploaded_file.name:
        st.session_state.doc_orig = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        st.session_state.file_id = uploaded_file.name
        st.session_state.out_pdf = None 
        st.session_state.out_notes = None

    total_p = len(st.session_state.doc_orig)
    page_sel = st.sidebar.slider("View Page #", 1, total_p, 1) - 1
    # 1. First, check the limit
    usage_count, can_process = check_usage_limit(20)
    st.sidebar.markdown(f"**Translation count :** {usage_count} / 20 documents")

    # 2. Logic: If we are under the limit, show the button
    if can_process:
        if st.button("🚀 Process Document"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            pdf_res, notes_res = asyncio.run(main_engine(tmp_path, sel_lang, uploaded_file.name))
            st.session_state.out_pdf = pdf_res
            st.session_state.out_notes = notes_res
            st.rerun() # Refresh to update the usage count immediately
    else:
        # 3. ONLY show error if can_process is False (limit actually reached)
        st.error(f"🚫 Limit Reached! You have already processed {usage_count} documents.")
        st.info("Please ask Ram Kumar to reset the limit.")

    # 4. Display results (this stays outside so you can still view/download past work)
   
    if st.session_state.get("out_pdf"):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original")
            p_orig = st.session_state.doc_orig[page_sel].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            st.image(Image.frombuffer("RGB", [p_orig.width, p_orig.height], p_orig.samples))
        with c2:
            st.subheader("Translated")
            doc_res = fitz.open(st.session_state.out_pdf)
            p_res = doc_res[page_sel].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            st.image(Image.frombuffer("RGB", [p_res.width, p_res.height], p_res.samples))

        st.divider()
        db1, db2 = st.columns(2)
        with db1: 
            st.download_button("📥 Download Translation as PDF", open(st.session_state.out_pdf, "rb"), file_name=f"mirrored_{uploaded_file.name}")
        with db2: 
            st.download_button("📝 Download Translated texts", open(st.session_state.out_notes, "rb"), file_name="translation_notes.md")
