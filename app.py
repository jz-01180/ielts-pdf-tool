import io
import json
import os
import re
import random
from difflib import SequenceMatcher
from typing import Any

import fitz
import pytesseract
import streamlit as st

from PIL import Image
from openai import OpenAI


# =========================
# 基本设置
# =========================

st.set_page_config(
    page_title="IELTS 阅读 PDF 自动批注工具",
    page_icon="📘",
    layout="wide"
)


# Windows、macOS、Linux 均可通过环境变量指定 Tesseract 路径
tesseract_cmd = os.getenv("TESSERACT_CMD")

if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
else:
    default_windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if os.path.exists(default_windows_path):
        pytesseract.pytesseract.tesseract_cmd = default_windows_path


# =========================
# 工具函数
# =========================

def normalize_text(text: str) -> str:
    """统一空格、连字符和大小写，方便模糊匹配。"""

    text = text.replace("\u00ad", "")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def clean_json_text(text: str) -> str:
    """清除 AI 可能返回的 Markdown 代码围栏。"""

    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


def extract_json(text: str) -> dict[str, Any]:
    """提取 AI 返回的 JSON。"""

    text = clean_json_text(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])

        raise ValueError("AI 返回的内容不是有效 JSON")


# =========================
# OCR 和 PDF 文字提取
# =========================

def ocr_page(page: fitz.Page) -> tuple[str, list[dict[str, Any]]]:
    """
    对扫描页进行 OCR。
    返回：
    1. 页面文字
    2. 每个 OCR 行的文字和坐标
    """

    matrix = fitz.Matrix(2.5, 2.5)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    image = Image.frombytes(
        "RGB",
        [pixmap.width, pixmap.height],
        pixmap.samples
    )

    data = pytesseract.image_to_data(
        image,
        lang="eng",
        output_type=pytesseract.Output.DICT
    )

    lines = {}

    for i, raw_word in enumerate(data["text"]):
        word = raw_word.strip()

        if not word:
            continue

        confidence = data["conf"][i]

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0

        if confidence < 20:
            continue

        block_num = data["block_num"][i]
        par_num = data["par_num"][i]
        line_num = data["line_num"][i]

        key = (block_num, par_num, line_num)

        x = data["left"][i] / 2.5
        y = data["top"][i] / 2.5
        w = data["width"][i] / 2.5
        h = data["height"][i] / 2.5

        if key not in lines:
            lines[key] = {
                "text": [],
                "rects": []
            }

        lines[key]["text"].append(word)
        lines[key]["rects"].append(
            fitz.Rect(x, y, x + w, y + h)
        )

    result_lines = []

    for item in lines.values():
        text = " ".join(item["text"]).strip()

        if text:
            result_lines.append({
                "text": text,
                "rects": item["rects"]
            })

    result_lines.sort(
        key=lambda item: (
            item["rects"][0].y0,
            item["rects"][0].x0
        )
    )

    page_text = "\n".join(
        item["text"]
        for item in result_lines
    )

    return page_text, result_lines


def extract_pdf(pdf_bytes: bytes) -> tuple[fitz.Document, list[dict[str, Any]], str]:
    """
    提取整个 PDF。
    普通 PDF 直接提取文字。
    扫描 PDF 自动 OCR。
    """

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    page_infos = []
    all_text = []

    for page_index, page in enumerate(document):
        native_text = page.get_text("text").strip()

        # 普通文字 PDF
        if len(native_text) >= 80:
            page_text = native_text
            ocr_lines = []
            is_ocr = False

        # 扫描 PDF
        else:
            page_text, ocr_lines = ocr_page(page)
            is_ocr = True

        page_info = {
            "page_number": page_index + 1,
            "text": page_text,
            "ocr_lines": ocr_lines,
            "is_ocr": is_ocr
        }

        page_infos.append(page_info)

        all_text.append(
            f"\n--- PDF PAGE {page_index + 1} ---\n{page_text}"
        )

    return document, page_infos, "\n".join(all_text)


# =========================
# AI 分析
# =========================

def build_prompt(pdf_text: str, min_wrong: int, max_wrong: int) -> str:
    return f"""
你是一名专业雅思阅读教师、试卷识别器和 PDF 批注规划器。

请分析下面的雅思阅读 PDF 文本，并且只返回合法 JSON。
不要返回 Markdown，不要返回解释文字。

重要要求：

1. 识别每一篇 Reading Passage。
2. 识别每一道题，包括题号、题目、题型和选项。
3. 为每一道题只生成一个最终答案。
4. 每篇阅读必须有 {min_wrong} 到 {max_wrong} 道错误答案。
5. 错误答案必须是合理的、容易犯的错误。
6. 不能在同一道题中同时展示正确答案和错误答案。
7. 对于错误答案，仍然必须给出与该答案相关的原文证据。
8. explanation 用中文说明解题思路。
9. explanation 不要直接说“这是故意设置的错误”。
10. evidence_text 必须尽可能是 PDF 原文中连续出现的文字。
11. evidence_page 必须是原文证据所在页码，从 1 开始。
12. question_page 必须是题目所在页码，从 1 开始。
13. 如果证据横跨两页，选择证据主要所在的页面。
14. 如果找不到精确证据，选择最相关的连续短语。
15. 对于选择题，answer 写选项字母和选项内容。
16. 对于判断题，answer 只能是 TRUE、FALSE 或 NOT GIVEN。
17. 对于填空题，answer 写最终填入的词语或短语。
18. question_number 必须是数字。
19. is_correct 必须是 true 或 false。

如果某篇题目总数小于 {min_wrong} 道，则将所有可能的题目设置为错误题，
不要虚构不存在的题目。

严格按照下面格式返回：

{{
  "passages": [
    {{
      "passage_number": 1,
      "title": "文章标题",
      "questions": [
        {{
          "question_number": 1,
          "question": "题目原文",
          "question_type": "multiple_choice",
          "answer": "B. 选项内容",
          "is_correct": true,
          "question_page": 1,
          "evidence_page": 1,
          "evidence_text": "原文连续短语",
          "explanation": "中文解题思路"
        }}
      ]
    }}
  ]
}}

题型可以使用：

- multiple_choice
- true_false_not_given
- yes_no_not_given
- matching_headings
- matching_information
- matching_features
- sentence_completion
- summary_completion
- table_completion
- short_answer
- other

每篇的错误题数量必须处在 {min_wrong} 到 {max_wrong} 之间。

PDF 文本如下：

{pdf_text}
"""


def analyze_pdf(
    client: OpenAI,
    pdf_text: str,
    model_name: str,
    min_wrong: int,
    max_wrong: int
) -> dict[str, Any]:

    prompt = build_prompt(
        pdf_text,
        min_wrong,
        max_wrong
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.25,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "你只能返回合法 JSON。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content

    return extract_json(content)


def wrong_answer_count(result: dict[str, Any]) -> dict[int, int]:
    counts = {}

    for passage in result.get("passages", []):
        passage_number = int(
            passage.get("passage_number", 0)
        )

        count = 0

        for question in passage.get("questions", []):
            if question.get("is_correct") is False:
                count += 1

        counts[passage_number] = count

    return counts


def is_wrong_count_valid(
    result: dict[str, Any],
    min_wrong: int,
    max_wrong: int
) -> bool:

    for passage in result.get("passages", []):
        questions = passage.get("questions", [])

        if not questions:
            continue

        wrong_count = sum(
            1
            for question in questions
            if question.get("is_correct") is False
        )

        allowed_min = min(min_wrong, len(questions))

        if wrong_count < allowed_min:
            return False

        if wrong_count > min(max_wrong, len(questions)):
            return False

    return True


def analyze_with_retry(
    client: OpenAI,
    pdf_text: str,
    model_name: str,
    min_wrong: int,
    max_wrong: int,
    attempts: int = 4
) -> dict[str, Any]:

    last_result = None

    for _ in range(attempts):
        result = analyze_pdf(
            client=client,
            pdf_text=pdf_text,
            model_name=model_name,
            min_wrong=min_wrong,
            max_wrong=max_wrong
        )

        last_result = result

        if is_wrong_count_valid(
            result,
            min_wrong,
            max_wrong
        ):
            return result

    return last_result


# =========================
# PDF 定位和批注
# =========================

def find_text_rects(
    page: fitz.Page,
    page_info: dict[str, Any],
    target: str
) -> list[fitz.Rect]:

    if not target:
        return []

    # 普通 PDF：直接使用 PDF 的搜索功能
    if not page_info["is_ocr"]:
        exact = page.search_for(target)

        if exact:
            return exact

        # 逐渐缩短文本，处理换行、标点不同等情况
        words = re.findall(r"\S+", target)

        for size in [20, 12, 8, 5, 3]:
            if len(words) < size:
                continue

            partial = " ".join(words[:size])
            found = page.search_for(partial)

            if found:
                return found

    # OCR PDF：根据 OCR 行文字进行模糊匹配
    normalized_target = normalize_text(target)

    if not normalized_target:
        return []

    best_score = 0
    best_rects = []

    for line in page_info.get("ocr_lines", []):
        line_text = normalize_text(line["text"])

        score = SequenceMatcher(
            None,
            normalized_target[:160],
            line_text[:160]
        ).ratio()

        if normalized_target[:30] in line_text:
            score = max(score, 0.9)

        if score > best_score:
            best_score = score
            best_rects = line["rects"]

    if best_score >= 0.45:
        return best_rects

    return []


def highlight_rects(
    page: fitz.Page,
    rects: list[fitz.Rect],
    color=(1, 0.85, 0)
) -> None:

    for rect in rects[:8]:
        annotation = page.add_highlight_annot(rect)
        annotation.set_colors(stroke=color)
        annotation.update()


def find_question_position(
    page: fitz.Page,
    question_number: int,
    question_index: int
) -> fitz.Point:
    """
    动态寻找题号。
    支持：
    1.
    1)
    Question 1
    """

    candidates = [
        f"{question_number}.",
        f"{question_number})",
        f"Question {question_number}",
        f"QUESTION {question_number}"
    ]

    for candidate in candidates:
        rects = page.search_for(candidate)

        if rects:
            rect = rects[0]
            return fitz.Point(
                min(rect.x1 + 15, page.rect.width - 250),
                rect.y0
            )

    # 找不到题号时，使用页面右侧空白区域
    # 按题目序号错开，避免全部重叠
    y = 35 + (question_index % 10) * 55

    return fitz.Point(
        max(page.rect.width - 230, 20),
        min(y, page.rect.height - 70)
    )


def add_answer_box(
    page: fitz.Page,
    question_number: int,
    answer: str,
    explanation: str,
    question_index: int
) -> None:

    point = find_question_position(
        page,
        question_number,
        question_index
    )

    x = point.x
    y = point.y

    # 如果右侧空间不足，就放到题号附近下方
    if x + 220 > page.rect.width:
        x = max(15, page.rect.width - 225)

    if y + 58 > page.rect.height:
        y = max(15, page.rect.height - 65)

    box = fitz.Rect(
        x,
        y,
        min(x + 215, page.rect.width - 8),
        min(y + 55, page.rect.height - 8)
    )

    # 画浅蓝背景框
    page.draw_rect(
        box,
        color=(0.1, 0.3, 0.8),
        fill=(0.92, 0.96, 1.0),
        width=0.7,
        overlay=True
    )

    short_explanation = explanation.strip()

    # 防止 AI 返回过长内容导致批注溢出
    if len(short_explanation) > 150:
        short_explanation = short_explanation[:147] + "..."

    text = (
        f"答案：{answer}\n"
        f"思路：{short_explanation}"
    )

    text_box = fitz.Rect(
        box.x0 + 4,
        box.y0 + 3,
        box.x1 - 4,
        box.y1 - 3
    )

    page.insert_textbox(
        text_box,
        text,
        fontsize=7,
        fontname="helv",
        color=(0.05, 0.15, 0.65),
        align=0,
        overlay=True
    )


def add_evidence_label(
    page: fitz.Page,
    rects: list[fitz.Rect],
    question_number: int
) -> None:

    if not rects:
        return

    rect = rects[0]

    x = min(rect.x1 + 4, page.rect.width - 75)
    y = min(rect.y0, page.rect.height - 20)

    label_rect = fitz.Rect(
        x,
        y,
        min(x + 70, page.rect.width - 5),
        min(y + 16, page.rect.height - 4)
    )

    page.draw_rect(
        label_rect,
        color=(0.8, 0.5, 0.05),
        fill=(1.0, 0.97, 0.75),
        width=0.4,
        overlay=True
    )

    page.insert_textbox(
        label_rect,
        f"Evidence Q{question_number}",
        fontsize=6.5,
        fontname="helv",
        color=(0.55, 0.25, 0.0),
        overlay=True
    )


def annotate_pdf(
    pdf_bytes: bytes,
    page_infos: list[dict[str, Any]],
    result: dict[str, Any]
) -> bytes:

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    question_index = 0

    for passage in result.get("passages", []):
        for question in passage.get("questions", []):
            question_index += 1

            question_number = int(
                question.get("question_number", question_index)
            )

            question_page = int(
                question.get("question_page", 1)
            )

            evidence_page = int(
                question.get("evidence_page", 1)
            )

            answer = str(
                question.get("answer", "")
            )

            explanation = str(
                question.get("explanation", "")
            )

            evidence_text = str(
                question.get("evidence_text", "")
            )

            # 页面编号安全修正
            question_page = max(
                1,
                min(question_page, len(document))
            )

            evidence_page = max(
                1,
                min(evidence_page, len(document))
            )

            # 在题目页添加答案和思路
            add_answer_box(
                page=document[question_page - 1],
                question_number=question_number,
                answer=answer,
                explanation=explanation,
                question_index=question_index
            )

            # 在原文页寻找证据
            evidence_page_info = page_infos[evidence_page - 1]

            evidence_rects = find_text_rects(
                page=document[evidence_page - 1],
                page_info=evidence_page_info,
                target=evidence_text
            )

            if evidence_rects:
                highlight_rects(
                    document[evidence_page - 1],
                    evidence_rects
                )

                add_evidence_label(
                    document[evidence_page - 1],
                    evidence_rects,
                    question_number
                )

            else:
                # 找不到证据时，不乱标位置，只添加页脚提示
                page = document[evidence_page - 1]

                warning_rect = fitz.Rect(
                    10,
                    page.rect.height - 22,
                    min(300, page.rect.width - 10),
                    page.rect.height - 7
                )

                page.insert_textbox(
                    warning_rect,
                    f"Q{question_number}: evidence location needs review",
                    fontsize=6.5,
                    fontname="helv",
                    color=(0.8, 0.1, 0.1),
                    overlay=True
                )

    output = io.BytesIO()

    document.save(
        output,
        garbage=4,
        deflate=True,
        clean=True
    )

    document.close()

    return output.getvalue()


# =========================
# Streamlit 界面
# =========================

st.title("📘 IELTS 阅读 PDF 自动批注工具")

st.write(
    "上传任意雅思阅读 PDF，程序会自动识别题目、生成答案、"
    "添加解题思路，并在原文证据处添加高亮。"
)

st.warning(
    "AI 生成结果不是官方答案。请人工复核题目、答案和证据，"
    "尤其是扫描质量较差、表格复杂或排版特殊的 PDF。"
)

with st.sidebar:
    st.header("设置")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password"
    )

    model_name = st.text_input(
        "AI 模型",
        value="gpt-4o-mini"
    )

    min_wrong = st.slider(
        "每篇最少错误题数",
        min_value=1,
        max_value=5,
        value=2
    )

    max_wrong = st.slider(
        "每篇最多错误题数",
        min_value=1,
        max_value=8,
        value=5
    )

    retry_times = st.slider(
        "AI 自动重试次数",
        min_value=1,
        max_value=6,
        value=4
    )

uploaded_file = st.file_uploader(
    "上传雅思阅读 PDF 文件",
    type=["pdf"]
)

if uploaded_file is None:
    st.info("请先上传 PDF 文件。")
    st.stop()

if min_wrong > max_wrong:
    st.error("最少错误题数不能大于最多错误题数。")
    st.stop()

pdf_bytes = uploaded_file.read()

st.success(
    f"已上传：{uploaded_file.name}，文件大小："
    f"{len(pdf_bytes) / 1024:.1f} KB"
)

if st.button("开始识别并生成修改版 PDF", type="primary"):

    if not api_key:
        st.error("请先输入 OpenAI API Key。")
        st.stop()

    try:
        with st.spinner("正在提取 PDF 文字并进行 OCR……"):
            document, page_infos, full_text = extract_pdf(pdf_bytes)

        st.write(
            f"检测到 {len(page_infos)} 页，"
            f"提取文字约 {len(full_text)} 个字符。"
        )

        ocr_pages = sum(
            1
            for page in page_infos
            if page["is_ocr"]
        )

        if ocr_pages:
            st.info(
                f"其中 {ocr_pages} 页使用了 OCR。"
            )

        if len(full_text.strip()) < 100:
            st.error(
                "无法从 PDF 中提取足够文字。可能是加密 PDF、"
                "图片质量过低或文件损坏。"
            )
            st.stop()

        with st.spinner("AI 正在识别文章、题目、答案和原文证据……"):
            client = OpenAI(api_key=api_key)

            result = analyze_with_retry(
                client=client,
                pdf_text=full_text,
                model_name=model_name,
                min_wrong=min_wrong,
                max_wrong=max_wrong,
                attempts=retry_times
            )

        counts = wrong_answer_count(result)

        st.subheader("识别结果")

        for passage_number, count in counts.items():
            st.write(
                f"第 {passage_number} 篇："
                f"{count} 道错误答案"
            )

        if not is_wrong_count_valid(
            result,
            min_wrong,
            max_wrong
        ):
            st.warning(
                "AI 多次生成后，部分文章的错误答案数量仍不在目标范围内。"
                "程序仍会继续生成 PDF，但建议检查结果。"
            )

        with st.spinner("正在把答案、批注和高亮写入 PDF……"):
            final_pdf = annotate_pdf(
                pdf_bytes=pdf_bytes,
                page_infos=page_infos,
                result=result
            )

        st.success("修改版 PDF 已生成。")

        output_name = os.path.splitext(
            uploaded_file.name
        )[0] + "_annotated.pdf"

        st.download_button(
            label="下载修改完成的 PDF",
            data=final_pdf,
            file_name=output_name,
            mime="application/pdf"
        )

        with st.expander("查看 AI 识别到的结构化结果"):
            st.json(result)

    except Exception as error:
        st.error(f"处理失败：{error}")
        st.exception(error)
