import os
import json
from google import genai
from google.genai import types

# Lấy API key từ GEMINI_API_KEY, fallback GOOGLE_API_KEY và luôn strip để tránh ký tự thừa.
RAW_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
API_KEY = RAW_API_KEY.strip() if RAW_API_KEY else ""

if not API_KEY:
    raise ValueError("Không tìm thấy GEMINI_API_KEY hoặc GOOGLE_API_KEY")

if API_KEY == "YOUR_KEY" or len(API_KEY) < 20 or not API_KEY.startswith("AIza"):
    raise ValueError(
        "GEMINI_API_KEY không hợp lệ. Hãy dùng API key từ Google AI Studio (thường bắt đầu bằng 'AIza')."
    )

# Dùng API v1beta cho Gemini Developer API để hỗ trợ tools và embeddings.
API_VERSION = "v1beta"
client = genai.Client(
    api_key=API_KEY,
    http_options={"api_version": API_VERSION},
)

ARTICLE_MODELS = [
    os.getenv("GEMINI_ARTICLE_MODEL", "gemini-2.5-flash"),
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

ANALYSIS_MODELS = [
    os.getenv("GEMINI_ANALYSIS_MODEL", "gemini-2.5-pro"),
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]

print(
    f"[Gemini] api_version={API_VERSION}, article_model={ARTICLE_MODELS[0]}, analysis_model={ANALYSIS_MODELS[0]}"
)

if os.getenv("GEMINI_LIST_MODELS_ON_STARTUP", "false").strip().lower() in {"1", "true", "yes"}:
    try:
        for available_model in client.models.list():
            print(available_model.name)
    except Exception as model_list_error:
        print(f"Không thể lấy danh sách model Gemini: {model_list_error}")


def _unique_models(models: list[str]) -> list[str]:
    unique: list[str] = []
    for model in models:
        if model and model not in unique:
            unique.append(model)
    return unique


def _generate_with_model_fallback(
    model_candidates: list[str],
    contents,
    temperature: float,
    tools: list = None,
):
    last_error = None
    for model_name in _unique_models(model_candidates):
        try:
            config = types.GenerateContentConfig(temperature=temperature)
            if tools:
                config.tools = tools
                
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as err:
            last_error = err
            continue

    raise RuntimeError(f"Không có model Gemini khả dụng. Lỗi cuối: {last_error}")

def generate_article(title: str, category: str) -> str:
    prompt = f"""
    Bạn là một nhà báo và biên tập viên chuyên nghiệp. Hãy viết một bài báo hoàn chỉnh, chất lượng cao dựa trên các thông tin sau.

    Tiêu đề / Chủ đề: {title}
    Thể loại: {category}

    Yêu cầu quan trọng:
    1. Ngôn ngữ: Tự động phát hiện ngôn ngữ của Tiêu đề ({title}). Nếu Tiêu đề bằng tiếng Việt, hãy viết toàn bộ bài báo bằng tiếng Việt. Nếu Tiêu đề bằng tiếng Anh, hãy viết toàn bộ bài báo bằng tiếng Anh.
    2. Cấu trúc bài viết: Phải có Mở bài thu hút, Thân bài chi tiết (chia thành các mục nhỏ/subheadings), và Kết luận đúc kết lại vấn đề.
    3. Văn phong: Chuyên nghiệp, khách quan, sâu sắc và thu hút người đọc.
    4. Độ dài: Khoảng 500 - 800 từ.
    5. Định dạng: Trình bày đẹp mắt bằng Markdown (sử dụng in đậm, in nghiêng, danh sách, và các thẻ heading H2, H3 phù hợp).
    """

    try:
        response = _generate_with_model_fallback(
            model_candidates=ARTICLE_MODELS,
            contents=prompt,
            temperature=0.7,
        )
        return response.text if response.text else "Không tạo được nội dung"

    except Exception as e:
        raise Exception(f"Lỗi Gemini: {str(e)}")

def analyze_medical_record_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    prompt = """
    Bạn là một bác sĩ và trợ lý y tế AI. Nhiệm vụ của bạn là đọc hình ảnh đơn thuốc/hồ sơ khám bệnh và trích xuất thông tin.
    
    Yêu cầu số 1 (Kiểm tra chất lượng ảnh):
    - Đánh giá xem ảnh có quá mờ, bị cắt xén mất thông tin, hoặc hoàn toàn không phải là đơn thuốc/hồ sơ y tế hay không.
    - Nếu ảnh KHÔNG THỂ đọc được, hãy đặt "is_readable": false và ghi rõ lý do vào "error_message" (Ví dụ: "Hình ảnh đơn thuốc quá mờ hoặc bị lóa sáng, vui lòng chụp lại rõ nét hơn."). Các trường khác có thể để trống.
    
    Yêu cầu số 2 (Trích xuất thông tin nếu ảnh đọc được):
    - "diagnosis": Chẩn đoán bệnh của bác sĩ.
    - "source": Luôn luôn trả về chuỗi "image_ocr".
    - "medications": Danh sách các loại thuốc. Với mỗi loại thuốc, trích xuất:
        + "name": Tên thuốc phải được chuẩn hóa gọn gàng. Cấu trúc chuẩn: Tên gốc + Hàm lượng + (Tên thương mại/biệt dược nếu có). Ví dụ: "Paracetamol 500mg (Partamol Tab)". Không được để quá dư thừa.
        + "duration_days": Số ngày uống (kiểu số nguyên). BẮT BUỘC tính toán dựa trên tổng số lượng chia cho số lượng dùng mỗi ngày (Ví dụ: Tổng 30 viên, ngày uống 2 viên -> duration_days = 15). Dữ liệu này dùng để tự động expire hồ sơ khám bệnh. Nếu hoàn toàn không có thông tin, để null.
        + "note": Lời dặn cụ thể (Ví dụ: "Uống sau ăn no", "Ngậm dưới lưỡi").
        + "schedules": Danh sách CÁC KHUNG GIỜ uống thuốc trong ngày. Bạn phải TỰ ĐỘNG QUY ĐỔI các chữ như "Sáng", "Trưa", "Chiều", "Tối" thành giờ chuẩn (Format HH:mm).
             Quy ước quy đổi: Sáng -> "08:00", Trưa -> "12:00", Chiều -> "17:00", Tối -> "20:00".
             Mỗi khung giờ kèm theo "dosage" (liều lượng). Ví dụ: "Sáng 1 viên, Tối 1 viên" -> [ {"time": "08:00", "dosage": "1 viên"}, {"time": "20:00", "dosage": "1 viên"} ].
    
    Hãy trả về ĐÚNG định dạng JSON sau, không kèm theo bất kì chữ nào khác ngoài JSON:
    {
      "is_readable": true,
      "error_message": "",
      "diagnosis": "Chẩn đoán bệnh",
      "source": "image_ocr",
      "medications": [
        {
          "name": "Paracetamol 500mg (Partamol Tab)",
          "duration_days": 5,
          "schedules": [
            { "time": "08:00", "dosage": "1 viên" },
            { "time": "20:00", "dosage": "1 viên" }
          ],
          "note": "Uống sau khi ăn"
        }
      ]
    }
    """
    try:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        response = _generate_with_model_fallback(
            model_candidates=ANALYSIS_MODELS,
            contents=[image_part, prompt],
            temperature=0.2,
        )
        
        text_response = response.text
        # Cleanup json markdown codeblocks if any
        if text_response.startswith("```json"):
            text_response = text_response.replace("```json", "", 1)
            if text_response.endswith("```"):
                text_response = text_response[:-3]
        elif text_response.startswith("```"):
            text_response = text_response.replace("```", "", 1)
            if text_response.endswith("```"):
                text_response = text_response[:-3]
        return text_response.strip()

    except Exception as e:
        # Trả về JSON lỗi nếu thất bại để client không bị crash
        return json.dumps({
            "is_readable": False,
            "error_message": f"Lỗi xử lý AI: {str(e)}",
            "diagnosis": "",
            "source": "image_ocr",
            "medications": []
        })

def generate_health_forecast(user_data: dict) -> str:
    lang = user_data.get("language", "vi")
    
    if lang == "vi":
        lang_instruction = "Viết toàn bộ phản hồi bằng tiếng Việt (bao gồm cả headline, insight, và recommendations)."
        headline_desc = "Một câu tóm tắt ngắn gọn trạng thái sức khỏe bằng tiếng Việt (Ví dụ: 'Phát hiện dấu hiệu rối loạn chuyển hóa' hoặc 'Chỉ số tim mạch ổn định')"
        insight_desc = "Đoạn phân tích ngắn gọn (3-4 câu) bằng tiếng Việt. Phải tập trung trực tiếp và duy nhất vào chẩn đoán/dự đoán nguy cơ thuộc 2 nhóm bệnh: Rối loạn chuyển hóa (béo phì, đái tháo đường/tiểu đường, gút, mỡ máu) và Tim mạch (huyết áp, nhịp tim). Không phân tích lan man sang chủ đề khác."
        workout_desc = "Gợi ý 1 kế hoạch tập luyện phù hợp bằng tiếng Việt (Ví dụ: '30 phút đi bộ nhanh cải thiện tim mạch')"
        diet_desc = "Gợi ý chế độ ăn phù hợp bằng tiếng Việt (Ví dụ: 'Hạn chế carbohydrate nhanh để ổn định đường huyết')"
    else:
        lang_instruction = "Write the entire response in English."
        headline_desc = "A short headline in English summarizing the health state (e.g., 'Metabolic Strain Risk' or 'Cardiovascular Health Stable')"
        insight_desc = "A detailed analysis in English (3-4 sentences). Must focus directly and exclusively on diagnosing/predicting risks related to two disease groups: Metabolic disorders (obesity, diabetes, gout, dyslipidemia) and Cardiovascular diseases (hypertension, resting heart rate). Do not drift into other health topics."
        workout_desc = "A workout suggestion in English (e.g., '30 mins brisk walking for cardiovascular health')"
        diet_desc = "A diet recommendation in English (e.g., 'Limit simple carbohydrates to stabilize blood sugar')"

    prompt = f"""
    Bạn là một chuyên gia y tế và phân tích dữ liệu sức khỏe AI (AI Health Forecaster).
    Nhiệm vụ của bạn là phân tích các chỉ số sức khỏe, thói quen sinh hoạt và hồ sơ bệnh án của người dùng để đưa ra đánh giá hiện tại và dự đoán rủi ro/cơ hội trong tương lai.

    ĐẶC BIỆT: Bạn phải khoanh vùng phân tích chẩn đoán trực tiếp vào 2 nhóm bệnh lý chính sau, không phân tích chung chung hay lan man:
    1. Rối loạn chuyển hóa (như thừa cân, béo phì, tiểu đường/đái tháo đường, gút, rối loạn mỡ máu).
    2. Tim mạch (như tăng huyết áp, nhịp tim nhanh/chậm bất thường, suy tim, bệnh mạch vành).

    Dữ liệu người dùng:
    {json.dumps(user_data, ensure_ascii=False, indent=2)}

    Yêu cầu ĐẦU RA (phải trả về ĐÚNG định dạng JSON sau, không kèm bất kỳ văn bản nào khác. {lang_instruction}):
    {{
        "health_score": [Điểm sức khỏe tổng quát từ 0 - 100],
        "status": "[GOOD hoặc AT_RISK. GOOD nếu điểm >= 75, ngược lại là AT_RISK]",
        "headline": "[{headline_desc}]",
        "insight": "[{insight_desc}]",
        "metrics": {{
            "sleep_score": [Điểm chất lượng giấc ngủ 0-100],
            "stress_index": "[Low, Medium, hoặc High]"
        }},
        "recommendations": {{
            "workout": "[{workout_desc}]",
            "diet": "[{diet_desc}]"
        }}
    }}
    """
    try:
        response = _generate_with_model_fallback(
            model_candidates=ANALYSIS_MODELS,
            contents=prompt,
            temperature=0.3,
        )
        
        text_response = response.text
        if text_response.startswith("```json"):
            text_response = text_response.replace("```json", "", 1)
            if text_response.endswith("```"):
                text_response = text_response[:-3]
        elif text_response.startswith("```"):
            text_response = text_response.replace("```", "", 1)
            if text_response.endswith("```"):
                text_response = text_response[:-3]
        return text_response.strip()

    except Exception as e:
        return json.dumps({
            "health_score": 50,
            "status": "AT_RISK",
            "headline": "System Error",
            "insight": f"Unable to analyze data at the moment. Error: {str(e)}",
            "metrics": {"sleep_score": 0, "stress_index": "Unknown"},
            "recommendations": {"workout": "Rest", "diet": "Stay hydrated"}
        })