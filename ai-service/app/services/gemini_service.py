import google.generativeai as genai
import os

# lấy API key
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Không tìm thấy GEMINI_API_KEY")

# cấu hình API
genai.configure(api_key=API_KEY)

# tạo model
model = genai.GenerativeModel('gemini-flash-latest')

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
        response = model.generate_content(prompt)
        return response.text if response.text else "Không tạo được nội dung"

    except Exception as e:
        return f"Lỗi Gemini: {str(e)}"

def analyze_medical_record_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    prompt = """
    Bạn là một bác sĩ hỗ trợ phân tích hồ sơ khám bệnh/đơn thuốc. Hãy đọc hình ảnh đơn thuốc/hồ sơ này và trích xuất ra:
    1. Danh sách các loại thuốc cần uống (tên thuốc, liều dùng, số ngày, lưu ý).
    2. Danh sách các nhóm thức ăn, món ăn mà bác sĩ yêu cầu kiêng cữ (nếu có, tự suy luận dựa trên bệnh án nếu cần thiết, ví dụ: bệnh đau dạ dày thì kiêng chua cay).
    
    Hãy trả về ĐÚNG định dạng JSON sau, không kèm theo bất kì chữ nào khác ngoài JSON:
    {
      "medications": [
        { "name": "Tên thuốc", "dosage": "Liều dùng", "duration": "Thời gian (ví dụ: 5 ngày)", "note": "Lưu ý" }
      ],
      "forbidden_foods": ["Món 1", "Món 2"],
      "diagnosis": "Chẩn đoán bệnh"
    }
    """
    try:
        parts = [
            {"mime_type": mime_type, "data": image_bytes},
            prompt
        ]
        vision_model = genai.GenerativeModel('gemini-1.5-flash')
        response = vision_model.generate_content(parts)
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
        import json
        # Trả về JSON lỗi nếU thấT bạI đễ client khôNg bị crash
        return json.dumps({
            "error": f"Lỗi Gemini Vision: {str(e)}",
            "medications": [],
            "forbidden_foods": [],
            "diagnosis": "Không thể phân tích hồ sơ"
        })