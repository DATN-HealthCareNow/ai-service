import os
import json
from google import genai
from google.genai import types

# lấy API key
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Không tìm thấy GEMINI_API_KEY")

# khởi tạo client với SDK mới
client = genai.Client(api_key=API_KEY)

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
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
            )
        )
        return response.text if response.text else "Không tạo được nội dung"

    except Exception as e:
        return f"Lỗi Gemini: {str(e)}"

def analyze_medical_record_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    prompt = """
    Bạn là một bác sĩ và trợ lý y tế AI. Nhiệm vụ của bạn là đọc hình ảnh đơn thuốc/hồ sơ khám bệnh và trích xuất thông tin.
    
    Yêu cầu số 1 (Kiểm tra chất lượng ảnh):
    - Đánh giá xem ảnh có quá mờ, bị cắt xén mất thông tin, hoặc hoàn toàn không phải là đơn thuốc/hồ sơ y tế hay không.
    - Nếu ảnh KHÔNG THỂ đọc được, hãy đặt "is_readable": false và ghi rõ lý do vào "error_message" (Ví dụ: "Hình ảnh đơn thuốc quá mờ hoặc bị lóa sáng, vui lòng chụp lại rõ nét hơn."). Các trường khác có thể để trống.
    
    Yêu cầu số 2 (Trích xuất thông tin nếu ảnh đọc được):
    - "diagnosis": Chẩn đoán bệnh của bác sĩ.
    - "forbidden_foods": Dựa vào bệnh lý và lời dặn, suy luận các món ăn/nhóm thực phẩm cần kiêng cữ (ví dụ: đau dạ dày -> đồ chua cay). Trả về danh sách string.
    - "medications": Danh sách các loại thuốc. Với mỗi loại thuốc, trích xuất:
        + "name": Tên thuốc (kèm hàm lượng nếu có).
        + "duration_days": Số ngày uống (kiểu số nguyên). Nếu không rõ, để null.
        + "note": Lời dặn cụ thể (Ví dụ: "Uống sau ăn no", "Ngậm dưới lưỡi").
        + "schedules": Danh sách CÁC KHUNG GIỜ uống thuốc trong ngày. Bạn phải TỰ ĐỘNG QUY ĐỔI các chữ như "Sáng", "Trưa", "Chiều", "Tối" thành giờ chuẩn (Format HH:mm).
             Quy ước quy đổi: Sáng -> "08:00", Trưa -> "12:00", Chiều -> "17:00", Tối -> "20:00".
             Mỗi khung giờ kèm theo "dosage" (liều lượng). Ví dụ: "Sáng 1 viên, Tối 1 viên" -> [ {"time": "08:00", "dosage": "1 viên"}, {"time": "20:00", "dosage": "1 viên"} ].
    
    Hãy trả về ĐÚNG định dạng JSON sau, không kèm theo bất kì chữ nào khác ngoài JSON:
    {
      "is_readable": true,
      "error_message": "",
      "diagnosis": "Chẩn đoán bệnh",
      "medications": [
        {
          "name": "Tên thuốc",
          "duration_days": 5,
          "schedules": [
            { "time": "08:00", "dosage": "1 viên" },
            { "time": "20:00", "dosage": "1 viên" }
          ],
          "note": "Uống sau khi ăn"
        }
      ],
      "forbidden_foods": ["Món 1", "Món 2"]
    }
    """
    try:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        
        # Dùng gemini-1.5-pro vì flash dường như vẫn bị giới hạn quyền truy cập đa phương thức trên project của bạn
        response = client.models.generate_content(
            model='gemini-1.5-pro',
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                temperature=0.2, # Nhiệt độ thấp để ra JSON chuẩn
            )
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
            "medications": [],
            "forbidden_foods": []
        })