# Requirements: UI học tiếng Hoa Đài Loan (MVP)

## Mục tiêu

Người lớn tự học có thể đăng nhập, tạo hồ sơ, chọn sách và bài, hoàn thành một hoạt động nghe/đọc/nói, rồi thấy kết quả và bước học tiếp theo trong hồ sơ. Giao diện dùng được trên điện thoại và máy tính.

## Luồng và màn hình

| Ưu tiên | Màn hình | Yêu cầu tối thiểu |
| --- | --- | --- |
| P0 | Trang công khai | Giải thích ngắn ba kỹ năng, nội dung hiện có, nút đăng nhập Google và liên kết quyền riêng tư. |
| P0 | Onboarding / Cài đặt hồ sơ | Tên hiển thị, ngôn ngữ giao diện (zh-TW, en, vi, id, hi), mục tiêu học, số ngày học mỗi tuần (mặc định 3). Cho sửa lại sau. |
| P0 | Thư viện | Chọn Câu trong sách → Bài 5–8 → câu chuyện → Nghe/Nói/Đọc; chọn Từ và cụm từ → Chương 5 → Nói. Tìm và tiếp tục bài đang học. |
| P0 | Nghe | Phát audio theo cảnh, điều chỉnh tốc độ, trả lời 3 câu hỏi. Sau khi nộp, hiện đáp án, giải thích, lời thoại, pinyin và bản dịch. |
| P0 | Đọc | Hiện đoạn chữ Hoa và 3 câu hỏi. Sau khi nộp, hiện đáp án, giải thích, pinyin, bản dịch và audio. |
| P0 | Nói | Nghe mẫu, hiệu chỉnh giọng, ghi âm, nghe lại, gửi chấm, xem phản hồi phát âm; cho luyện lại. |
| P0 | Hồ sơ | Ba thẻ tiến độ Nghe/Nói/Đọc, bài tiếp tục, kết quả gần đây, nhận xét phát âm và mục tiêu tuần. Phân biệt điểm gần nhất với điểm cao nhất. |
| P0 | Quản lý tài khoản | Đăng xuất, tải dữ liệu, xóa tài khoản. Bảng xếp hạng chỉ hiện biệt danh khi người học bật tham gia. |
| P1 | Thành tích / bảng xếp hạng | Huy hiệu và bảng xếp hạng tự nguyện; không cản trở luồng học chính. |

## Dữ liệu và quy tắc

- Bản đầu: 12 câu chuyện thuộc Bài 5–8 (100 câu), 53 từ/cụm từ Chương 5, và 72 câu hỏi hiểu bài (3 nghe + 3 đọc cho mỗi câu chuyện). Nội dung, câu hỏi và lựa chọn có ID, phiên bản ổn định và bản dịch đủ năm ngôn ngữ.
- Hoàn thành bài nghe/đọc khi nộp câu trả lời; cho làm lại và lưu từng lượt. Tiến độ nghe/đọc là số câu chuyện đã làm trên 12. Tiến độ nói là số mục đã luyện trên 153.
- Hồ sơ thuộc tài khoản Google theo `sub`; tên và email không phải khóa tài khoản. Hồ sơ chọn theo tên của bản cũ không tự nhập vào tài khoản công khai.
- Lượt học, điểm và ghi âm chỉ hiện cho chủ tài khoản. Ngày học tuần tính từ hoạt động đã lưu, theo múi giờ của người học.
- Trước khi nộp không gửi đáp án đúng hoặc giải thích trong payload bài tập.

## Yêu cầu giao diện và trạng thái

- Token màu: nền `#F7F8FC`, thẻ trắng, chữ `#17213D`, chàm `#4F46E5`, bạc hà đậm `#0B786C`; hổ phách dùng cho nhấn nhẹ. Inter cho UI, Noto Sans TC cho chữ Hoa.
- Kiểm tra bố cục ở 390, 768, 1024 và 1440 px; điều hướng, câu hỏi, trình phát và nút ghi âm vẫn thao tác được bằng bàn phím.
- Tương phản tối thiểu WCAG AA, nhãn rõ cho nút/biểu mẫu, thông báo trạng thái cho trình đọc màn hình, tôn trọng giảm chuyển động.
- Có trạng thái chờ, trống, lỗi mạng, phiên hết hạn, audio thiếu, không cấp quyền micro, và máy chấm nói bận (`429`); giữ câu trả lời chưa nộp khi có lỗi tạm thời.

## Hợp đồng backend và điều kiện phát hành

- Repo này cung cấp API phiên đăng nhập/hồ sơ, danh mục và bài học, câu hỏi và nộp bài, audio, dashboard và hoạt động, xuất/xóa dữ liệu. Dịch vụ `D:\TaiwanMandarinAI` chỉ phụ trách chấm nói. Server website ánh xạ hồ sơ bằng ID nội bộ, gọi dịch vụ chấm nói qua loopback với `save_attempt=false`, rồi tự lưu kết quả và ghi âm; trình duyệt chỉ gọi API học đã xác thực.
- Google OIDC, cookie `HttpOnly`/`Secure`, CSRF và kiểm tra quyền sở hữu phải chạy trên backend HTTPS. Local preview login chỉ dùng trên loopback.
- Biên tập viên rà soát câu hỏi, lựa chọn và giải thích; xác nhận quyền xuất bản sách và audio. Máy chủ cần dữ liệu sách/audio, MFA, FFmpeg, kho SQLite/ghi âm bền vững và sao lưu hằng ngày.

## Thứ tự hoàn tất và nghiệm thu

1. Kiểm tra hành trình Google → hồ sơ → thư viện trên API học của repo này; chạy hợp đồng chấm nói qua HTTP với dịch vụ riêng.
2. Kiểm tra một bài nghe, đọc và nói trên dữ liệu thật; thử nộp lại và xem điểm gần nhất/điểm cao nhất trong hồ sơ.
3. Rà soát bản dịch, nội dung và toàn bộ trạng thái lỗi; kiểm tra quyền riêng tư, truy cập chéo tài khoản và tải/xóa dữ liệu.
4. Kiểm tra bàn phím, trình đọc màn hình, bốn kích thước màn hình và build production; phát hành sau khi Google callback và sao lưu hoạt động.

**Tiêu chí đạt MVP:** người học mới hoàn tất ba kỹ năng trên một câu chuyện và một mục từ, quay lại thấy tiến độ đúng trong hồ sơ, không xem được dữ liệu của người khác, và xử lý được lỗi micro/audio/phiên đăng nhập mà không kẹt luồng học.

**Rà soát nội dung trước phát hành:** `public_curriculum.py` hiện tạo 72 câu hỏi dạng chọn nghĩa từ bản dịch sẵn có. Đây là nội dung thử luồng, chưa phải 72 câu hỏi hiểu bài được biên soạn và duyệt theo kế hoạch. Đặc biệt bài Đọc ở ngôn ngữ `zh-TW` có lựa chọn trùng với câu chữ Hoa đang hiển thị. Cần biên soạn từng câu chuyện, kiểm tra đủ năm bản dịch và cho người phụ trách nội dung duyệt trước khi công bố.
