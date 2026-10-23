# Quy tắc frontend

Áp dụng cho mã React trong `frontend/`. Đây là chuẩn cho phần mã mới và các phần được sửa; tách dần các màn hình cũ trong `App.jsx` khi chạm vào chúng. Backend chính thuộc dịch vụ `D:\TaiwanMandarinAI`; frontend giao tiếp qua HTTP API và không đọc trực tiếp SQLite hay ghi âm.

## 1. Tổ chức mã

- Một màn hình phụ trách một luồng người học; thành phần dùng lại (trình phát, ghi âm, phản hồi, thẻ tiến độ) đặt trong file riêng. Tránh thêm luồng lớn vào `App.jsx`.
- Tách dữ liệu, trạng thái và giao diện: hàm gọi API chung xử lý URL, cookie, lỗi; component hiển thị không biết cấu trúc lưu trữ backend.
- Dùng tên dễ hiểu theo nghiệp vụ (`story`, `scene`, `attempt`, `profile`); không dùng `data1`, `temp` cho state lâu dài.
- Không thay đổi hợp đồng API bằng suy đoán. Khi thêm trường hoặc endpoint, ghi rõ request, response, lỗi và quyền truy cập trong tài liệu trước khi phụ thuộc vào nó.

## 2. Hệ thống thiết kế và responsive

- Dùng token màu hiện có: nền `#F7F8FC`, chữ `#17213D`, chàm `#4F46E5`, bạc hà `#0B786C`; không thêm màu tùy ý cho cùng một ý nghĩa. Trạng thái đúng/sai/chưa rõ có màu và chữ hoặc biểu tượng, không chỉ màu.
- Giữ thang khoảng cách, bo góc và kiểu chữ nhất quán. Inter cho UI, Noto Sans TC cho chữ Hoa. Không dùng font nhỏ để nhồi nội dung học.
- Thiết kế cho 390, 768, 1024 và 1440 px. Ở màn hình nhỏ, hành động chính và tiến độ bài luôn nhìn thấy hoặc tìm thấy dễ dàng; không có cuộn ngang ngoài vùng nội dung chủ đích.
- Mọi thành phần tương tác có trạng thái mặc định, hover, focus, disabled, đang xử lý và lỗi. Chỉ dùng chuyển động khi nó giúp hiểu thay đổi; tôn trọng `prefers-reduced-motion`.

## 3. Accessibility và ngôn ngữ

- Dùng HTML đúng ngữ nghĩa: `button` cho hành động, `a` cho điều hướng, `fieldset/legend` cho nhóm câu hỏi; mọi input có nhãn. Điều hướng và hoàn thành bài phải dùng được bằng bàn phím.
- Focus phải nhìn rõ; hộp thoại cần quản lý focus; thông báo lưu/nộp/lỗi dùng `role="status"` hoặc `role="alert"` phù hợp. Đạt tương phản WCAG AA cho chữ và điều khiển.
- Chuỗi người học nhìn thấy phải qua bảng dịch cho `zh-TW`, `en`, `vi`, `id`, `hi`; không thêm chuỗi tiếng Anh hoặc tiếng Việt rải trong JSX. Chữ Hoa của bài học lấy từ dữ liệu, có pinyin và bản dịch theo ngôn ngữ hồ sơ.
- Không tự diễn giải mức chấm nói thành điểm phần trăm nếu API chỉ trả nhãn hoặc số đếm. Hiển thị kết quả gần nhất và tốt nhất với đơn vị thật của API.

## 4. API, trạng thái và quyền riêng tư

- Mọi request dùng cookie phiên (`credentials: 'include'`); yêu cầu ghi dùng CSRF. Không lưu token đăng nhập trong `localStorage` và không gửi `student_id` từ trình duyệt để xác định chủ dữ liệu.
- Mỗi màn hình có trạng thái tải, rỗng, thành công và lỗi kèm cách thử lại. Lỗi mạng không xóa đáp án hoặc bản ghi chưa nộp. `401` đưa về đăng nhập; `429` giải thích máy chấm đang bận và cho thử lại.
- Câu hỏi trước khi nộp không chứa đáp án đúng hoặc giải thích. Mỗi lần nộp cho phản hồi rõ ràng; làm lại tạo lượt mới và cập nhật hồ sơ từ dữ liệu server.
- Bản ghi micro chỉ gửi khi người học bấm chấm; dừng track khi ngừng ghi, đổi bài hoặc rời màn hình. Thu hồi URL blob khi không dùng. Xóa tài khoản phải yêu cầu xác nhận rõ ràng.
- Không hiện email trên bảng xếp hạng; chỉ dùng biệt danh khi người học tự bật tham gia. Giao diện tải/xóa dữ liệu phải truy cập được từ hồ sơ.

## 5. Chất lượng và điều kiện merge

- `npm run build` phải đạt. Kiểm tra thủ công luồng khách → đăng nhập → hồ sơ → thư viện → nghe/đọc/nói → kết quả → hồ sơ ở desktop và mobile khi thay đổi luồng.
- Kiểm tra lỗi micro, audio thiếu, mạng mất, phiên hết hạn và máy chấm bận khi thay đổi các phần liên quan. Kiểm tra focus và bàn phím cho thành phần tương tác mới.
- Test tự động cho quy tắc nghiệp vụ hoặc lỗi từng xảy ra; tránh test chỉ lặp lại cấu trúc JSX/CSS. Không phát hành với dữ liệu câu hỏi chưa biên tập hoặc API backend chưa được đối chiếu.
- PR mô tả hành vi đổi, ảnh hoặc video nếu đổi bố cục, cách kiểm chứng và rủi ro còn lại. Không commit `dist/`, `node_modules/`, dữ liệu người học hoặc ghi âm.

## Việc cần làm tiếp để tuân thủ đầy đủ

1. Tách `App.jsx` thành các màn hình, hook và API client nhỏ; chuẩn hóa xử lý `401`/`429`.
2. Chuyển chuỗi còn viết trực tiếp trong JSX (trang quyền riêng tư, hướng dẫn hiệu chỉnh) vào bảng dịch năm ngôn ngữ.
3. Quản lý vòng đời blob URL và stream micro khi đổi trang; bổ sung kiểm tra browser cho ghi âm trên mobile.
4. Đối chiếu toàn bộ API UI đang gọi với dịch vụ backend riêng trước khi triển khai production.
