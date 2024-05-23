FROM python:3.10.11

# Thư mục làm việc
WORKDIR /chat_pdf_code

# Sao chép requirements.txt và cài đặt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép code vào container
COPY . .

# Mở port Streamlit mặc định 8501
EXPOSE 8501

# Chạy ứng dụng
CMD ["streamlit", "run", "main.py"] 
