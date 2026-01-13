FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    requests \
    flask \
    python-dotenv

COPY telegram_alert_bot.py .
COPY .env .

EXPOSE 8080

CMD ["python", "telegram_alert_bot.py"]