"""
Telegram Alert Bot for Prometheus AlertManager
Forwards alerts to Telegram
"""
import os
# import json
import requests
from flask import Flask, request
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Telegram config
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# Emoji mapping
SEVERITY_EMOJI = {
    'critical': '🔴',
    'warning': '⚠️',
    'info': 'ℹ️'
}

CATEGORY_EMOJI = {
    'reliability': '💀',
    'performance': '⚡',
    'model_quality': '🎯',
    'operability': '🔧',
    'cost': '💰',
    'security': '🔒',
    'scalability': '📈'
}


def format_alert_message(alert):
    """Format alert for Telegram"""
    status = alert.get('status', 'unknown').upper()
    labels = alert.get('labels', {})
    annotations = alert.get('annotations', {})
    
    alertname = labels.get('alertname', 'Unknown')
    severity = labels.get('severity', 'info')
    category = labels.get('category', 'unknown')
    endpoint = labels.get('endpoint', 'unknown')
    
    severity_emoji = SEVERITY_EMOJI.get(severity, '❓')
    category_emoji = CATEGORY_EMOJI.get(category, '📊')
    
    # Status emoji
    if status == 'FIRING':
        status_emoji = '🚨'
    elif status == 'RESOLVED':
        status_emoji = '✅'
    else:
        status_emoji = '❓'
    
    # Build message
    message = f"{status_emoji} *{status}* {severity_emoji} {category_emoji}\n\n"
    message += f"*Alert:* `{alertname}`\n"
    message += f"*Endpoint:* `{endpoint}`\n"
    message += f"*Severity:* {severity}\n"
    message += f"*Category:* {category}\n"
    
    # Add summary
    if 'summary' in annotations:
        message += f"\n📝 *Summary:*\n{annotations['summary']}\n"
    
    # Add description
    if 'description' in annotations:
        message += f"\n💬 *Details:*\n{annotations['description']}\n"
    
    # Add timestamp
    starts_at = alert.get('startsAt', '')
    if starts_at:
        try:
            dt = datetime.fromisoformat(starts_at.replace('Z', '+00:00'))
            message += f"\n⏰ *Time:* {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        except Exception:
            pass
    
    # Add runbook link if exists
    if 'runbook_url' in annotations:
        message += f"\n📖 [Runbook]({annotations['runbook_url']})\n"
    
    return message


def send_telegram_message(message):
    """Send message to Telegram"""
    try:
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown',
            'disable_web_page_preview': True
        }
        
        response = requests.post(TELEGRAM_API_URL, json=payload, timeout=10)
        
        if response.status_code != 200:
            print(f"Failed to send Telegram message: {response.text}")
            return False
        
        return True
    
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False


@app.route('/alert', methods=['POST'])
def webhook():
    """Receive alerts from AlertManager"""
    try:
        data = request.get_json()
        
        if not data:
            return {'status': 'error', 'message': 'No data received'}, 400
        
        alerts = data.get('alerts', [])
        
        if not alerts:
            return {'status': 'ok', 'message': 'No alerts to process'}, 200
        
        # Group alerts by status
        firing_alerts = [a for a in alerts if a.get('status') == 'firing']
        resolved_alerts = [a for a in alerts if a.get('status') == 'resolved']
        
        # Send firing alerts
        for alert in firing_alerts:
            message = format_alert_message(alert)
            send_telegram_message(message)
        
        # Send resolved alerts (grouped)
        if resolved_alerts:
            if len(resolved_alerts) == 1:
                message = format_alert_message(resolved_alerts[0])
                send_telegram_message(message)
            else:
                # Group message for multiple resolved alerts
                message = f"✅ *RESOLVED* - {len(resolved_alerts)} alerts\n\n"
                for alert in resolved_alerts:
                    labels = alert.get('labels', {})
                    message += f"• `{labels.get('alertname', 'Unknown')}` ({labels.get('endpoint', 'unknown')})\n"
                send_telegram_message(message)
        
        return {'status': 'ok', 'processed': len(alerts)}, 200
    
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return {'status': 'error', 'message': str(e)}, 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return {'status': 'healthy'}, 200


@app.route('/test', methods=['GET'])
def test():
    """Test Telegram connection"""
    message = "🧪 *Test Alert*\n\nTelegram bot is working!"
    success = send_telegram_message(message)
    
    if success:
        return {'status': 'ok', 'message': 'Test message sent'}, 200
    else:
        return {'status': 'error', 'message': 'Failed to send test message'}, 500


if __name__ == '__main__':
    print("Starting Telegram Alert Bot...")
    print(f"Bot Token: {TELEGRAM_BOT_TOKEN[:10]}...")
    print(f"Chat ID: {TELEGRAM_CHAT_ID}")
    
    app.run(host='0.0.0.0', port=8080)