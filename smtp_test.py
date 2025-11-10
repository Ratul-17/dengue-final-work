# smtp_test.py
import ssl, smtplib
from email.message import EmailMessage

HOST = "smtp.gmail.com"
PORT = 465  # SSL
USER = "rafkhata.18349@gmail.com"   # e.g. rafkhata18349@gmail.com
PWD  = "xlaczqolqutfiplx"  # no spaces required
TO   = "islamariful384@gmail.com"  # can be same as USER for testing

msg = EmailMessage()
msg["Subject"] = "SMTP Test — Dengue Allocation"
msg["From"] = USER
msg["To"] = TO
msg.set_content("If you see this, SMTP with Gmail App Password works.")

context = ssl.create_default_context()
try:
    with smtplib.SMTP_SSL(HOST, PORT, context=context, timeout=20) as s:
        s.login(USER, PWD)
        s.send_message(msg)
    print("✅ Test email sent. Check your inbox (and Spam).")
except smtplib.SMTPAuthenticationError as e:
    print("❌ AUTH failed. Use a Gmail App Password and correct Gmail address.")
    print(e)
except Exception as e:
    print("❌ Other error:", type(e).__name__, e)
