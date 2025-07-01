from dotenv import load_dotenv
import os

load_dotenv()

print("FLASK_SECRET_KEY =", os.getenv("FLASK_SECRET_KEY"))
print("ADMIN_PASSWORD =", os.getenv("ADMIN_PASSWORD"))
