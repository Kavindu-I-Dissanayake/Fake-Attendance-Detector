from pymongo import MongoClient
from .config import MONGO_URI, MONGO_DB

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

students_col = db["students"]
sessions_col = db["sessions"]
attendance_records_col = db["attendance_records"]

# Indexes (safe to call repeatedly)
students_col.create_index([("className", 1), ("studentId", 1)], unique=True)
sessions_col.create_index([("sessionId", 1)], unique=True)
attendance_records_col.create_index([("sessionId", 1), ("studentId", 1)], unique=True)
