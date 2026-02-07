# Segment Anything (SAM2) – Full Stack Demo

This project is a full-stack implementation of Meta’s Segment Anything (SAM2) model with:

- 🖥️ Frontend: Next.js / React  
- 🧠 Backend: FastAPI + SAM2  
- 🎥 Supports image/video segmentation  

---

## 🚀 Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/varun-kolluru/Segment_Anything.git
cd Segment_Anything
```

### 2 Setup frontend
```bash
cd frontend
npm install
npm run dev
```

### 3 Setup backend
```bash
cd ../backend

# create virtual environment
python -m venv venv

# activate venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

# install dependencies
pip install -r requirements.txt

# run backend server
uvicorn main:app --host 0.0.0.0 --port 8000
```


