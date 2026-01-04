Ensure the following are installed on your system:

1.Python 3.9+

2.pip

3.virtualenv (or venv module)

4.Git


project-root/
│── app/                    # FastAPI application
│── config/
│   └── serviceAccountKey.json   # Firebase service account key
│── .env                    # Environment variables (Gemini API key)
│── requirements.txt
│── .gitignore
│── README.md

1.Go to Firebase Console → Project Settings → Service Accounts

2.Generate a new Service Account Key

3.Place the downloaded JSON file here:

config/serviceAccountKey.json

Environment Variables

Create a .env file in the project root and add your Gemini API key:
GEMINI_API_KEY=your_gemini_api_key_here

First, create a virtual environment:

First, create a virtual environment:

On Windows
python -m venv venv
venv\Scripts\activate

On macOS / Linux
python3 -m venv venv
source venv/bin/activate


After activating the virtual environment, install required packages:

pip install -r requirements.txt

▶️ Run the FastAPI Server

Start the development server using Uvicorn:

uvicorn fastapi:app --reload

*  API Base URL:

http://127.0.0.1:8000


*  Interactive API Docs (Swagger UI):

http://127.0.0.1:8000/docs


*  ReDoc Documentation:

http://127.0.0.1:8000/redoc
