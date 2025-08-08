# ID Photo Background Manager

A web-based tool for removing and replacing backgrounds in ID photos with AI-powered processing.

## Features

- AI-powered background removal using rembg
- Custom background replacement with color or image
- Drag-and-drop image upload interface
- Real-time preview of modifications
- Multiple export formats (PNG, JPEG)
- Error handling and progress feedback
- Image caching for improved performance

## Tech Stack

- Backend: FastAPI (Python 3.9+)
- Frontend: React
- Key Libraries: rembg, Pillow, NumPy

## Installation

### Backend Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Build the frontend:
```bash
npm run build
```

## Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## API Endpoints

- `POST /api/upload`: Upload an image file
- `POST /api/remove-background`: Remove background from image
- `POST /api/replace-background`: Replace background with color or image
- `GET /api/processed/{image_id}`: Get processed image

## Configuration

Configuration settings can be modified in:
- Backend: `backend/core/config.py`
- Frontend: Environment variables in `.env` file

## Security Features

- Input validation for file types
- File size limits
- Rate limiting for API endpoints
- Secure file handling

## Performance Optimizations

- Image compression before processing
- Caching of processed results
- Lazy loading of components
- Progressive image loading

## Error Handling

The application includes comprehensive error handling:
- Graceful fallbacks for failed processing
- User-friendly error messages
- Logging of processing errors
- Retry mechanisms for failed operations

## License

MIT License