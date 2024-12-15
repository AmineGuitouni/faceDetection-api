# Face Detection and Recognition API

This project is a FastAPI-based web application for facial recognition. It allows you to add users with their facial embeddings and check for matching faces using a database.

## Features
- **Add User**: Add a new user to the database with facial embeddings.
- **Delete User**: Remove a user from the database by ID.
- **Check User**: Identify a user by comparing facial embeddings.
- **Database**: SQLite is used to store user information and embeddings.

## Prerequisites

Before running this application, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required Python packages.

## Usage

Start the server:
```bash
python app.py
```

By default, the application will run on `http://0.0.0.0:8000/`.

### Endpoints

#### Health Check
- **GET** `/`
  - Response: `{ "status": "ok" }`

#### Add User
- **POST** `/add_user`
  - Request Body:
    ```json
    {
      "first_name": "John",
      "last_name": "Doe",
      "images": ["<base64_encoded_image1>", "<base64_encoded_image2>"]
    }
    ```
  - Response: `{ "message": "User added successfully." }`

#### Delete User
- **DELETE** `/delete_user/{user_id}`
  - Path Parameter: `user_id` (integer)
  - Response: `{ "message": "User deleted successfully." }`

#### Check User
- **POST** `/check_user`
  - Request Body:
    ```json
    {
      "image": "<base64_encoded_image>"
    }
    ```
  - Response:
    ```json
    {
      "user_id": 1,
      "first_name": "John",
      "last_name": "Doe",
      "distance": 0.45
    }
    ```