# Book Recommendation System

A Streamlit application that recommends books based on text similarity. The app combines book title, author, genres, description, and series name, converts the text into TF-IDF vectors, and ranks books using cosine similarity.

## Features

- Select a book from the dataset
- Choose between 1 and 10 recommendations
- View recommended titles, authors, and similarity scores
- Display recommendations in an interactive table

## Project Structure

```text
book_recommendation/
|-- app.py
|-- books_dataset.csv
|-- requirement.txt
`-- README.md
```

## Requirements

- Python 3.9 or newer
- pip

Python packages:

- Streamlit
- Pandas
- NLTK
- scikit-learn

## Setup

1. Open a terminal and navigate to the project directory:

```powershell
cd "book_recommendation"
```

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install the dependencies:

```powershell
python -m pip install -r requirement.txt
```

## Run the Application

From the project directory, run:

```powershell
python -m streamlit run app.py
```

Streamlit will provide a local URL, normally `http://localhost:8501`.

## How It Works

1. The application loads `books_dataset.csv`.
2. Missing values in the text fields are replaced with empty strings.
3. Title, author, genres, description, and series name are combined into a search text.
4. Text is cleaned and converted into TF-IDF feature vectors.
5. Cosine similarity compares the selected book with all books in the dataset.
6. The application excludes the selected book and shows the most similar results.

## Data

The application uses `books_dataset.csv`. It expects the following fields:

- `title`
- `author`
- `genres`
- `description`
- `series_name`
