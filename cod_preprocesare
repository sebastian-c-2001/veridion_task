import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# Asigură-te că ai descărcat resursele necesare
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# Inițializează lematizatorul și lista de stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Inițializează modelul Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def preprocess_text(text):
    if pd.isna(text):  # Verifică dacă e NaN
        return ""
    text = text.lower()  # Transformă în litere mici
    text = re.sub(r"[^a-z0-9 ]", "", text)  # Elimină caractere speciale
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization și eliminare stopwords
    return " ".join(words)

# Încarcă fișierul CSV
file_path = "date.csv"  # Schimbă cu calea fișierului tău
df = pd.read_csv(file_path)

# Aplică preprocesarea pe toate coloanele textuale
df = df.applymap(lambda x: preprocess_text(x) if isinstance(x, str) else x)

# Aplică Sentence Transformer pentru embedding-uri
df["embedding"] = df.apply(lambda row: model.encode(" ".join(row.astype(str))), axis=1)

# Salvează rezultatul
output_path = "preprocessed_file.csv"
df.to_csv(output_path, index=False)
print(f"Fișierul preprocesat și embed-uit a fost salvat ca {output_path}")
