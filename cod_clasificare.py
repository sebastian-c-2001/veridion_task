import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Inițializare model embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Citire date companii și taxonomie
company_df = pd.read_csv("date.csv")  # Se lucrează direct pe `date.csv`
taxonomy_df = pd.read_csv("insurance_taxonomy.csv")

# Generare embeddings pentru taxonomie
taxonomy_df["embedding"] = taxonomy_df["label"].apply(lambda x: model.encode(x))
taxonomy_embeddings = np.vstack(taxonomy_df["embedding"])

# Setare ponderi pentru fiecare atribut
weights = {
    # "category": 0.2,  # Cel mai important
    # "niche": 0.2,
    # "description": 0.2,
    # "sector": 0.1,
    # "business_tags": 0.3  # Cel mai puțin important

    "category": 0.2,  # Cel mai important
    "niche": 0.2,
    "description": 0.4,
    "sector": 0.2,
    "business_tags": 0  # Cel mai puțin important
}

# Funcție pentru generarea embedding-urilor ponderate
def compute_weighted_embedding(row):
    components = []
    total_weight = 0

    for col, weight in weights.items():
        if pd.notna(row[col]):
            embedding = model.encode(str(row[col]))
            components.append(weight * embedding)
            total_weight += weight

    return np.sum(components, axis=0) / total_weight if components else np.zeros(384)  # 384 = dimensiune embeddings

# Aplicăm pe toate companiile cu o bară de progres
company_embeddings = []
for _, row in tqdm(company_df.iterrows(), total=len(company_df), desc="Generating embeddings"):
    company_embeddings.append(compute_weighted_embedding(row))

company_embeddings = np.vstack(company_embeddings)

# Atribuire etichetă pe baza similarității cosinus
def assign_label(company_embedding):
    similarities = cosine_similarity([company_embedding], taxonomy_embeddings)
    best_match_idx = np.argmax(similarities)
    return taxonomy_df.iloc[best_match_idx]["label"], similarities[0, best_match_idx]

assigned_labels_and_scores = [assign_label(embedding) for embedding in tqdm(company_embeddings, desc="Assigning labels")]

# Adăugăm rezultatele în `date.csv`
company_df["assigned_label"] = [x[0] for x in assigned_labels_and_scores]

# Salvăm `date.csv` cu noua coloană
company_df.to_csv("date.csv", index=False)

# 🔹 Distribuția claselor atribuite
taxonomy_distribution = company_df["assigned_label"].value_counts().reset_index()
taxonomy_distribution.columns = ["label", "count"]
taxonomy_distribution.to_csv("taxonomy_distribution.csv", index=False)

# 🔹 Calcul acuratețe medie pe baza scorului de similaritate cosinus
mean_similarity_score = np.mean([x[1] for x in assigned_labels_and_scores])
print(f"✔ Acuratețea medie bazată pe similaritatea cosinus: {mean_similarity_score:.4f}")

# 🔹 Analiza distribuției scorurilor de similaritate
similarity_scores = [x[1] for x in assigned_labels_and_scores]
similarity_stats = pd.Series(similarity_scores).describe()
print("✔ Distribuția scorurilor de similaritate:")
print(similarity_stats)

print("✔ Toate fișierele au fost actualizate cu succes!")
