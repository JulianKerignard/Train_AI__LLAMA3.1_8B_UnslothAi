# partie2_dataset.py
# Cette partie prépare le dataset - exécutez-la après partie1_model.py

from datasets import load_dataset, Dataset
import os

# Chargement du dataset ReactJS avec un nombre très limité d'exemples
print("Chargement du dataset ReactJS...")
ds = load_dataset("Hardik1234/reactjs-train")
train_ds = ds["train"]

# Utiliser une taille très petite pour commencer (augmentez progressivement)
MAX_EXAMPLES = 100000  # Commencez avec 1000, puis essayez 2000, 5000...


def create_small_dataset(train_ds, max_examples):
    # Préparer les structures pour stocker les données
    data_dict = {
        'instruction': [],
        'input': [],
        'output': []
    }

    # Limiter le nombre d'exemples
    total = min(max_examples, len(train_ds))
    print(f"Création d'un dataset avec {total} exemples...")

    # Traiter les exemples par lots de 100 pour économiser la mémoire
    batch_size = 100
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        print(f"Traitement des exemples {start_idx} à {end_idx}...")

        # Traiter chaque exemple dans ce lot
        for i in range(start_idx, end_idx):
            example = train_ds[i]
            data_dict['instruction'].append(
                f"Generate React.js code for file: {example['path']} in repository: {example['repo_name']}"
            )
            data_dict['input'].append("")
            data_dict['output'].append(example['content'])

    # Créer et sauvegarder le dataset
    return Dataset.from_dict(data_dict)


# Créer le dataset
print("Création du dataset...")
modified_ds = create_small_dataset(train_ds, MAX_EXAMPLES)
print(f"Dataset créé avec {len(modified_ds)} exemples")

# Sauvegarder le dataset pour l'utiliser dans l'étape suivante
print("Sauvegarde du dataset...")
modified_ds.save_to_disk("temp_dataset")
print("Dataset sauvegardé!")