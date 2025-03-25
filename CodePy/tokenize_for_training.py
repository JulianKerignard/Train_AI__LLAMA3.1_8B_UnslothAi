# tokenize_for_training.py
# Script qui convertit les données de temp_dataset au format attendu par fast_stable_training.py

import os
import torch
import argparse
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
import gc

# Paramètres configurables
parser = argparse.ArgumentParser(description="Tokenisation des données pour l'entraînement")
parser.add_argument("--input_dir", type=str, default="temp_dataset", help="Dossier contenant les données brutes")
parser.add_argument("--output_dir", type=str, default="optimized_lora_model", help="Dossier de sortie")
parser.add_argument("--start_idx", type=int, default=0, help="Index de départ pour le nom de fichier")
parser.add_argument("--num_examples", type=int, default=70500, help="Nombre d'exemples pour le nom de fichier")
parser.add_argument("--max_length", type=int, default=1000, help="Longueur maximale de séquence")
args = parser.parse_args()


# Fonction pour nettoyer la mémoire
def clean_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("🧹 Mémoire nettoyée")


print(f"\n{'=' * 70}")
print(f"🔄 TOKENISATION DES DONNÉES")
print(f"{'=' * 70}")

# Vérifier si les données sources existent
if not os.path.exists(args.input_dir):
    print(f"❌ Erreur: Le dossier source {args.input_dir} n'existe pas")
    exit(1)

# Créer les dossiers de sortie
output_data_dir = os.path.join(args.output_dir, "prepared_data")
os.makedirs(output_data_dir, exist_ok=True)

# Chemin pour le fichier de données tokenisées
tokenized_data_path = os.path.join(output_data_dir,
                                   f"tokenized_data_{args.start_idx}_{args.start_idx + args.num_examples}")

# Vérifier si les données tokenisées existent déjà
if os.path.exists(tokenized_data_path):
    print(f"⚠️ Les données tokenisées existent déjà dans {tokenized_data_path}")
    overwrite = input("Voulez-vous les écraser? (o/n): ")
    if overwrite.lower() != 'o':
        print("Tokenisation annulée.")
        exit(0)

try:
    # Charger les données brutes
    print(f"Chargement des données depuis {args.input_dir}...")
    raw_dataset = load_from_disk(args.input_dir)
    print(f"Données chargées: {len(raw_dataset)} exemples")

    # Afficher un exemple pour vérification
    print("\nStructure des données chargées:")
    print(raw_dataset.column_names)

    # Vérifier si les données ont le bon format
    if not all(col in raw_dataset.column_names for col in ['instruction', 'output']):
        print("⚠️ Les données n'ont pas le format attendu (instruction, input, output)")
        print("Colonnes trouvées:", raw_dataset.column_names)

    # Créer le template pour la tokenisation
    print("Préparation du template pour tokenisation...")


    # Fonction pour combiner instruction et output dans un seul texte
    def combine_fields(example):
        template = f"""Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
        return {"text": template}


    # Appliquer la transformation
    print("Application du template...")
    formatted_dataset = raw_dataset.map(combine_fields)

    # Charger le tokenizer
    print("Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
    tokenizer.pad_token = tokenizer.eos_token


    # Fonction de tokenisation
    def tokenize_function(example):
        # Tokeniser le texte
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Convertir les tenseurs en listes
        return {
            "input_ids": tokens["input_ids"][0].tolist(),
            "attention_mask": tokens["attention_mask"][0].tolist(),
            "labels": tokens["input_ids"][0].tolist()  # Pour causal LM
        }


    # Tokeniser en traitant les exemples par lots
    print(f"Tokenisation des données (max_length={args.max_length})...")

    tokenized_examples = []
    batch_size = 1000
    for i in range(0, len(formatted_dataset), batch_size):
        end_idx = min(i + batch_size, len(formatted_dataset))
        print(f"  Tokenisation des exemples {i} à {end_idx}...")

        batch = formatted_dataset.select(range(i, end_idx))
        for example in batch:
            tokenized_examples.append(tokenize_function(example))

        # Libérer la mémoire après chaque lot
        del batch
        clean_memory()

    # Créer le dataset tokenisé
    tokenized_dataset = Dataset.from_list(tokenized_examples)
    print(f"Données tokenisées: {len(tokenized_dataset)} exemples")

    # Sauvegarder les données tokenisées
    print(f"Sauvegarde des données tokenisées dans {tokenized_data_path}...")
    tokenized_dataset.save_to_disk(tokenized_data_path)

    print(f"\n✅ TOKENISATION TERMINÉE")
    print(f"Les données sont prêtes pour l'entraînement!")
    print(f"Vous pouvez maintenant lancer:")
    print(f"python fast_stable_training.py")

except Exception as e:
    print(f"❌ Erreur lors de la tokenisation: {e}")
    exit(1)