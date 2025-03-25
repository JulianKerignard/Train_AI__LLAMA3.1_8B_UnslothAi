# Enregistrez ce script sous check_data.py et exécutez-le
# python check_data.py

import os

# Chemin vers le dossier prepared_data
data_dir = "D:\\PythonProject\\CodePy\\optimized_lora_model\\prepared_data"

print(f"Contenu du dossier {data_dir}:")
for item in os.listdir(data_dir):
    item_path = os.path.join(data_dir, item)
    if os.path.isdir(item_path):
        print(f"  📁 {item} (dossier)")
        # Afficher les premiers fichiers du sous-dossier
        subfiles = os.listdir(item_path)[:5]  # Limiter à 5 fichiers pour clarté
        for subfile in subfiles:
            print(f"    - {subfile}")
        if len(os.listdir(item_path)) > 5:
            print(f"    - ... et {len(os.listdir(item_path))-5} autres fichiers")
    else:
        print(f"  📄 {item} (fichier)")