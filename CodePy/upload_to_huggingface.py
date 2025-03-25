# upload_to_hf.py
# Script basique pour uploader un modèle sur Hugging Face

import os
import argparse
from huggingface_hub import login, upload_folder, create_repo


def main():
    # Analyser les arguments
    parser = argparse.ArgumentParser(description="Upload d'un modèle vers Hugging Face")
    parser.add_argument("--model_dir", type=str, default="optimized_lora_model",
                        help="Dossier contenant le modèle à uploader")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="Nom du dépôt sur Hugging Face (format: votre-nom/nom-du-modele)")
    parser.add_argument("--token", type=str, required=True,
                        help="Token d'accès Hugging Face (obtenir sur https://huggingface.co/settings/tokens)")
    parser.add_argument("--private", action="store_true",
                        help="Rendre le dépôt privé")

    args = parser.parse_args()

    # Vérifier que le dossier du modèle existe
    if not os.path.exists(args.model_dir):
        print(f"❌ Erreur: Le dossier {args.model_dir} n'existe pas")
        exit(1)

    # Vérifier le format du nom du repo
    if "/" not in args.repo_name or len(args.repo_name.split("/")) != 2:
        print("❌ Erreur: Le nom du dépôt doit être au format 'votre-nom/nom-du-modele'")
        exit(1)

    # S'authentifier avec le token
    print(f"🔑 Connexion à Hugging Face avec le token...")
    login(token=args.token)

    # Créer le dépôt si nécessaire
    print(f"📁 Création du dépôt {args.repo_name}...")
    try:
        create_repo(
            repo_id=args.repo_name,
            exist_ok=True,
            private=args.private
        )
    except Exception as e:
        print(f"⚠️ Note: {e}")
        print("Tentative d'upload dans un dépôt existant...")

    # Uploader le modèle
    print(f"📤 Upload du modèle vers {args.repo_name}...")
    try:
        result = upload_folder(
            folder_path=args.model_dir,
            repo_id=args.repo_name,
            commit_message="Upload du modèle LoRA pour génération de code React",
            ignore_patterns=["*.log", "checkpoint-*", "*~", "__pycache__"],
        )
        print(f"✅ Upload terminé avec succès!")
        print(f"🔗 URL du modèle: https://huggingface.co/{args.repo_name}")
    except Exception as e:
        print(f"❌ Erreur lors de l'upload: {e}")
        exit(1)


if __name__ == "__main__":
    main()