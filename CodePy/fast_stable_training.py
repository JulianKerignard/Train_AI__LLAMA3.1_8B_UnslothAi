# incremental_training.py
# Version optimisée pour l'entraînement incrémental

import os
import torch
import argparse
from datasets import load_from_disk, Dataset
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from datetime import datetime
import time
import shutil


# Analyse des arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement incrémental")
    parser.add_argument("--dataset_folder", type=str, default="tokenized_data_0_5000", help="Dossier des données")
    parser.add_argument("--batch_size", type=int, default=4, help="Taille du batch")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Accumulation")
    parser.add_argument("--max_length", type=int, default=500, help="Longueur max")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rang LoRA")
    parser.add_argument("--max_steps", type=int, default=300, help="Étapes max")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Taux d'apprentissage")
    parser.add_argument("--output_dir", type=str, default="optimized_lora_model", help="Dossier de sortie")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Dossier de sauvegarde (pour ne pas écraser le modèle)")
    return parser.parse_args()


# Fonction pour nettoyer la mémoire
def clean_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("🧹 Mémoire nettoyée")


# Point d'entrée principal
def main():
    args = parse_args()

    # Configurer le dossier de sortie
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = args.output_dir

    os.makedirs(save_dir, exist_ok=True)

    # Afficher la configuration
    print("\n⚡ ENTRAÎNEMENT INCRÉMENTAL SUR RTX 3080 ⚡")
    print(f"Batch size: {args.batch_size}, Accumulation: {args.gradient_accumulation}")
    print(f"Max length: {args.max_length}, Étapes: {args.max_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Données: {args.dataset_folder}")
    print(f"Sauvegarde: {save_dir}")

    # === ÉTAPE 1: CHARGEMENT DES DONNÉES ===
    print("\n🔍 ÉTAPE 1: CHARGEMENT DES DONNÉES")
    try:
        # Chemin des données
        tokenized_data_path = os.path.join(args.output_dir, "prepared_data", args.dataset_folder)

        if os.path.exists(tokenized_data_path):
            print(f"Chargement des données depuis {tokenized_data_path}...")
            tokenized_dataset = load_from_disk(tokenized_data_path)
            print(f"Données chargées: {len(tokenized_dataset)} exemples")

            # Vérification des données
            sample = tokenized_dataset[0]
            print(f"Longueur de séquence: {len(sample['input_ids'])}")

            # Si les séquences sont trop longues, les tronquer
            if len(sample['input_ids']) > args.max_length:
                print(f"Troncation des séquences à {args.max_length} tokens...")

                # Créer une copie tronquée
                truncated_dataset = []
                for example in tokenized_dataset:
                    truncated_example = {
                        'input_ids': example['input_ids'][:args.max_length],
                        'attention_mask': example['attention_mask'][:args.max_length],
                        'labels': example['labels'][:args.max_length]
                    }
                    truncated_dataset.append(truncated_example)

                tokenized_dataset = Dataset.from_list(truncated_dataset)
                print(f"Données tronquées: {len(tokenized_dataset)} exemples")
        else:
            print(f"❌ Données non trouvées dans {tokenized_data_path}")
            print("Veuillez vérifier le nom du dossier de données")
            exit(1)

        clean_memory()
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        exit(1)

    # === ÉTAPE 2: CRÉATION D'UN NOUVEAU MODÈLE ===
    print("\n🔧 ÉTAPE 2: CRÉATION D'UN NOUVEAU MODÈLE")
    try:
        # Charger le tokenizer
        print("Chargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
        tokenizer.pad_token = tokenizer.eos_token

        # Configuration de quantification optimisée
        print("Configuration de quantification optimisée...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # Options communes pour le modèle
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "quantization_config": bnb_config,
        }

        # Charger un nouveau modèle
        print("Chargement d'un nouveau modèle de base...")
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/llama-3-8b-bnb-4bit", **model_kwargs
        )

        # Préparer pour LoRA
        print("Préparation pour LoRA...")
        model = prepare_model_for_kbit_training(model)

        # Configuration LoRA
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            lora_dropout=0.0,  # Pas de dropout pour plus de vitesse
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

        # Si un modèle précédent existe, charger ses poids
        old_adapter_path = os.path.join(args.output_dir, "adapter_model.safetensors")
        if os.path.exists(old_adapter_path):
            print(f"Chargement des poids du modèle précédent depuis {old_adapter_path}...")
            # Utiliser le merge_and_unload pour charger les poids adaptateurs
            try:
                state_dict = torch.load(old_adapter_path)
                # Charger les poids adaptateurs
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"Chargement terminé. {len(missing)} clés manquantes, {len(unexpected)} inattendues")
            except Exception as e:
                print(f"Erreur lors du chargement des poids: {e}")
                print("Continuons avec un nouveau modèle...")

        # Vérifier que le modèle est en mode entraînement
        model.train()
        print("Modèle configuré en mode entraînement")

        # Afficher les paramètres
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Paramètres totaux: {total_params:,}")
        print(f"Paramètres entraînables: {trainable_params:,} ({trainable_params / total_params:.2%})")

        # Vérification des paramètres entraînables
        if trainable_params == 0:
            print("❌ ERREUR: Aucun paramètre entraînable!")
            exit(1)

        clean_memory()
    except Exception as e:
        print(f"❌ Erreur lors de la création du modèle: {e}")
        exit(1)

    # === ÉTAPE 3: CONFIGURATION DE L'ENTRAÎNEMENT ===
    print("\n⚙️ ÉTAPE 3: CONFIGURATION DE L'ENTRAÎNEMENT")
    try:
        # Dossier pour les checkpoints
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(save_dir, f"checkpoint_{timestamp}")

        # Configuration optimisée pour la vitesse
        print("Configuration optimisée pour la vitesse...")
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            max_steps=args.max_steps,
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            fp16=True,
            gradient_checkpointing=True,
            optim="adamw_torch",  # Plus stable que fused qui peut causer des erreurs
            weight_decay=0.01,
            max_grad_norm=1.0,
            report_to="none",
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            disable_tqdm=False,
            seed=42
        )

        # Data collator optimisé
        print("Configuration du data collator optimisé...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Configuration du Trainer
        print("Configuration du Trainer optimisé...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        clean_memory()
    except Exception as e:
        print(f"❌ Erreur lors de la configuration de l'entraînement: {e}")
        exit(1)

    # === ÉTAPE 4: LANCEMENT DE L'ENTRAÎNEMENT ===
    print("\n🚀 ÉTAPE 4: LANCEMENT DE L'ENTRAÎNEMENT")
    try:
        # Afficher l'utilisation de la mémoire
        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
            gpu_mem_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
            print(f"Mémoire GPU avant entraînement: {gpu_mem_alloc:.2f} GB alloués, {gpu_mem_reserved:.2f} GB réservés")

        # Démarrer l'entraînement avec timer
        start_time = time.time()
        print(f"Démarrage de l'entraînement à {datetime.now().strftime('%H:%M:%S')}...")

        # Afficher la configuration finale
        print(
            f"Configuration finale: batch={args.batch_size}*{args.gradient_accumulation}={args.batch_size * args.gradient_accumulation}, lr={args.learning_rate}")

        # Désactiver explicitement le cache pour éviter les problèmes avec gradient checkpointing
        model.config.use_cache = False  # Important pour éviter l'erreur "use_cache=True is incompatible with gradient checkpointing"

        # Lancer l'entraînement
        trainer.train()

        # Calculer le temps écoulé
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Entraînement terminé en {int(hours)}h {int(minutes)}m {int(seconds)}s")

        # Afficher l'utilisation finale de la mémoire
        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
            gpu_mem_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
            print(f"Pic de mémoire GPU: {gpu_mem_alloc:.2f} GB alloués, {gpu_mem_reserved:.2f} GB réservés")

        # Sauvegarder le modèle final
        print(f"Sauvegarde du modèle final dans {save_dir}...")
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Modèle et tokenizer sauvegardés dans {save_dir}")

        # Créer un fichier d'information
        with open(os.path.join(save_dir, "training_info.txt"), "w") as f:
            f.write(f"Entraînement incrémental terminé le {datetime.now()}\n")
            f.write(f"Durée: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
            f.write(f"Exemples: {len(tokenized_dataset)} (dossier {args.dataset_folder})\n")
            f.write(
                f"Configuration: batch={args.batch_size}*{args.gradient_accumulation}, lr={args.learning_rate}, steps={args.max_steps}\n")
    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement: {e}")
        # Tenter de sauvegarder quand même
        try:
            print("Tentative de sauvegarde malgré l'erreur...")
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Modèle sauvegardé dans {save_dir}")
        except Exception as e2:
            print(f"❌ Impossible de sauvegarder: {e2}")

    print("\n✅ ENTRAÎNEMENT INCRÉMENTAL TERMINÉ")
    print(f"Modèle disponible dans: {save_dir}")
    print(f"Pour continuer l'entraînement avec d'autres données:")
    print(f"  python {__file__} --dataset_folder autre_dossier_de_données")


if __name__ == "__main__":
    main()