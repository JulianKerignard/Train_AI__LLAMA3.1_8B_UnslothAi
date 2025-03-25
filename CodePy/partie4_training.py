# fast_training.py
# Version optimisée pour la vitesse sur RTX 3080

import os
import torch
import argparse
from datasets import load_from_disk
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from transformers import BitsAndBytesConfig
from datetime import datetime


# Analyse des arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement rapide")
    parser.add_argument("--continue_training", action="store_true", help="Continuer l'entraînement")
    parser.add_argument("--start_idx", type=int, default=0, help="Index de départ")
    parser.add_argument("--num_examples", type=int, default=30000, help="Nombre d'exemples")
    parser.add_argument("--batch_size", type=int, default=4, help="Taille du batch")  # Augmenté à 4
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Accumulation")  # Réduit à 4
    parser.add_argument("--max_length", type=int, default=500, help="Longueur max")  # Réduit à 384
    parser.add_argument("--lora_rank", type=int, default=8, help="Rang LoRA")
    parser.add_argument("--max_steps", type=int, default=300, help="Étapes max")  # Réduit à 150
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Taux d'apprentissage")  # Augmenté à 5e-4
    parser.add_argument("--output_dir", type=str, default="optimized_lora_model", help="Dossier de sortie")
    parser.add_argument("--use_flash_attention", action="store_true", help="Utiliser Flash Attention")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true",
                        help="Désactiver gradient checkpointing")
    return parser.parse_args()


# Fonction pour nettoyer la mémoire
def clean_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("🧹 Mémoire nettoyée")


# Point d'entrée principal
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Afficher la configuration
    print("\n⚡ ENTRAÎNEMENT RAPIDE SUR RTX 3080 ⚡")
    print(f"Batch size: {args.batch_size}, Accumulation: {args.gradient_accumulation}")
    print(f"Max length: {args.max_length}, Étapes: {args.max_steps}")
    print(
        f"Learning rate: {args.learning_rate}, Gradient checkpointing: {'Désactivé' if args.disable_gradient_checkpointing else 'Activé'}")

    # === ÉTAPE 1: CHARGEMENT DES DONNÉES ===
    print("\n🔍 ÉTAPE 1: CHARGEMENT DES DONNÉES")
    try:
        tokenized_data_path = os.path.join(args.output_dir, "prepared_data",
                                           f"tokenized_data_{args.start_idx}_{args.start_idx + args.num_examples}")

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

                from datasets import Dataset
                tokenized_dataset = Dataset.from_list(truncated_dataset)
                print(f"Données tronquées: {len(tokenized_dataset)} exemples")
        else:
            print(f"❌ Données non trouvées dans {tokenized_data_path}")
            print("Veuillez d'abord préparer les données avec le script de préparation")
            exit(1)

        clean_memory()
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        exit(1)

    # === ÉTAPE 2: CHARGEMENT DU MODÈLE ===
    print("\n🔧 ÉTAPE 2: CHARGEMENT DU MODÈLE OPTIMISÉ")
    try:
        # Charger le tokenizer
        print("Chargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
        tokenizer.pad_token = tokenizer.eos_token

        # Options avancées pour la vitesse
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "use_cache": not args.disable_gradient_checkpointing,
            # Activer le cache si gradient checkpointing est désactivé
        }

        # Configuration de quantification optimisée
        print("Configuration de quantification optimisée...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model_kwargs["quantization_config"] = bnb_config

        # Si Flash Attention est activé
        if args.use_flash_attention:
            print("Activation de Flash Attention...")
            # Tentative d'activation de Flash Attention si disponible
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception as e:
                print(f"Flash Attention non disponible: {e}")

        # Charger le modèle
        if args.continue_training and os.path.exists(args.output_dir):
            try:
                print(f"Chargement du modèle précédent depuis {args.output_dir}...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    "unsloth/llama-3-8b-bnb-4bit", **model_kwargs
                )
                model = PeftModel.from_pretrained(base_model, args.output_dir)
                print("Modèle précédent chargé avec succès")
            except Exception as e:
                print(f"Erreur lors du chargement du modèle précédent: {e}")
                args.continue_training = False

        if not args.continue_training:
            print("Chargement d'un nouveau modèle...")
            model = AutoModelForCausalLM.from_pretrained(
                "unsloth/llama-3-8b-bnb-4bit", **model_kwargs
            )
            print("Modèle chargé avec succès")

            # Préparer pour LoRA rapide
            print("Préparation pour LoRA optimisé...")
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

        # Essayer d'activer torch.compile pour la vitesse
        try:
            print("Tentative d'activation de torch.compile pour accélération...")
            if hasattr(torch, 'compile'):
                # Seules les fonctions forward des modules spécifiques seront compilées
                model.base_model.model.model.layers[0].self_attn.forward = torch.compile(
                    model.base_model.model.model.layers[0].self_attn.forward,
                    mode="reduce-overhead"
                )
                print("Première couche d'attention compilée avec torch.compile")

                # Compiler plus de couches si nécessaire (ATTENTION: peut augmenter le temps de compilation)
                # for layer in model.base_model.model.model.layers[:4]:  # Compiler les 4 premières couches
                #     layer.self_attn.forward = torch.compile(layer.self_attn.forward, mode="reduce-overhead")
            else:
                print("torch.compile non disponible dans cette version")
        except Exception as e:
            print(f"Impossible d'utiliser torch.compile: {e}")

        # Afficher les paramètres
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Paramètres totaux: {total_params:,}")
        print(f"Paramètres entraînables: {trainable_params:,} ({trainable_params / total_params:.2%})")

        clean_memory()
    except Exception as e:
        print(f"❌ Erreur lors de la configuration du modèle: {e}")
        exit(1)

    # === ÉTAPE 3: CONFIGURATION DE L'ENTRAÎNEMENT RAPIDE ===
    print("\n⚙️ ÉTAPE 3: CONFIGURATION DE L'ENTRAÎNEMENT RAPIDE")
    try:
        # Dossier pour les checkpoints
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_{timestamp}")

        # Configuration optimisée pour la vitesse
        print("Configuration optimisée pour la vitesse...")
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,  # Learning rate plus élevé pour converger plus vite
            lr_scheduler_type="cosine",  # Scheduler cosine plus efficace
            warmup_ratio=0.03,  # Warmup basé sur ratio plutôt que steps
            max_steps=args.max_steps,
            logging_steps=5,  # Logs plus fréquents pour suivre la progression
            save_steps=50,
            save_total_limit=2,  # Garder moins de checkpoints pour économiser l'espace
            fp16=True,
            gradient_checkpointing=not args.disable_gradient_checkpointing,
            optim="adamw_torch_fused",  # Optimiseur fusionné plus rapide
            weight_decay=0.01,
            max_grad_norm=1.0,
            report_to="none",
            dataloader_num_workers=2,  # Utiliser 2 workers pour le chargement des données
            dataloader_pin_memory=True,  # Épingler la mémoire pour des transferts plus rapides
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

    # === ÉTAPE 4: LANCEMENT DE L'ENTRAÎNEMENT RAPIDE ===
    print("\n🚀 ÉTAPE 4: LANCEMENT DE L'ENTRAÎNEMENT RAPIDE")
    try:
        # Afficher l'utilisation de la mémoire
        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
            gpu_mem_reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
            print(f"Mémoire GPU avant entraînement: {gpu_mem_alloc:.2f} GB alloués, {gpu_mem_reserved:.2f} GB réservés")

        # Démarrer l'entraînement avec timer
        import time
        start_time = time.time()
        print(f"Démarrage de l'entraînement à {datetime.now().strftime('%H:%M:%S')}...")

        # Afficher la configuration finale
        print(
            f"Configuration finale: batch={args.batch_size}*{args.gradient_accumulation}={args.batch_size * args.gradient_accumulation}, lr={args.learning_rate}")

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
        print("Sauvegarde du modèle final...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Modèle et tokenizer sauvegardés dans {args.output_dir}")

        # Créer un fichier d'information
        with open(os.path.join(args.output_dir, "training_info.txt"), "w") as f:
            f.write(f"Entraînement rapide terminé le {datetime.now()}\n")
            f.write(f"Durée: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
            f.write(f"Exemples: {len(tokenized_dataset)} (de l'index {args.start_idx})\n")
            f.write(
                f"Configuration: batch={args.batch_size}*{args.gradient_accumulation}, lr={args.learning_rate}, steps={args.max_steps}\n")
    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement: {e}")
        # Tenter de sauvegarder quand même
        try:
            print("Tentative de sauvegarde malgré l'erreur...")
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Modèle sauvegardé dans {args.output_dir}")
        except Exception as e2:
            print(f"❌ Impossible de sauvegarder: {e2}")

    print("\n✅ ENTRAÎNEMENT RAPIDE TERMINÉ")
    print(f"Modèle disponible dans: {args.output_dir}")
    print(
        f"Pour continuer avec de nouvelles données: python {__file__} --continue_training --start_idx {args.start_idx + args.num_examples}")


if __name__ == "__main__":
    main()