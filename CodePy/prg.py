# -*- coding: utf-8 -*-
# Import des bibliothèques nécessaires
from unsloth import FastLanguageModel, to_sharegpt, apply_chat_template, standardize_sharegpt, is_bfloat16_supported
import torch
from datasets import load_dataset, Dataset
import pandas as pd

# Configuration des paramètres
max_seq_length = 2048
dtype = None  # Auto detection
load_in_4bit = True

# Chargement du modèle
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Ajout des adaptateurs LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Chargement du dataset ReactJS
ds = load_dataset("Hardik1234/reactjs-train")
print("Informations sur le dataset:")
print(ds)

# Affichage des noms de colonnes
if "train" in ds:
    print("\nNoms des colonnes:")
    print(ds["train"].column_names)
    print("\nTaille du dataset:")
    print(f"Split 'train': {len(ds['train'])} exemples")
    print("\nExemple de données:")
    print(ds["train"][0])

# Si le dataset a plusieurs splits, sélectionner le split "train"
if isinstance(ds, dict) and "train" in ds:
    train_ds = ds["train"]
else:
    train_ds = ds


# Utiliser la méthode de création directe pour éviter les problèmes de mémoire
def create_dataset_directly(train_ds, max_examples=10000):
    # Préparer les données au format souhaité
    instructions = []
    inputs = []
    outputs = []

    # Limiter le nombre d'exemples
    total = min(max_examples, len(train_ds))

    print(f"Création d'un dataset avec {total} exemples...")

    for i in range(total):
        if i % 1000 == 0:
            print(f"Traitement de l'exemple {i}/{total}")

        example = train_ds[i]
        instructions.append(f"Generate React.js code for file: {example['path']} in repository: {example['repo_name']}")
        inputs.append("")
        outputs.append(example['content'])

    # Créer le dataset directement
    return Dataset.from_dict({
        'instruction': instructions,
        'input': inputs,
        'output': outputs
    })


# Utiliser cette fonction pour créer le dataset
print("\nCréation du dataset avec un nombre limité d'exemples...")
modified_ds = create_dataset_directly(train_ds, max_examples=10000)  # Limité à 10000 exemples
print(f"Dataset créé avec {len(modified_ds)} exemples")

# Conversion au format ShareGPT
print("\nConversion au format ShareGPT...")
converted_ds = to_sharegpt(
    modified_ds,
    merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
    output_column_name="output",
    conversation_extension=3,
)
print("Conversion terminée")

# Standardisation du dataset
print("\nStandardisation du dataset...")
standardized_ds = standardize_sharegpt(converted_ds)
print("Standardisation terminée")

# Définition du template de chat
chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

# Application du template de chat
print("\nApplication du template de chat...")
formatted_ds = apply_chat_template(
    standardized_ds,
    tokenizer=tokenizer,
    chat_template=chat_template,
)
print("Application du template terminée")

# Configuration de l'entraînement
print("\nConfiguration de l'entraînement...")
from trl import SFTTrainer
from transformers import TrainingArguments

# Affichage des statistiques mémoire initiales
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Mémoire max = {max_memory} GB.")
print(f"{start_gpu_memory} GB de mémoire réservée.")

# Définition du trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_ds,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=500,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="../outputs",
        report_to="none",
    ),
)

# Lancement de l'entraînement
print("\nDébut de l'entraînement...")
trainer_stats = trainer.train()

# Affichage des statistiques finales
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} secondes utilisées pour l'entraînement.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes utilisées pour l'entraînement.")
print(f"Pic de mémoire réservée = {used_memory} GB.")
print(f"Pic de mémoire réservée pour l'entraînement = {used_memory_for_lora} GB.")
print(f"Pourcentage de mémoire max réservée = {used_percentage} %.")
print(f"Pourcentage de mémoire max réservée pour l'entraînement = {lora_percentage} %.")

# Sauvegarde du modèle
print("\nSauvegarde du modèle...")
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("Modèle sauvegardé avec succès!")

# Test du modèle
print("\nTest du modèle...")
FastLanguageModel.for_inference(model)
messages = [
    {"role": "user", "content": "Create a React component for a login form with email and password fields"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
print("Génération de réponse:")
_ = model.generate(
    input_ids,
    streamer=text_streamer,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id
)

print("\nScript terminé avec succès!")