# partie1_model.py
# Cette partie charge uniquement le modèle - exécutez-la séparément

from unsloth import FastLanguageModel
import torch

# Configuration des paramètres
max_seq_length = 2048
dtype = None  # Auto detection
load_in_4bit = True

# Chargement du modèle
print("Chargement du modèle...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Ajout des adaptateurs LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Sauvegarde temporaire du modèle et du tokenizer
print("Sauvegarde du modèle initial...")
model.save_pretrained("temp_model")
tokenizer.save_pretrained("temp_model")
print("Modèle initial sauvegardé avec succès!")