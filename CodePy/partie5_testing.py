# enhanced_react_chat.py
# Chat interactif amélioré pour votre modèle fine-tuné React

import os
import torch
import threading
import time
import re
from datetime import datetime
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

# Configuration
FINETUNED_MODEL_PATH = "../optimized_lora_model"  # Chemin vers votre modèle fine-tuné
MAX_NEW_TOKENS = 800
TEMPERATURE = 0.7
SAVE_DIR = "../generated_react_code"

# Initialiser Rich pour améliorer l'affichage console
console = Console()

# Créer le dossier de sauvegarde
os.makedirs(SAVE_DIR, exist_ok=True)


def print_header():
    """Affiche un en-tête stylisé pour l'application"""
    console.print(Panel.fit(
        "[bold blue]GÉNÉRATEUR DE CODE REACT[/bold blue]\n"
        "[yellow]Modèle fine-tuné pour créer des composants React[/yellow]",
        border_style="green",
        width=80
    ))


# Affichage du titre
print_header()

# Chargement du modèle et du tokenizer
console.print("\n[bold]Chargement du modèle fine-tuné...[/bold]")

try:
    # Configuration 4-bit pour économiser la mémoire
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # Charger le modèle fine-tuné
    model = AutoPeftModelForCausalLM.from_pretrained(
        FINETUNED_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )

    console.print("[bold green]✓[/bold green] Modèle fine-tuné chargé avec succès!")
except Exception as e:
    console.print(f"[bold red]✗[/bold red] Erreur lors du chargement du modèle: {e}")
    console.print("Essayez de vérifier le chemin du modèle et les permissions des fichiers.")
    exit(1)

# Optimisation du modèle pour l'inférence
model.eval()  # Mettre en mode évaluation
if hasattr(model, 'config'):
    model.config.use_cache = True  # Activer le cache KV pour une inférence plus rapide


# Fonction pour générer une réponse avec streaming
def generate_response_streaming(prompt):
    """Génère une réponse en utilisant le streaming token par token"""
    # Formatage spécifique pour les modèles fine-tunés sur React
    formatted_prompt = f"""Below are some instructions that describe a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""

    # Tokeniser avec attention mask explicite
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Initialiser le streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Paramètres de génération
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    # Démarrer la génération dans un thread séparé
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Collecter la sortie
    generated_text = ""

    # Animation de chargement
    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    spinner_idx = 0

    console.print("\n[bold]Génération en cours:[/bold]", end="")

    # Afficher le texte généré au fur et à mesure
    for text in streamer:
        # Effacer l'animation de chargement si c'est le premier token
        if not generated_text:
            console.print("\r" + " " * 30 + "\r", end="")

        console.print(text, end="", highlight=False)
        generated_text += text

    console.print("\n")
    return generated_text


# Fonction pour sauvegarder le code React généré
def save_react_code(code, description=""):
    """Sauvegarde le code React dans un fichier"""
    # Nettoyer le code (extraire du bloc de code s'il est présent)
    code_blocks = re.findall(r"```(?:jsx|javascript|react|js)?(.*?)```", code, re.DOTALL)

    if code_blocks:
        # Utiliser le premier bloc de code trouvé
        clean_code = code_blocks[0].strip()
    else:
        # S'il n'y a pas de bloc de code, utiliser tout le texte
        clean_code = code

    # Déterminer la meilleure extension en fonction du contenu
    if any(react_marker in code.lower() for react_marker in ["react", "jsx", "component", "usestate"]):
        extension = ".jsx"
    elif "<html" in code or "<body" in code:
        extension = ".html"
    elif "const " in code or "let " in code or "var " in code:
        extension = ".js"
    else:
        extension = ".txt"

    # Créer un nom de fichier basé sur la description et l'horodatage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shortened_desc = "".join(c if c.isalnum() else "_" for c in description[:20].lower())
    filename = f"{SAVE_DIR}/{shortened_desc}_{timestamp}{extension}"

    # Enregistrer le code dans un fichier
    with open(filename, "w", encoding="utf-8") as f:
        f.write(clean_code)

    console.print(f"[bold green]✓[/bold green] Code sauvegardé dans: [blue]{filename}[/blue]")
    return filename, extension


def display_formatted_code(code, language="jsx"):
    """Affiche le code avec coloration syntaxique"""
    # Extraire le code des blocs de code markdown si présents
    code_blocks = re.findall(r"```(?:jsx|javascript|react|js)?(.*?)```", code, re.DOTALL)
    if code_blocks:
        code = code_blocks[0].strip()

    # Afficher avec la coloration syntaxique
    syntax = Syntax(code, language, theme="monokai", line_numbers=True, word_wrap=True)
    console.print(Panel(syntax, title="[bold]Code généré[/bold]", border_style="green"))


# Initialiser l'historique des prompts
history = []

# Try to use prompt_toolkit for enhanced features, with fallback to standard input
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style
    from pygments import highlight
    from pygments.lexers import JsxLexer
    from pygments.formatters import TerminalFormatter

    session = PromptSession()


    def get_user_input(prompt_text):
        """Obtenir l'entrée utilisateur avec prompt_toolkit"""
        return session.prompt(prompt_text)

except Exception as e:
    console.print(f"[yellow]Note: Fonctionnalités avancées de prompt non disponibles: {e}[/yellow]")
    console.print("[yellow]Utilisation de la méthode d'entrée standard[/yellow]")


    # Fallback à l'entrée standard
    def get_user_input(prompt_text):
        """Obtenir l'entrée utilisateur avec input standard"""
        console.print(prompt_text, end="")
        return input()

# Boucle interactive principale
console.print("\n[bold yellow]🧑‍💻 Prêt à générer du code React![/bold yellow]")
console.print("[italic]Posez vos questions ou décrivez ce que vous voulez créer.[/italic]")
console.print("Commandes spéciales:")
console.print("  [blue]!help[/blue]    - Afficher l'aide")
console.print("  [blue]!history[/blue] - Afficher l'historique des prompts")
console.print("  [blue]!temp X[/blue]  - Régler la température à X (ex: !temp 0.8)")
console.print("  [blue]!exit[/blue]    - Quitter")

while True:
    # Obtenir l'entrée utilisateur avec notre fonction adaptative
    try:
        user_input = get_user_input("\n🔷 [bold blue]Votre demande:[/bold blue] ")
    except KeyboardInterrupt:
        console.print("\nAu revoir! 👋")
        break
    except EOFError:
        console.print("\nAu revoir! 👋")
        break

    # Vérifier les commandes spéciales
    if user_input.lower() in ["!exit", "!quit", "!q"]:
        console.print("\n[bold]Au revoir! 👋[/bold]")
        break

    elif user_input.lower() == "!help":
        console.print(Panel.fit(
            "[bold]Aide du générateur de code React[/bold]\n\n"
            "- Entrez une description du composant React que vous souhaitez créer\n"
            "- Le modèle génèrera le code JSX correspondant\n"
            "- Le code sera automatiquement sauvegardé dans un fichier\n\n"
            "[bold]Commandes:[/bold]\n"
            "!help    - Afficher cette aide\n"
            "!history - Voir l'historique des prompts\n"
            "!temp X  - Régler la température (0.1-1.0)\n"
            "!exit    - Quitter l'application",
            title="Aide",
            border_style="blue"
        ))
        continue

    elif user_input.lower() == "!history":
        if not history:
            console.print("[italic]Aucun prompt dans l'historique[/italic]")
            continue

        console.print("\n[bold]Historique des prompts:[/bold]")
        for i, (prompt, _) in enumerate(history):
            console.print(
                f"[blue]{i + 1}.[/blue] {prompt[:50]}..." if len(prompt) > 50 else f"[blue]{i + 1}.[/blue] {prompt}")

        # Permettre de réutiliser un prompt de l'historique
        try:
            choice = get_user_input("\nEntrez le numéro pour réutiliser un prompt (ou 'c' pour annuler): ")
            if choice.lower() == 'c':
                continue

            if choice.isdigit() and 0 < int(choice) <= len(history):
                user_input = history[int(choice) - 1][0]
                console.print(f"Prompt sélectionné: [italic]{user_input}[/italic]")
            else:
                console.print("[red]Choix invalide[/red]")
                continue
        except (KeyboardInterrupt, EOFError):
            continue

    elif user_input.lower().startswith("!temp "):
        try:
            temp_value = float(user_input.split(" ")[1])
            if 0.1 <= temp_value <= 1.0:
                TEMPERATURE = temp_value
                console.print(f"[green]Température réglée à {TEMPERATURE}[/green]")
            else:
                console.print("[red]La température doit être entre 0.1 et 1.0[/red]")
        except (ValueError, IndexError):
            console.print("[red]Format invalide. Utilisez !temp X où X est un nombre entre 0.1 et 1.0[/red]")
        continue

    # Ignorer les entrées vides
    if not user_input.strip():
        continue

    # Ajouter une directive React si non spécifiée
    if not any(term in user_input.lower() for term in ["react", "component", "jsx", "create", "generate"]):
        enhanced_prompt = f"Create a React component for {user_input}"
        console.print(f"Prompt amélioré: [italic]'{enhanced_prompt}'[/italic]")
    else:
        enhanced_prompt = user_input

    # Générer la réponse avec streaming
    try:
        response = generate_response_streaming(enhanced_prompt)

        # Ajouter à l'historique
        history.append((enhanced_prompt, response))

        # Sauvegarder le code
        filename, extension = save_react_code(response, enhanced_prompt)

        # Afficher avec coloration syntaxique
        display_formatted_code(response, language=extension.lstrip('.'))

    except Exception as e:
        console.print(f"[bold red]❌ Erreur lors de la génération:[/bold red] {e}")