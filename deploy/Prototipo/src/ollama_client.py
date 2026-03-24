"""Gestion de modelos Ollama y verificacion de estado."""
import sys
from pathlib import Path
from typing import Dict, List

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, SLM_MODELS


def check_ollama_running() -> bool:
    """Verifica si el servidor Ollama esta activo."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def list_installed_models() -> List[str]:
    """Retorna la lista de modelos instalados en Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    except Exception:
        return []


def get_model_info(model_name: str) -> Dict:
    """Obtiene detalles de un modelo desde la API de Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/show",
            json={"name": model_name},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_available_models(required_models: List[str] = None) -> Dict[str, bool]:
    """
    Verifica cuales de los modelos requeridos estan disponibles.

    Returns:
        Dict modelo -> esta_disponible (bool).
    """
    if required_models is None:
        required_models = SLM_MODELS

    installed = list_installed_models()
    # Normalizar: Ollama puede agregar :latest
    installed_normalized = set()
    for name in installed:
        installed_normalized.add(name)
        # "qwen2.5:3b" -> tambien aceptar sin tag si es :latest
        if name.endswith(":latest"):
            installed_normalized.add(name.replace(":latest", ""))

    result = {}
    for model in required_models:
        # Coincidencia exacta del nombre completo
        is_available = model in installed_normalized
        result[model] = is_available

    return result


def print_status():
    """Imprime el estado completo de Ollama y modelos."""
    print("=" * 50)
    print("ESTADO DE OLLAMA")
    print("=" * 50)

    if not check_ollama_running():
        print("OLLAMA NO ESTA ACTIVO")
        print("Inicia Ollama antes de continuar:")
        print("  Windows: Abre la aplicacion Ollama")
        print("  Linux: ollama serve")
        return False

    print("Ollama activo")
    print()

    available = get_available_models()
    installed_count = sum(1 for v in available.values() if v)
    print(f"Modelos instalados: {installed_count}/{len(available)}")
    print()

    for model, is_available in available.items():
        status = "INSTALADO" if is_available else "NO INSTALADO"
        symbol = "+" if is_available else "-"
        print(f"  [{symbol}] {model}: {status}")

    missing = [m for m, v in available.items() if not v]
    if missing:
        print("\nPara instalar los modelos faltantes:")
        for model in missing:
            print(f"  ollama pull {model}")

    print("=" * 50)
    return installed_count > 0


if __name__ == "__main__":
    print_status()
