import requests

BASE = "https://pokeapi.co/api/v2"
IDIOMAS = ("pt-br", "pt", "en")   # ordem de preferência para a descrição

TIPOS = {
    "normal": "normal", "fire": "fogo", "water": "água", "grass": "planta",
    "electric": "elétrico", "ice": "gelo", "fighting": "lutador", "poison": "venenoso",
    "ground": "terrestre", "flying": "voador", "psychic": "psíquico", "bug": "inseto",
    "rock": "pedra", "ghost": "fantasma", "dragon": "dragão", "dark": "sombrio",
    "steel": "aço", "fairy": "fada",
}

_cache = {}


def slug(nome):
    """'Mr. Mime' -> 'mr-mime', 'Nidoran♀' -> 'nidoran-f' (formato dos nomes na PokéAPI)."""
    s = nome.lower().replace("♀", "-f").replace("♂", "-m")
    s = s.replace(". ", "-").replace(".", "").replace("'", "").replace(" ", "-")
    return s


def _descricao(entradas):
    for idioma in IDIOMAS:
        for e in entradas:
            if e["language"]["name"] == idioma:
                return " ".join(e["flavor_text"].split())   # remove \n e \f do texto
    return None


def buscar(nome):
    if nome in _cache:
        return _cache[nome]

    s = slug(nome)
    try:
        p = requests.get(f"{BASE}/pokemon/{s}", timeout=5)
        p.raise_for_status()
        e = requests.get(f"{BASE}/pokemon-species/{s}", timeout=5)
        e.raise_for_status()
    except requests.RequestException:
        return None                        # falhas não entram no cache

    p, e = p.json(), e.json()
    dados = {
        "numero": p["id"],
        "nome": nome,
        "tipos": [TIPOS.get(t["type"]["name"], t["type"]["name"])
                  for t in sorted(p["types"], key=lambda t: t["slot"])],
        "altura_m": p["height"] / 10,      # a API usa decímetros
        "peso_kg": p["weight"] / 10,       # e hectogramas
        "sprite": p["sprites"]["front_default"],
        "descricao": _descricao(e["flavor_text_entries"]),
    }
    _cache[nome] = dados
    return dados