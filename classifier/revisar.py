"""Página local para revisar as imagens do classificador, das mais suspeitas para as menos.

    python classifier/revisar.py          # abre em http://127.0.0.1:8001

Clique numa imagem para excluí-la (clique de novo desfaz). Cada clique já grava em
classifier/excluir.txt. Depois da revisão:

    python classifier/preparar_dados.py   # tira as excluídas do manifesto (splits não mudam)
    python classifier/treinar_resnet.py

A ordem vem de data/classificador/triagem.csv (classifier/triagem_clip.py). Sem ele, as
imagens aparecem na ordem do manifesto.
"""
import csv
import io
import json
import sys
import threading
from functools import lru_cache
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from classifier.fundos import compor
from classifier.preparar_dados import EXCLUIR, GEN1, abrir

ROOT = Path(__file__).resolve().parents[1]
DADOS = ROOT / "data/classificador"
REVISADAS = Path(__file__).resolve().parent / "revisadas.json"
LIMITE_SUSPEITA = 0.75     # a partir daqui a nota aparece em destaque
TOP_GERAL = 400            # quantas imagens a visão "todas as classes" mostra

_trava = threading.Lock()


def _carregar():
    arquivo = DADOS / "triagem.csv"
    if not arquivo.exists():
        arquivo = DADOS / "manifest.csv"
    with open(arquivo, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    for i, l in enumerate(linhas):
        l["id"] = i
        l["suspeita"] = float(l.get("suspeita") or 0)
    return linhas


IMGS = _carregar()
POR_CLASSE = {c: [l for l in IMGS if l["classe"] == c] for c in GEN1}
MULTI = "*varios"   # visão especial: imagens com mais de um Pokémon


def _prioridade_multi(l):
    """2 = duas espécies separadas, 1 = várias figuras ou o CLIP achou um grupo, 0 = nada."""
    m = l.get("motivo", "")
    if "2+ espécies" in m:
        return 2
    return int("várias figuras" in m or "vários Pokémon" in m or "linha evolutiva" in m)


def _ler_excluir():
    """md5 -> linha do arquivo (preserva comentários e exclusões de imagens fora do manifesto)."""
    if not EXCLUIR.exists():
        return {}
    out = {}
    for l in EXCLUIR.read_text(encoding="utf-8").splitlines():
        if l.strip() and not l.startswith("#"):
            out[l.split()[0]] = l
    return out


def _gravar_excluir(excl):
    cabecalho = ("# Imagens tiradas do dataset do classificador na revisão manual.\n"
                 "# Formato: md5  Classe/arquivo. Lido por classifier/preparar_dados.py.\n")
    corpo = "\n".join(sorted(excl.values(), key=lambda l: l.split()[1] if len(l.split()) > 1 else l))
    EXCLUIR.write_text(cabecalho + corpo + ("\n" if corpo else ""), encoding="utf-8")


def _ler_revisadas():
    return set(json.loads(REVISADAS.read_text(encoding="utf-8"))) if REVISADAS.exists() else set()


app = FastAPI(title="Revisão do dataset")


@app.get("/", response_class=HTMLResponse)
def pagina():
    return PAGINA


@app.get("/api/classes")
def classes():
    excl, rev = _ler_excluir(), _ler_revisadas()
    out = []
    for c in GEN1:
        da_classe = POR_CLASSE[c]
        out.append({
            "nome": c,
            "total": len(da_classe),
            "suspeitas": sum(l["suspeita"] >= LIMITE_SUSPEITA for l in da_classe),
            "excluidas": sum(l["md5"] in excl for l in da_classe),
            "revisada": c in rev,
        })
    return out


@app.get("/api/imagens")
def imagens(classe: str = "", split: str = "todos"):
    excl = _ler_excluir()
    if classe == MULTI:
        sel = [l for l in IMGS if _prioridade_multi(l)]
    else:
        sel = POR_CLASSE.get(classe, []) if classe else IMGS
    if split == "valteste":
        sel = [l for l in sel if l["split"] in ("val", "test")]
    elif split == "treino":
        sel = [l for l in sel if l["split"] == "train"]
    if classe == MULTI:
        sel = sorted(sel, key=lambda l: (-_prioridade_multi(l), -l["suspeita"]))
    else:
        sel = sorted(sel, key=lambda l: -l["suspeita"])
    if not classe:
        sel = sel[:TOP_GERAL]
    campos = ("id", "classe", "split", "fonte", "suspeita", "clip_pred", "clip_pred_prob", "motivo")
    return [{**{k: l.get(k, "") for k in campos}, "excluida": l["md5"] in excl} for l in sel]


class Marcacao(BaseModel):
    id: int
    excluir: bool


@app.post("/api/excluir")
def excluir(m: Marcacao):
    if not 0 <= m.id < len(IMGS):
        raise HTTPException(404)
    l = IMGS[m.id]
    with _trava:
        excl = _ler_excluir()
        if m.excluir:
            excl[l["md5"]] = f"{l['md5']}  {l['classe']}/{Path(l['caminho']).name}"
        else:
            excl.pop(l["md5"], None)
        _gravar_excluir(excl)
    return {"ok": True}


class Revisao(BaseModel):
    classe: str
    revisada: bool


@app.post("/api/revisada")
def revisada(r: Revisao):
    with _trava:
        rev = _ler_revisadas()
        (rev.add if r.revisada else rev.discard)(r.classe)
        REVISADAS.write_text(json.dumps(sorted(rev, key=GEN1.index), ensure_ascii=False, indent=1),
                             encoding="utf-8")
    return {"ok": True}


@lru_cache(maxsize=6000)
def _miniatura(i, lado):
    img = compor(abrir(IMGS[i]["caminho"]), aleatorio=False)
    img.thumbnail((lado, lado))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


@app.get("/img/{i}")
def imagem(i: int, lado: int = 220):
    if not 0 <= i < len(IMGS):
        raise HTTPException(404)
    return Response(_miniatura(i, min(lado, 1600)), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=86400"})


PAGINA = """<!doctype html>
<html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Revisão do dataset</title>
<style>
:root { --bg:#f4f4f2; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#deded9;
        --red:#c62828; --amber:#b26a00; --green:#2e7d32; --accent:#1f5fbf; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#18181a; --fg:#ececea; --muted:#9a9a95; --card:#232326; --line:#34343a;
          --red:#ef5350; --amber:#ffb74d; --green:#66bb6a; --accent:#6ea8ff; } }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--fg);
       display:grid; grid-template-columns:230px 1fr; height:100vh; }
aside { overflow-y:auto; border-right:1px solid var(--line); padding:8px 0; }
aside button { display:flex; justify-content:space-between; gap:6px; width:100%; border:0;
       background:none; color:inherit; padding:4px 12px; text-align:left; cursor:pointer; font:inherit; }
aside button:hover { background:var(--line); }
aside button.ativa { background:var(--accent); color:#fff; }
aside .meta { color:var(--muted); font-size:12px; white-space:nowrap; }
aside button.ativa .meta { color:#fff; }
.ok { color:var(--green); }
main { overflow-y:auto; padding:12px 16px 40px; }
header { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-bottom:12px; }
header h1 { font-size:20px; margin:0 8px 0 0; }
header .info { color:var(--muted); }
header select, header button { font:inherit; padding:4px 10px; border-radius:6px;
       border:1px solid var(--line); background:var(--card); color:inherit; cursor:pointer; }
header button.prim { background:var(--accent); color:#fff; border-color:var(--accent); }
.dica { color:var(--muted); font-size:12px; width:100%; }
.grade { display:grid; grid-template-columns:repeat(auto-fill, minmax(170px, 1fr)); gap:10px; }
.card { background:var(--card); border:2px solid var(--line); border-radius:8px; overflow:hidden;
        cursor:pointer; position:relative; }
.card img { display:block; width:100%; height:170px; object-fit:contain; background:#fff; }
.card .txt { padding:4px 6px; font-size:12px; display:flex; flex-wrap:wrap; gap:4px 8px; }
.card .nota { font-weight:600; }
.card .nota.alta { color:var(--amber); }
.card .tag { color:var(--muted); }
.card .clip { color:var(--red); width:100%; }
.card.excluida { border-color:var(--red); }
.card.excluida img { opacity:.25; }
.card.excluida::after { content:"excluída"; position:absolute; top:60px; left:0; right:0;
        text-align:center; color:var(--red); font-weight:700; font-size:18px; }
.card a { position:absolute; top:4px; right:6px; text-decoration:none; font-size:16px;
        background:var(--card); border-radius:4px; padding:0 4px; color:var(--fg); }
</style></head>
<body>
<aside id="lista"></aside>
<main>
  <header>
    <h1 id="titulo">…</h1><span class="info" id="info"></span>
    <select id="split">
      <option value="todos">todos os splits</option>
      <option value="valteste">só val + teste</option>
      <option value="treino">só treino</option>
    </select>
    <button class="prim" id="proxima">Classe revisada → próxima (N)</button>
    <div class="dica">Mais suspeitas primeiro. Clique para excluir ou desfazer; ↗ abre a imagem grande.
      Em vermelho: espécie que o CLIP enxerga, quando é outra.</div>
  </header>
  <div class="grade" id="grade"></div>
</main>
<script>
let classes = [], atual = "";
const MULTI = "*varios";
const $ = s => document.querySelector(s);

async function carregarClasses() {
  classes = await (await fetch("/api/classes")).json();
  const lista = $("#lista");
  lista.innerHTML = "";
  const todas = document.createElement("button");
  todas.innerHTML = `<span>★ Mais suspeitas (geral)</span>`;
  todas.className = atual === "" ? "ativa" : "";
  todas.onclick = () => abrir("");
  lista.appendChild(todas);
  const multi = document.createElement("button");
  multi.innerHTML = `<span>★ Mais de um Pokémon</span>`;
  multi.className = atual === MULTI ? "ativa" : "";
  multi.onclick = () => abrir(MULTI);
  lista.appendChild(multi);
  for (const c of classes) {
    const b = document.createElement("button");
    b.className = c.nome === atual ? "ativa" : "";
    const marca = c.revisada ? '<span class="ok">✓</span> ' : "";
    b.innerHTML = `<span>${marca}${c.nome}</span><span class="meta">${c.excluidas ? c.excluidas + " ✕ · " : ""}${c.total}</span>`;
    b.onclick = () => abrir(c.nome);
    lista.appendChild(b);
  }
}

async function abrir(classe) {
  atual = classe;
  const split = $("#split").value;
  const itens = await (await fetch(`/api/imagens?classe=${encodeURIComponent(classe)}&split=${split}`)).json();
  const real = classes.some(c => c.nome === classe);
  $("#titulo").textContent = classe === MULTI ? "Mais de um Pokémon na imagem"
                           : classe || "Mais suspeitas de todas as classes";
  const info = () => {
    const ex = itens.filter(i => i.excluida).length;
    const c = classes.find(x => x.nome === classe);
    $("#info").textContent = `${itens.length} imagens · ${ex} excluídas${c && c.revisada ? " · revisada ✓" : ""}`;
  };
  info();
  $("#proxima").style.display = real ? "" : "none";
  const grade = $("#grade");
  grade.innerHTML = "";
  for (const it of itens) {
    const d = document.createElement("div");
    d.className = "card" + (it.excluida ? " excluida" : "");
    const outra = it.clip_pred && it.clip_pred !== it.classe
      ? `<span class="clip">CLIP: ${it.clip_pred} ${Math.round(it.clip_pred_prob * 100)}%</span>` : "";
    d.innerHTML = `<img loading="lazy" src="/img/${it.id}" alt="">
      <a href="/img/${it.id}?lado=1200" target="_blank" title="abrir grande">↗</a>
      <div class="txt"><span class="nota ${it.suspeita >= 0.75 ? "alta" : ""}">${it.suspeita.toFixed(2)}</span>
      ${real ? "" : `<span>${it.classe}</span>`}
      <span class="tag">${it.split}</span>${it.motivo ? `<span class="tag">${it.motivo}</span>` : ""}${outra}</div>`;
    d.onclick = async ev => {
      if (ev.target.tagName === "A") return;
      it.excluida = !it.excluida;
      d.classList.toggle("excluida", it.excluida);
      info();
      await fetch("/api/excluir", { method: "POST", headers: { "Content-Type": "application/json" },
                                   body: JSON.stringify({ id: it.id, excluir: it.excluida }) });
      carregarClasses();
    };
    grade.appendChild(d);
  }
  $("main").scrollTop = 0;
  carregarClasses();
}

async function proxima() {
  if (!classes.some(c => c.nome === atual)) return;
  await fetch("/api/revisada", { method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ classe: atual, revisada: true }) });
  const i = classes.findIndex(c => c.nome === atual);
  await carregarClasses();
  const prox = classes.slice(i + 1).find(c => !c.revisada) || classes.find(c => !c.revisada);
  abrir(prox ? prox.nome : atual);
}

$("#proxima").onclick = proxima;
$("#split").onchange = () => abrir(atual);
document.addEventListener("keydown", e => { if (e.key === "n" || e.key === "N") proxima(); });
carregarClasses().then(() => abrir(""));
</script>
</body></html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
