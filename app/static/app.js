const el = (id) => document.getElementById(id);
const campos = {
  numero: el("numero"), nome: el("nome"), tipos: el("tipos"),
  altura: el("altura"), peso: el("peso"), descricao: el("descricao"),
  sprite: el("sprite"), barra: el("barra"), confianca: el("confianca"),
};
const led = el("led");
const status = el("status");
const visor = el("visor");
const visorImg = el("visor-img");
const btnCamera = el("btn-camera");
const btnImagem = el("btn-imagem");
const btnRaiox = el("btn-raiox");

const LIMIAR = 0.6;          // confiança mínima para mostrar a ficha
const RETENCAO_MS = 30_000;  // quanto tempo a ficha fica sem o Pokémon ser visto

let timer = null;
let fichaAtual = null;       // nome do Pokémon cuja ficha está na tela
let vistoEm = 0;             // última vez que ele foi visto com confiança >= LIMIAR
let raiox = false;            // raio-x (Grad-CAM) ligado?
let ultimaImagem = null;      // última imagem enviada no modo imagem

/* ---------- ficha ---------- */
function barraConfianca(valor) {
  const cheios = Math.round(valor * 10);
  return "▮".repeat(cheios) + "▯".repeat(10 - cheios);
}

function mostrarConfianca(valor) {
  campos.barra.textContent = barraConfianca(valor);
  campos.confianca.textContent = `${Math.round(valor * 100)}%`;
}

function limparFicha() {
  fichaAtual = null;
  campos.numero.textContent = "nº ---";
  campos.nome.textContent = "Procurando...";
  campos.tipos.textContent = "";
  campos.altura.textContent = "—";
  campos.peso.textContent = "—";
  campos.descricao.textContent = "";
  campos.sprite.removeAttribute("src");
  campos.barra.textContent = "";
  campos.confianca.textContent = "";
  led.classList.remove("ativo");
}

async function mostrarFicha(nome, confianca) {
  fichaAtual = nome;
  vistoEm = Date.now();
  campos.nome.textContent = nome;
  mostrarConfianca(confianca);
  led.classList.add("ativo");
  campos.descricao.textContent = "consultando a pokéapi...";

  try {
    const resp = await fetch(`/pokemon/${encodeURIComponent(nome)}`);
    if (!resp.ok) throw new Error();
    const p = await resp.json();
    if (fichaAtual !== nome) return;     // outro Pokémon chegou enquanto esperávamos

    campos.numero.textContent = `nº ${String(p.numero).padStart(3, "0")}`;
    campos.tipos.textContent = "";
    for (const tipo of p.tipos) {
      const span = document.createElement("span");
      span.className = "tipo";
      span.textContent = tipo;
      campos.tipos.appendChild(span);
    }
    campos.altura.textContent = `${p.altura_m.toFixed(1)} m`;
    campos.peso.textContent = `${p.peso_kg.toFixed(1)} kg`;
    campos.descricao.textContent = p.descricao ?? "";
    if (p.sprite) campos.sprite.src = p.sprite;
  } catch {
    if (fichaAtual === nome) campos.descricao.textContent = "dados indisponíveis";
  }
}

/* ---------- regra de exibição ---------- */
function avaliar(info) {
  const confiante = info.detectado && info.confianca >= LIMIAR;

  if (confiante && info.nome !== fichaAtual) {
    mostrarFicha(info.nome, info.confianca);   // outro Pokémon confiável: troca na hora
  } else if (confiante) {
    vistoEm = Date.now();                       // o mesmo continua na frente: renova
    mostrarConfianca(info.confianca);
  } else if (fichaAtual && Date.now() - vistoEm > RETENCAO_MS) {
    limparFicha();                              // 30 s sem vê-lo com confiança
  }
}

function textoStatus(info) {
  if (!info.detectado) return fichaAtual ? "ficha em espera" : "scan ativo";
  const pct = Math.round(info.confianca * 100);
  return info.confianca >= LIMIAR
    ? `identificado: ${info.nome} ${pct}%`
    : `analisando: ${info.nome}? ${pct}%`;
}

async function atualizar() {
  try {
    const info = await (await fetch("/info")).json();
    avaliar(info);
    status.textContent = textoStatus(info);
  } catch {
    status.textContent = "sem conexão com o servidor";
  }
}

/* ---------- modos ---------- */
function marcarBotao(ativo) {
  btnCamera.classList.toggle("ativo", ativo === "camera");
  btnImagem.classList.toggle("ativo", ativo === "imagem");
}

function modoCamera() {
  visor.classList.remove("modo-imagem", "aguardando");
  marcarBotao("camera");
  visorImg.src = "/video";
  clearInterval(timer);
  timer = setInterval(atualizar, 500);
}

function modoImagem() {
  clearInterval(timer);
  visorImg.removeAttribute("src");
  visor.classList.add("modo-imagem", "aguardando");
  marcarBotao("imagem");
  ultimaImagem = null;
  limparFicha();
  status.textContent = "arraste ou cole uma imagem";
}

async function enviarImagem(file) {
  ultimaImagem = file;
  visor.classList.remove("aguardando");
  status.textContent = "analisando...";

  const form = new FormData();
  form.append("arquivo", file);

  try {
    const resp = await fetch(`/identificar?gradcam=${raiox}`, { method: "POST", body: form });
    const info = await resp.json();
    if (!resp.ok) throw new Error(info.detail);

    visorImg.src = info.imagem;
    avaliar(info);
    status.textContent = info.detectado
      ? textoStatus(info)
      : "imagem · nada encontrado";
  } catch (erro) {
    status.textContent = `erro: ${erro.message}`;
  }
}

/* ---------- arrastar, soltar e colar ---------- */
visor.addEventListener("dragover", (e) => {
  if (!visor.classList.contains("modo-imagem")) return;
  e.preventDefault();
  visor.classList.add("arrastando");
});

visor.addEventListener("dragleave", (e) => {
  if (!visor.contains(e.relatedTarget)) visor.classList.remove("arrastando");
});

visor.addEventListener("drop", (e) => {
  if (!visor.classList.contains("modo-imagem")) return;
  e.preventDefault();
  visor.classList.remove("arrastando");
  const file = e.dataTransfer.files[0];
  if (!file || !file.type.startsWith("image/")) {
    status.textContent = "solte um arquivo de imagem";
    return;
  }
  enviarImagem(file);
});

document.addEventListener("paste", (e) => {
  const file = [...e.clipboardData.files].find((f) => f.type.startsWith("image/"));
  if (!file) return;
  e.preventDefault();
  if (!visor.classList.contains("modo-imagem")) modoImagem();
  enviarImagem(file);
});

window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

btnCamera.addEventListener("click", modoCamera);
btnImagem.addEventListener("click", modoImagem);
btnRaiox.addEventListener("click", async () => {
  raiox = !raiox;
  btnRaiox.classList.toggle("ativo", raiox);
  await fetch(`/gradcam?ativo=${raiox}`, { method: "POST" });
  if (visor.classList.contains("modo-imagem") && ultimaImagem) {
    enviarImagem(ultimaImagem);           // reanalisa a imagem com ou sem raio-x
  }
});

limparFicha();
modoCamera();