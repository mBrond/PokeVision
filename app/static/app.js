const el = (id) => document.getElementById(id);
const nome = el("nome");
const confianca = el("confianca");
const barra = el("barra");
const led = el("led");
const status = el("status");
const visor = el("visor");
const visorImg = el("visor-img");
const btnCamera = el("btn-camera");
const btnImagem = el("btn-imagem");

let timer = null;

function barraConfianca(valor) {
  const cheios = Math.round(valor * 10);
  return "▮".repeat(cheios) + "▯".repeat(10 - cheios);
}

function preencher(info) {
  if (info.detectado) {
    nome.textContent = info.nome;
    barra.textContent = barraConfianca(info.confianca);
    confianca.textContent = `${Math.round(info.confianca * 100)}%`;
    led.classList.add("ativo");
  } else {
    nome.textContent = "Procurando...";
    barra.textContent = "";
    confianca.textContent = "";
    led.classList.remove("ativo");
  }
}

async function atualizar() {
  try {
    const info = await (await fetch("/info")).json();
    preencher(info);
    status.textContent = info.detectado ? "pokémon detectado" : "scan ativo";
  } catch {
    nome.textContent = "Sem conexão";
    led.classList.remove("ativo");
  }
}

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
  clearInterval(timer);                   // para de consultar a câmera
  visorImg.removeAttribute("src");        // fecha o stream de vídeo
  visor.classList.add("modo-imagem", "aguardando");
  marcarBotao("imagem");
  preencher({ detectado: false });
  status.textContent = "arraste ou cole uma imagem";
}

async function enviarImagem(file) {
  visor.classList.remove("aguardando");
  status.textContent = "analisando...";

  const form = new FormData();
  form.append("arquivo", file);

  try {
    const resp = await fetch("/identificar", { method: "POST", body: form });
    const info = await resp.json();
    if (!resp.ok) throw new Error(info.detail);

    visorImg.src = info.imagem;
    preencher(info);
    status.textContent = info.detectado
      ? `imagem · ${info.todos.length} pokémon`
      : "imagem · nada encontrado";
  } catch (erro) {
    status.textContent = `erro: ${erro.message}`;
  }
}

/* ---------- arrastar e soltar ---------- */
visor.addEventListener("dragover", (e) => {
  if (!visor.classList.contains("modo-imagem")) return;
  e.preventDefault();                     // sem isso o navegador não deixa soltar
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

// Evita que o navegador abra a imagem se ela for solta fora do visor
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

document.addEventListener("paste", (e) => {
  const file = [...e.clipboardData.files].find((f) => f.type.startsWith("image/"));
  if (!file) return;                     

  e.preventDefault();
  if (!visor.classList.contains("modo-imagem")) modoImagem();
  enviarImagem(file);
});


btnCamera.addEventListener("click", modoCamera);
btnImagem.addEventListener("click", modoImagem);

modoCamera();