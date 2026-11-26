import csv
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):      # rodado como script (python classifier/treinar_resnet.py)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from classifier.fundos import compor
from classifier.preparar_dados import abrir


ROOT = Path(__file__).resolve().parents[1]
DADOS = ROOT / "data/classificador"          # gerado por classifier/preparar_dados.py
SAIDA = ROOT / "models/classifier"
PESOS = SAIDA / "resnet18_best.pth"
CLASSES_JSON = SAIDA / "classes.json"

EPOCHS = 20
BATCH = 32
LR = 1e-4
WORKERS = 4
SEED = 42

MEAN = [0.485, 0.456, 0.406]   # normalização do ImageNet
STD = [0.229, 0.224, 0.225]
COR_MEDIA = tuple(round(m * 255) for m in MEAN)   # preenche os cantos da rotação (vira ~0 após normalizar)

def para_rgb(img):
    return img.convert("RGB")

train_tf = transforms.Compose([
    transforms.Lambda(para_rgb),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20, fill=COR_MEDIA),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

eval_tf = transforms.Compose([
    transforms.Lambda(para_rgb),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


class DatasetManifesto(Dataset):
    """Lê as imagens listadas no manifest.csv. Imagens transparentes ganham fundo
    aleatório (treino) ou branco (avaliação), em vez do preto do convert("RGB")."""

    def __init__(self, linhas, classes, transform, fundo_aleatorio):
        idx = {c: i for i, c in enumerate(classes)}
        self.samples = [(l["caminho"], idx[l["classe"]]) for l in linhas]   # mesmo formato do ImageFolder
        self.fontes = [l["fonte"] for l in linhas]
        self.transform = transform
        self.fundo_aleatorio = fundo_aleatorio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        caminho, rotulo = self.samples[i]
        img = compor(abrir(caminho), self.fundo_aleatorio)
        return self.transform(img), rotulo


def carregar_dados():
    manifesto = DADOS / "manifest.csv"
    if not manifesto.exists():
        raise SystemExit(f"{manifesto} não existe. Rode antes: python classifier/preparar_dados.py")

    classes = json.loads((DADOS / "classes.json").read_text(encoding="utf-8"))
    with open(manifesto, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    ds_train = DatasetManifesto(linhas, classes, train_tf, fundo_aleatorio=True)
    ds_eval = DatasetManifesto(linhas, classes, eval_tf, fundo_aleatorio=False)
    ids = {s: [i for i, l in enumerate(linhas) if l["split"] == s] for s in ("train", "val", "test")}

    def loader(ds, ids, shuffle):
        return DataLoader(Subset(ds, ids), batch_size=BATCH, shuffle=shuffle,
                          num_workers=WORKERS, pin_memory=True, persistent_workers=True)

    return (classes,
            loader(ds_train, ids["train"], True),
            loader(ds_eval, ids["val"], False),
            loader(ds_eval, ids["test"], False))


def criar_modelo(n_classes, device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)

def rodar_epoca(model, loader, criterion, device, optimizer=None):
    """Com optimizer = treino, sem = avaliação"""

    treino = optimizer is not None
    model.train(treino)

    perda_total, acertos, total = 0.0, 0, 0

    with torch.set_grad_enabled(treino):

        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            out = model(x)
            loss = criterion(out, y)

            if treino:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            perda_total += loss.item() * y.size(0)
            acertos += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return perda_total / total, acertos / total


def treinar(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    melhor = 0.0

    for epoca in range(1, EPOCHS + 1):
        tr_loss, tr_acc = rodar_epoca(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = rodar_epoca(model, val_loader, criterion, device)
        scheduler.step()


        marca = ""
        if va_acc > melhor:
            melhor = va_acc
            torch.save(model.state_dict(), PESOS)
            marca = "  <- melhor, salvo"

        print(f"[{epoca:2d}/{EPOCHS}] treino {tr_loss:.3f}/{tr_acc:.3f} | "
              f"val {va_loss:.3f}/{va_acc:.3f}{marca}")

    return melhor 


def avaliar_teste(model, test_loader, classes, device):
    model.load_state_dict(torch.load(PESOS, map_location=device))
    model.eval()
    preds, labels, top5_acertos = [], [], 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            top5 = out.topk(5, dim=1).indices
            top5_acertos += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            preds += out.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()


    preds, labels = np.array(preds), np.array(labels)
    print(f"\nTeste: top-1 {(preds == labels).mean():.4f} | top-5 {top5_acertos / len(labels):.4f}")

    subset = test_loader.dataset
    fontes = np.array([subset.dataset.fontes[i] for i in subset.indices])
    for fonte in sorted(set(fontes)):
        m = fontes == fonte
        print(f"  {fonte:13} top-1 {(preds[m] == labels[m]).mean():.4f}  ({m.sum()} imagens)")

    por_classe = [(classes[c], (preds[labels == c] == c).mean(), (labels == c).sum())
                  for c in range(len(classes)) if (labels == c).any()]
    print("\nClasses com menor acerto:")
    for nome, acc, n in sorted(por_classe, key=lambda t: t[1])[:8]:
        print(f"  {nome:12} {acc:.2f}  ({n} imagens)")

    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    np.fill_diagonal(cm, 0)   # ignora os acertos
    print("\nPares mais confundidos (real -> previsto: vezes):")
    for i in np.argsort(cm, axis=None)[::-1][:10]:
        real, prev = np.unravel_index(i, cm.shape)
        if cm[real, prev] == 0:
            break
        print(f"  {classes[real]:12} -> {classes[prev]:12}: {cm[real, prev]}")







def fixar_semente(seed=SEED):
    """Mesma ordem dos lotes, mesmos fundos aleatórios e mesma inicialização a cada rodada.

    Na GPU ainda sobra uma variação pequena entre rodadas: algumas operações do backward
    em CUDA não são determinísticas, e GPUs ou versões do PyTorch diferentes somam de outro jeito.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # também define a semente de cada worker do DataLoader
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    fixar_semente()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAIDA.mkdir(parents=True, exist_ok=True)

    classes, train_loader, val_loader, test_loader = carregar_dados()

    CLASSES_JSON.write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(classes)} classes | treino {len(train_loader.dataset)} | "
        f"val {len(val_loader.dataset)} | teste {len(test_loader.dataset)}")


    model = criar_modelo(len(classes), device)
    melhor = treinar(model, train_loader, val_loader, device)

    print(f"Melhor acuracia: {melhor:.4f}")

    avaliar_teste(model, test_loader, classes, device)

if __name__ == "__main__":
    main()