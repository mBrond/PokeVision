import json
from pathlib import Path

import kagglehub
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights


ROOT = Path(__file__).resolve().parents[1]
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

def para_rgb(img):
    return img.convert("RGB")

train_tf = transforms.Compose([
    transforms.Lambda(para_rgb),
    transforms.Resize((224, 244)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

eval_tf = transforms.Compose([
    transforms.Lambda(para_rgb),
    transforms.Resize((224, 244)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def carregar_dados():
    path = Path(kagglehub.dataset_download("mikoajkolman/pokemon-images-first-generation17000-files"))
    pasta = path / "pokemon"

    ds_train = ImageFolder(pasta, transform=train_tf)
    ds_eval = ImageFolder(pasta, transform=eval_tf)

    n = len(ds_train)
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(n, generator=g).tolist()
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]


    def loader(ds, ids, shuffle):
        return DataLoader(Subset(ds, ids), batch_size=BATCH, shuffle=shuffle,
                          num_workers=WORKERS, pin_memory=True, persistent_workers=True)


    return (ds_train.classes,
            loader(ds_train, idx_train, True),
            loader(ds_eval, idx_val, False),
            loader(ds_eval, idx_test, False))


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

    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    np.fill_diagonal(cm, 0)   # ignora os acertos
    print("\nPares mais confundidos (real -> previsto: vezes):")
    for i in np.argsort(cm, axis=None)[::-1][:10]:
        real, prev = np.unravel_index(i, cm.shape)
        if cm[real, prev] == 0:
            break
        print(f"  {classes[real]:12} -> {classes[prev]:12}: {cm[real, prev]}")







def main():
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