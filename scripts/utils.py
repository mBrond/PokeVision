from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # raiz do repositório

#bulba 93
#charmandar 123
#chamrleon 124
#charizard 121
import os


def mapear_ids_por_nome_arquivo(pasta_dataset, mapeamento_nomes):
    """Atualiza o ID da classe nos arquivos .txt com base no nome do próprio arquivo.

    :param pasta_dataset: Caminho para a pasta com os arquivos .txt do YOLO.
    :param mapeamento_nomes: Dicionário relacionando uma parte do nome do arquivo
    ao ID correto.
 Ex: {'cachorro': '0', 'gato': '1'} ou pelo nome exato
 {'foto_1': '0'}
    """
    if not os.path.exists(pasta_dataset):
        print(f"Erro: A pasta '{pasta_dataset}' não foi encontrada.")
        return

    arquivos_alterados = 0

    # Percorre todos os arquivos da pasta
    for nome_arquivo in os.listdir(pasta_dataset):
        if nome_arquivo.endswith(".txt"):
            # Remove a extensão .txt para analisar apenas o nome
            nome_sem_extensao = os.path.splitext(nome_arquivo)[0]

            id_novo = None
            for termo_chave, id_correspondente in mapeamento_nomes.items():
                # Se o termo chave estiver no nome do arquivo (ex: 'foto_gato_01' contém 'gato')
                if termo_chave in nome_sem_extensao:
                    id_novo = str(id_correspondente)
                    break  # Encontrou o ID correspondente, pode parar a busca para este arquivo

            # Se não encontrou um mapeamento para este arquivo, pula para o próximo
            if id_novo is None:
                print(
                    f"Aviso: Nenhum mapeamento encontrado para o arquivo '{nome_arquivo}'."
                )
                continue

            caminho_arquivo = os.path.join(pasta_dataset, nome_arquivo)

            # Lê o conteúdo original do arquivo
            with open(caminho_arquivo, "r", encoding="utf-8") as arquivo:
                linhas = arquivo.readlines()

            novas_linhas = []
            alterou_arquivo = False

            for linha in linhas:
                partes = linha.strip().split()

                if len(partes) >= 5:
                    id_atual = partes[0]

                    # Se o ID atual já for igual ao novo, não precisa reescrever à toa
                    if id_atual != id_novo:
                        partes[0] = id_novo
                        alterou_arquivo = True

                    novas_linhas.append(" ".join(partes) + "\n")
                else:
                    novas_linhas.append(linha)

            # Salva as alterações se houve mudança de ID
            if alterou_arquivo:
                with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
                    arquivo.writelines(novas_linhas)
                arquivos_alterados += 1

    print(
        f"Processo concluído! {arquivos_alterados} arquivos foram atualizados de acordo com seus nomes."
    )


import os
import shutil


def organizar_arquivos_dinamico(path_principal: str, mapeamento_pastas: dict):
    """Organiza os arquivos de um diretório com base em um dicionário de extensões.

    :param path_principal: Caminho absoluto ou relativo da pasta a ser organizada.
    :param mapeamento_pastas: Dicionário no formato {'.extensao': 'NomeDaPasta'}
    """
    # Garante que o caminho principal existe
    if not os.path.exists(path_principal):
        print(f"Erro: O caminho '{path_principal}' não existe.")
        return

    # Lista todos os itens na pasta principal
    itens = os.listdir(path_principal)

    for item in itens:
        caminho_item = os.path.join(path_principal, item)

        # Ignora se for uma pasta (foca apenas em arquivos)
        if os.path.isdir(caminho_item):
            continue

        # Extrai a extensão do arquivo em letras minúsculas (ex: '.png')
        _, extensao = os.path.splitext(item)
        extensao = extensao.lower()

        # Verifica se a extensão do arquivo está no dicionário recebido
        if extensao in mapeamento_pastas:
            nome_pasta_destino = mapeamento_pastas[extensao]
            caminho_pasta_destino = os.path.join(
                path_principal, nome_pasta_destino
            )

            # Cria a pasta de destino se ela ainda não existir
            if not os.path.exists(caminho_pasta_destino):
                os.makedirs(caminho_pasta_destino)

            # Move o arquivo para a nova pasta
            shutil.move(caminho_item, os.path.join(caminho_pasta_destino, item))
            print(f"Movido: {item} -> {nome_pasta_destino}/")

if __name__ == "__main__":

    # Defina o mapeamento baseado em palavras-chave ou nos nomes exatos dos arquivos.
    # O script usa uma verificação do tipo "se a chave está contida no nome do arquivo".
    meu_mapeamento = {
        "bulbasaur": "0",  # Arquivos com 'classeA' no nome terão o ID alterado para 0
        "charmander": "1",  # Arquivos com 'classeB' no nome terão o ID alterado para 1
        "charmeleon": "2", 
        "charizard": "3",
    }

    #mapear_ids_por_nome_arquivo(caminho_da_pasta, meu_mapeamento)

    organizar_arquivos_dinamico(ROOT / "data/pokemon-reduzido/test", {".txt": "labels", ".png":"images"})
    organizar_arquivos_dinamico(ROOT / "data/pokemon-reduzido/train", {".txt": "labels", ".png":"images"})
    organizar_arquivos_dinamico(ROOT / "data/pokemon-reduzido/val", {".txt": "labels", ".png":"images"})