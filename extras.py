import os
import glob


def contar_imagens(diretorio):
    """
    Conta a quantidade total de imagens em um diretório e seus subdiretórios.

    Args:
        diretorio (str): Caminho para o diretório a ser analisado.

    Returns:
        dict: Dicionário contendo:
            - 'total': quantidade total de imagens
            - 'por_tipo': contagem por extensão
            - 'por_subdir': contagem por subdiretório
    """
    extensoes_imagem = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    total_imagens = 0
    contagem_por_tipo = {}
    contagem_por_subdir = {}
    arquivos_contados = set()

    # Percorrer todos os arquivos no diretório e subdiretórios
    for root, dirs, files in os.walk(diretorio):
        imagens_nesse_dir = 0

        for ext in extensoes_imagem:
            # Buscar arquivos com essa extensão (case-insensitive)
            pattern = os.path.join(root, ext)
            arquivos = glob.glob(pattern)

            for arquivo in arquivos:
                arquivo_normalizado = os.path.normcase(
                    os.path.abspath(arquivo))
                if arquivo_normalizado in arquivos_contados:
                    continue

                arquivos_contados.add(arquivo_normalizado)
                total_imagens += 1
                imagens_nesse_dir += 1

                # Contar por tipo de extensão
                _, ext_arquivo = os.path.splitext(arquivo)
                ext_arquivo = ext_arquivo.lower()
                contagem_por_tipo[ext_arquivo] = contagem_por_tipo.get(
                    ext_arquivo, 0) + 1

        # Contar por subdiretório
        if imagens_nesse_dir > 0:
            subdir_relativo = os.path.relpath(root, diretorio)
            contagem_por_subdir[subdir_relativo] = imagens_nesse_dir

    return {
        'total': total_imagens,
        'por_tipo': contagem_por_tipo,
        'por_subdir': contagem_por_subdir
    }


def exibir_contagem(diretorio):
    """
    Exibe de forma formatada a contagem de imagens em um diretório.

    Args:
        diretorio (str): Caminho para o diretório a ser analisado.
    """
    resultado = contar_imagens(diretorio)

    print(f"\n{'='*50}")
    print(f"Contagem de Imagens: {diretorio}")
    print(f"{'='*50}")

    print(f"\n[TOTAL] {resultado['total']} imagens encontradas\n")

    if resultado['por_tipo']:
        print("[POR TIPO]")
        for ext, count in sorted(resultado['por_tipo'].items()):
            print(f"  {ext}: {count}")

    if resultado['por_subdir']:
        print("\n[POR SUBDIRETÓRIO]")
        for subdir, count in sorted(resultado['por_subdir'].items()):
            print(f"  {subdir}: {count}")

    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    # Exemplo de uso
    import sys

    if len(sys.argv) > 1:
        diretorio = sys.argv[1]
    else:
        diretorio = "."

    if os.path.isdir(diretorio):
        exibir_contagem(diretorio)
    else:
        print(f"Erro: '{diretorio}' não é um diretório válido.")

print("\n\n======Contagem de imagens de TESTE no dataset original======:\n\n")
exibir_contagem("test")
print("\n\n======Contagem de imagens de TREINAMENTO no dataset original======:\n\n")
exibir_contagem("train")
print("\n\n======Contagem de imagens de VALIDAÇÃO no dataset original======:\n\n")
exibir_contagem("valid")
print("\n\n======Contagem de imagens de SAÍDA DE RECORTE após pipeline======:\n\n")
exibir_contagem("saida_recortes")
print("\n\n======Contagem de imagens de SAÍDA SEM BALANCEAMENTO DE CORES após pipeline======:\n\n")
exibir_contagem("saida_sem_balanceamento_de_cores")