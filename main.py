from dataset import extrair_recortes
from train import treinar_e_avaliar_modelo

if __name__ == "__main__":
    ROOT_DIR = "."  # Diretório raiz onde estão as pastas 'train', 'valid', 'test'
    OUTPUT_DIR = "saida_recortes"

    PROCESS_ALL = False 
    MAX_IMAGES = 50  # Usado apenas se PROCESS_ALL for False

    print(">>> Iniciando Extração de Recortes...")
    extrair_recortes(ROOT_DIR, OUTPUT_DIR, process_all=PROCESS_ALL, max_images=MAX_IMAGES)

    print("\n>>> Iniciando Treinamento e Avaliação...")
    treinar_e_avaliar_modelo(OUTPUT_DIR, epochs=15, batch_size=32, lr=0.001)