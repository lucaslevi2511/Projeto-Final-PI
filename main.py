from dataset import extrair_recortes
from train import treinar_e_avaliar_modelo
from logger import enable_logging, disable_logging, logger

if __name__ == "__main__":
    enable_logging()  
    disable_logging()  # Descomente para desativar logs durante a execução
    logger.info("Habilitando logs...")

    ROOT_DIR = "."  # Diretório raiz onde estão as pastas 'train', 'valid', 'test'
    OUTPUT_DIR = "saida_sem_conversao_rgb_para_ycbcr_e_sem_luminancia"  # Diretório de saída para os recortes processados

    PROCESS_ALL = True  # Se True, processa todas as imagens; se False, processa apenas MAX_IMAGES
    MAX_IMAGES = 5 # Usado apenas se PROCESS_ALL for False

    print(">>> Iniciando Extração de Recortes...")
    extrair_recortes(ROOT_DIR, OUTPUT_DIR, process_all=PROCESS_ALL, max_images=MAX_IMAGES)

    print("\n>>> Iniciando Treinamento e Avaliação...")
    treinar_e_avaliar_modelo(OUTPUT_DIR, epochs=100, batch_size=32, lr=0.001)