import heapq
import numpy as np

class Node:
    def __init__(self, frequency, symbol, left=None, right=None):
        self.frequency = frequency
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huffman_direction = ''

    def __lt__(self, nxt):
        return self.frequency < nxt.frequency

# Dicionário global para armazenar os códigos gerados
huffman_codes = {}

def calculate_huffman_codes(node, code=''):
    global huffman_codes
    code += node.huffman_direction
    
    if node.left:
        calculate_huffman_codes(node.left, code)
    if node.right:
        calculate_huffman_codes(node.right, code)
        
    if not node.left and not node.right:
        huffman_codes[node.symbol] = code
    return huffman_codes

def get_merged_huffman_tree(byte_to_frequency):
    huffman_tree = []
    for byte, frequency in byte_to_frequency.items():
        heapq.heappush(huffman_tree, Node(frequency, byte))
    
    while len(huffman_tree) > 1:
        left = heapq.heappop(huffman_tree)
        right = heapq.heappop(huffman_tree)
        
        left.huffman_direction = "0"
        right.huffman_direction = "1"
        
        merged_node = Node(left.frequency + right.frequency, None, left, right)
        heapq.heappush(huffman_tree, merged_node)
        
    return huffman_tree[0]
    
def compress_numpy_image(img_np):
    global huffman_codes
    huffman_codes = {}
    
    valores, frequencias = np.unique(img_np, return_counts=True)
    byte_to_frequency = dict(zip(valores, frequencias))
    
    merged_huffman_tree = get_merged_huffman_tree(byte_to_frequency)
    calculate_huffman_codes(merged_huffman_tree)
    
    flat_img = img_np.ravel()
    compressed_bits = "".join(huffman_codes[pixel] for pixel in flat_img)
    
    return compressed_bits
    
def decompress_to_numpy(compressed_bits, original_shape):
    code_to_pixel = {code: pixel for pixel, code in huffman_codes.items()}
    
    pixels_reconstruidos = []
    current_code = ""
    
    for bit in compressed_bits:
        current_code += bit
        if current_code in code_to_pixel:
            pixels_reconstruidos.append(code_to_pixel[current_code])
            current_code = ""
                
    return np.array(pixels_reconstruidos, dtype=np.uint8).reshape(original_shape)