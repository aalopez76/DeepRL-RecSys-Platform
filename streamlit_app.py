import os
import sys

# Definir la raíz y la carpeta src
root_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(root_path, "src")

# Añadir ambas al path de búsqueda de Python
sys.path.append(root_path)
sys.path.append(src_path)

# Ahora la importación funcionará sin el prefijo 'src.'
from deeprl_recsys.ui.app import main

if __name__ == "__main__":
    main()
