"""
preprocessamento_utils.py
Alias para importacao do modulo 02_preprocessamento.
Permite que os outros scripts importem: from preprocessamento_utils import carregar_e_preparar
"""

import sys
import os

# Garante que o diretorio src esta no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importa diretamente do modulo original
from importlib import import_module

_mod = import_module("02_preprocessamento")
carregar_e_preparar = _mod.carregar_e_preparar
