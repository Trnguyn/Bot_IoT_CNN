import subprocess
import sys

print("Installing PyTorch...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
print("PyTorch installed successfully!")
