from setuptools import setup, find_packages

setup(
    name="seedvox",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "sphn",
        "torch",
        "torchaudio",
        "transformers",
        "tqdm",
        "numpy",
        "g2p_en",
        "phonemizer",
        "safetensors",
    ],
    python_requires=">=3.9",
)
