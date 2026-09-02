$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python -m pip install --upgrade pip
python -m pip install --force-reinstall --no-deps torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install pytorchvideo fvcore iopath
python -m pip install "numpy>=1.26,<3" "pandas>=2.2,<3" "pyarrow>=16,<22" "scipy>=1.13,<2" "scikit-learn>=1.5,<2" "statsmodels>=0.14,<0.15" "PyYAML>=6,<7" "Pillow>=10,<12" "tqdm>=4.66,<5" "matplotlib>=3.9,<4" "seaborn>=0.13,<0.14" "socksio>=1.0,<2" "pytest>=8,<9" "pytest-cov>=5,<7"
python -m pip install --no-deps "timm>=1.0,<2"
python -m pip install --no-deps -e '..'
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
python -m pip freeze | Set-Content -Encoding utf8 requirements-lock.txt
