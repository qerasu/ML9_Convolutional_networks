# ML9_Convolutional_networks

Проект для обучения и сравнения сверточных нейронных сетей на датасете изображений жестов — задача многоклассовой классификации.

Реализованы две архитектуры: сеть `LeNet5` и предобученная `ResNet18` из библиотеки `timm`. Проведены эксперименты с различными методами аугментации данных (базовые пространственные и цветовые преобразования через `albumentations`, MixUp, CutMix), а также применена аугментация на этапе тестирования (Test-Time Augmentation).

## Cтруктура проекта

```text
ML9_Convolutional_networks/
├── datasets/                           # Данные для обучения
├── src/
│   ├── main.ipynb                      # Использование прописанной логики
│   ├── data/                        
│   │   ├── datasets.py                 # PyTorch Dataset классы
│   │   └── preprocess.py               # Препроцессинг и разбиение датасета (train/val)
│   └── model/                        
│       ├── augmentations.py            # Настройки Albumentations, MixUp, CutMix
│       ├── models.py                   # Архитектуры нейронных сетей
│       ├── train.py                    # Циклы обучения и валидации модели
│       └── tta.py                      # Реализация Test-Time Augmentation
└── README.md
```

## Установка зависимостей

MacOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Запуск

Ноутбук использует относительные пути `../datasets/`, `data/` и `model/`, поэтому ячейки нужно запускать из директории `src/`.