from pathlib import Path

datasets = Path('datasets')

for dir in datasets.iterdir():
    label = datasets / dir.name

    for file in label.iterdir():
        if '.mat' not in file.name:
            file.unlink()