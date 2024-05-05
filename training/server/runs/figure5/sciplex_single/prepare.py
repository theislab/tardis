import nbformat
from nbformat import NotebookNode
import os

def load_notebook(path: str) -> NotebookNode:
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    return nb

def save_notebook(nb: NotebookNode, filename: str) -> None:
    with open(filename, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def increment_seed(nb: NotebookNode, var_name: str, start_seed: int, end_seed: int, identifier: int) -> None:
    for new_seed in range(start_seed, end_seed):
        for cell in nb.cells:
            if cell.cell_type == 'code':
                # Find the cell containing the variable and set the new value
                if var_name in cell.source:
                    lines = cell.source.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith(f'{var_name} ='):
                            # Replace the line with the new seed value
                            lines[i] = f'{var_name} = {new_seed}'
                            cell.source = '\n'.join(lines)
                            break
                    # Save the notebook with the new seed value
                    new_filename = f'notebook_{identifier}_seed_{new_seed}.ipynb'
                    save_notebook(nb, new_filename)
                    print(f'Saved: {new_filename}')
                    break

for identifier in [1, 2]:

    # Path to the original notebook
    original_notebook_path = f'/home/icb/kemal.inecik/work/codes/tardis/training/server/runs/figure5/sciplex_single/base/sciplex_dose_{identifier}.ipynb'

    # Load the original notebook
    nb = load_notebook(original_notebook_path)

    # Increment seed from 0 to 21
    increment_seed(nb, 'ood_seed', 0, 14, identifier=identifier)
