import json
import re

# Open and parse the notebook
with open('Momo - version3.ipynb', 'r') as f:
    data = json.load(f)

# Initialize containers for our analysis
imports = []
models = []
dataframes = []
visualizations = []
sections = []

# Analyze each cell
for i, cell in enumerate(data['cells']):
    if cell['cell_type'] == 'markdown':
        text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if re.search(r'^#+ ', text):
            sections.append(f'Section {i+1}: {text.strip()}')
    
    elif cell['cell_type'] == 'code':
        code = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        if re.search(r'import|from .* import', code):
            imports.append(code.strip())
            
        if re.search(r'(sklearn|xgboost|RandomForest|LogisticRegression|SVC|KNeighbors)', code):
            models.append(code.strip())
            
        if re.search(r'\.fit\(|\.predict\(', code):
            models.append(code.strip())
            
        if re.search(r'pd\.DataFrame|\.read_csv', code):
            dataframes.append(code.strip())
            
        if re.search(r'plot|plt\.|seaborn|sns\.|figure|bar|hist|scatter', code):
            visualizations.append(code.strip())

# Output analysis to a new file
with open('notebook_analysis.txt', 'w') as f:
    f.write('NOTEBOOK STRUCTURE ANALYSIS:\n')
    f.write(f'Total cells: {len(data["cells"])}\n')
    f.write(f'Code cells: {sum(1 for cell in data["cells"] if cell["cell_type"] == "code")}\n')
    f.write(f'Markdown cells: {sum(1 for cell in data["cells"] if cell["cell_type"] == "markdown")}\n\n')
    
    f.write('NOTEBOOK SECTIONS:\n')
    f.write('\n'.join(sections[:10]) + ('\n...' if len(sections) > 10 else '') + '\n\n')
    
    f.write('KEY IMPORTS (first 10):\n')
    unique_imports = list(set(imports))[:10]
    f.write('\n'.join(unique_imports) + ('\n...' if len(imports) > 10 else '') + '\n\n')
    
    f.write('DATAFRAME OPERATIONS (first 5):\n')
    f.write('\n'.join(dataframes[:5]) + ('\n...' if len(dataframes) > 5 else '') + '\n\n')
    
    f.write('MODEL RELATED CODE (first 5):\n')
    f.write('\n'.join(models[:5]) + ('\n...' if len(models) > 5 else '') + '\n\n')
    
    f.write('VISUALIZATIONS (first 5):\n')
    f.write('\n'.join(visualizations[:5]) + ('\n...' if len(visualizations) > 5 else '') + '\n')

print("Analysis complete. Results saved to notebook_analysis.txt") 