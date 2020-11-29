import pandas as pd
import glob
import os
from pathlib import Path

for n in range(1, 101):
    bdir = f'./example/bootstrap/bootstrap{n}/covariates'
    for f in glob.glob(f'{bdir}/*'):
        name = Path(f).stem
        df = pd.read_csv(f)
        df = df.rename(columns={'Qhat': 'Q', 'ghat': 'g'})
        df.to_csv(f'{bdir}/{name}.csv', index=False)
