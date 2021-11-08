from setuptools import setup
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README_PyPi.md").read_text()

setup(name='austen_plots',
      version='0.1.2',
      description='Sensitivity analysis for causal inference',
      url='https://github.com/anishazaveri/austen_plots',
      author='Anisha Zaveri',
      author_email='anishazaveri@gmail.com',
      license='MIT',
      packages=['austen_plots'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8"],
      install_requires=['pandas', 'plotnine', 'numpy', 'scikit-learn', 'scipy', 'tqdm'],
      zip_safe=False)
