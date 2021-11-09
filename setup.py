from setuptools import setup

setup(name='austen_plots',
      version='0.1.0',
      description='Sensitivity analysis for causal inference',
      url='https://github.com/anishazaveri/austen_plots',
      author='Anisha Zaveri',
      author_email='anishazaveri@gmail.com',
      license='MIT',
      packages=['austen_plots'],
      long_description="See [GitHub repo](https://github.com/anishazaveri/austen_plots) for more information",
      long_description_content_type='text/markdown',
      classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8"],
      install_requires=['pandas', 'plotnine', 'numpy', 'scikit-learn', 'scipy', 'tqdm'],
      zip_safe=False)
