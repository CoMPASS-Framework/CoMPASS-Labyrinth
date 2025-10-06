from setuptools import setup, find_packages

setup(
    name='CoMPASS',
    version='0.5.0',
    description='A Behavioral Modeling Toolkit using Hierarchical HMMs for assessing Goal-Directed Navigation',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    author='Shreya Bangera',
    author_email='',
    packages=find_packages(),  # automatically finds submodules
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'hmmlearn',
        'scipy',
        'geopandas',
        'plotly'
    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
