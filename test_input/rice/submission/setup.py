from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="statemask-reproduction",
    version="1.0.0",
    author="StateMask Reproduction Team",
    author_email="reproduction@statemask.ai",
    description="Reproduction of StateMask: Explaining Deep Reinforcement Learning through State Mask",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/statemask/reproduction",
    packages=find_packages(exclude=['tests*', 'docs*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800',
            'pre-commit>=2.0',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5',
            'sphinxcontrib-napoleon>=0.7',
        ],
        'experiments': [
            'wandb>=0.12',
            'tensorboard>=2.8',
            'matplotlib>=3.5',
            'seaborn>=0.11',
            'plotly>=5.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'statemask-train=statemask.scripts.train:main',
            'statemask-explain=statemask.scripts.explain:main',
            'statemask-refine=statemask.scripts.refine:main',
            'statemask-evaluate=statemask.scripts.evaluate:main',
            'statemask-reproduce=statemask.scripts.reproduce:main',
        ],
    },
    package_data={
        'statemask': [
            'configs/*.yaml',
            'configs/*.json',
            'data/*.pkl',
            'data/*.npz',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'reinforcement learning',
        'explainable ai',
        'deep learning',
        'state masking',
        'interpretability',
        'mujoco',
        'atari',
        'stable baselines3',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/statemask/reproduction/issues',
        'Source': 'https://github.com/statemask/reproduction',
        'Documentation': 'https://statemask-reproduction.readthedocs.io/',
        'Paper': 'https://arxiv.org/abs/statemask',
    },
)