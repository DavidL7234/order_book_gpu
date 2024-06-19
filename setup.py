from setuptools import setup, find_packages

setup(
    name='order-book-gpu',
    version='0.1.0',
    packages=find_packages(),
    description='A package for managing order book data with GPU acceleration',
    install_requires=[
        'numpy',
        'requests',
        'websocket-client',
        'torch',
    ],
    python_requires='>=3.6',
    classifiers=[],
)
