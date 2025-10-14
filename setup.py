from setuptools import setup, find_packages

setup(
    name="libsolver",
    version="0.1.0",
    author="Francisco Paulino, José Vinícius Evangelista Dias de Souza, Carlos Ryan Santos Silva",
    author_email="fpaulinofilho04@gmail.com, ryan.silvatrei@gmail.com, jv.eds2003@gmail.com",
    description="Pacote criado a partir de um repositório do GitHub",
    long_description_content_type="text/markdown",
    url="https://github.com/Francisco-Paulino-Arruda-Filho/LibSolver",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "plotly",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="optimization solver algorithms heuristics metaheuristics",
)
