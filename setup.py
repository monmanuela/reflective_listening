import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='reflective_listening',
    version='0.1.2',
    description='Reflective listening statements via paraphrase generation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/monmanuela/reflective_listening',
    author='Monika Manuela Hengki',
    author_email='e0014971@u.nus.edu',
    license='MIT',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires='>=3.6.2',
    install_requires=[
        'language_tool_python',
        'nltk',
        'numpy',
        'sentence_transformers',
        'scipy==1.4.1',
        'sentencepiece',
        'torch',
        'transformers',
    ],
    zip_safe=False
)
