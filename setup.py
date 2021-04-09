import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='reflective_listening',
    version='0.1',
    description='Reflective listening statements via paraphrase generation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/monmanuela/reflective_listening',
    author='Monika Manuela Hengki',
    author_email='e0014971@u.nus.edu',
    license='MIT',
    packages=['reflective_listening'],
    zip_safe=False
)
