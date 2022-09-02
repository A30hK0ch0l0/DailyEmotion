from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

setup(
    name='user_cluster',
    version='1.0',
    packages=find_packages(exclude=('tests',)),
    include_package_data=True,
    package_data={'': ['files/*', '*/*', '*/*/*', '*/*/*/*']},
    description='Inference User cluster',
    long_description=long_description,
    install_requires=requirements,
    license="MIT",
    author='Lifeweb',
    author_email='info@lifeweb.ir',
    url='https://lifeweb.ir',
    maintainer='Infrastructure',
    platforms='any'
)
