from setuptools import find_packages, setup

constant="-e ."
def get_requirements(file_path):
    rts=[]
    with open(file_path) as file:
        rts=file.readlines()
        rts=[x.replace("\n","") for x in rts if x!=constant]

    return rts

setup(
    name='ml-project',
    version='0.0.1',
    author='Asmita Saha',
    author_email='sahaasmita60@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)