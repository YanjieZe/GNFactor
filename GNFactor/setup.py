from setuptools import setup, find_packages

setup(
    name='gnfactor',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    description="GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields",
    author='Yanjie Ze',
    author_email='lastyanjieze@gmail.com',
    url='https://yanjieze.com/GNFactor/',
    keywords=['NeRF', 'Behavior-Cloning', 'Langauge', 'Robotics', 'Generalizable Manipulation'],
)