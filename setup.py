import pathlib
from setuptools import setup,find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="deploytestm7",
    version="1.0.0",
    description="classes",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/auddy99/Gesture-Detection",
    author="Nidhi Singh,Auddy Soumyadeep,Mayank Raj",
    author_email="rajstarts35325@gmail.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["absl-py==1.0.0","attrs==21.4.0","cycler==0.11.0","fonttools==4.30.0","joblib==1.1.0","kiwisolver==1.3.2","matplotlib==3.5.1","mediapipe==0.8.9.1","numpy==1.22.3"
,"opencv-contrib-python==4.5.5.64","opencv-python==4.5.5.64","packaging==21.3","pandas==1.4.1","Pillow==9.0.1","protobuf==3.19.4","pyparsing==3.0.7","python-dateutil==2.8.2","pytz==2021.3","scikit-learn==1.0.2","scipy==1.8.0","six==1.16.0","threadpoolctl==3.1.0"],
)