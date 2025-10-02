import logging

from setuptools import find_packages, setup

# Setup logging
logging.basicConfig(level=logging.INFO)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mr3",
    version="1.0.0",
    author="David Anugraha",
    author_email="david.anugraha@gmail.com",
    description="mR3: Multilingual Rubric-Agnostic Reward Reasoning Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0 License",
    url="https://github.com/rubricreward/mr3",
    project_urls={
        "Bug Tracker": "https://github.com/rubricreward/mr3/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "vllm==0.8.5",
        "deepspeed==0.15.4",
        "datasets==3.6.0",
        "triton==3.2.0",
        "scipy",
        "pandas",
        "flash-attn==2.7.4.post1",
    ],
    package_dir={"": "src"},
    packages = find_packages("src"),
    python_requires=">=3.12",
)
