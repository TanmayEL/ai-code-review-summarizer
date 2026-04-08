from setuptools import setup, find_packages

setup(
    name="multimodal-code-summarizer",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    author="Tanmay Eknath Lotankar",
    author_email="tanmay.elotankar@gmail.com",
    description="RAG + Claude pipeline for code review summarization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TanmayEL/multimodal-code-summarizer",
    python_requires=">=3.9",
)
