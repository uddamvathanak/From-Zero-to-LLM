# Applications 🚀

This section explores real-world use cases of Large Language Models (LLMs) through hands-on projects and implementations.

# Installation
Follow these steps to set up your environment and install the necessary dependencies for working with Large Language Models (LLMs).

## Step 1: Install Conda (if not already installed)
- Download and install Miniconda or Miniforge.
- Verify the installation with

```bash
conda --version
```

## Step 2: create a New Conda Environment

- Open a terminal or command prompt.
- Run the following command to create a new environment named llm-env with Python 3.9:

```bash
conda create -n llm-env python=3.9
```

- Activate the environment:

```bash
conda activate llm-env
```

## Step 3: Install Dependencies
Install the required Python Packages (We may need to refer to how to install each package based on their official documentation website such as Pytorch or Hugging face).

For example:

```bash
python -m pip install transformers datasets accelerate torch
```

For GPU installation (recommended but optional).

Refer to Pytorch website.

## Step 4: Verify Installation
Run the following command to check that the libraries are installed correctly:

```bash
python -c "import transformers; print(transformers.__version__)"
```
You should see the version of the Transformers library displayed.

## Step 5: Clone the Repository
Clone this repository to your local machine:

```bash
git clone https://github.com/uddamvathanak/From-Zero-to-LLM.git

cd From-Zero-to-LLM
```

# Applications Covered
1. [Text Classification](text_classification.ipynb)
2. [Text Summarization](text_summarisation.ipynb)
3. [Question Answering](text_QA.ipynb)
4. Code Generation
5. Conversational Agents

# Troubleshooting
If you encounter issues during installation, ensure that:
- Conda is correctly installed and added to your system PATH.
- You have Python 3.9 or a compatible version installed.
- Your CUDA driver is up to date for GPU support.