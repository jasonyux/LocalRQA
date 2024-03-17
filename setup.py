from setuptools import setup


setup(
    name='LocalRQA',
    version='0.1',
    url='http://github.com/jasonyux/LocalRQA',
    author='Xiao Yu, Yunan Lu',
    author_email='jasonyux17@gmail.edu',
    license='MIT',
    packages=['local_rqa'],
    install_requires=[
        'transformers>=4.36.2',
        'sentence-transformers',
        'evaluate',
        'sentencepiece',
        'deepspeed',
        'accelerate',
        'bitsandbytes',
        'openai==1.2.4',
        'tiktoken',
        'wandb',
        'langchain>=0.1.12',
        'llama-index',
        'faiss-gpu',
        'rouge_score',
        'rouge',
        'rank_bm25',
        'jsonlines',
        'better-abc',
        'retry',
        'text-generation',
        'selenium',
        'unstructured',
        'pytrec_eval',
        'packaging',
        'torch',
        'flash_attn>=2.1.0',
        'gradio==3.50.2',
        'gradio_client==0.6.1'
    ],
    zip_safe=False
)