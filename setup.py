from setuptools import setup


setup(
    name='LocalRQA',
    version='0.1',
    url='http://github.com/jasonyux/LocalRQA',
    author='Flying Circus',
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
        'langchain',
        'llama-index',
        'rouge_score',
        'rank_bm25',
        'pytrec_eval',
        'flash_attn>=2.1.0',
        'gradio==3.50.2',
        'gradio_client==0.6.1'
    ],
    zip_safe=False
)