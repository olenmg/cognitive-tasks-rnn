# cognitive-tasks-rnn
Simple re-implementation of Yang, et al. Task representations in neural networks trained to perform many cognitive tasks. (2019) using PyTorch


## Dependencies
- Python 3.9.7
- PyTorch 1.10.1
- Gym==0.17.3
- [NeuroGym](https://github.com/neurogym/neurogym)

**NeuroGym Installation**
```shell
git clone https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
```


## Usage
**Train**
```shell
python main.py
```

## References
- [neurogym/ngym_usage](https://github.com/neurogym/ngym_usage)
- Yang, Guangyu Robert, Madhura R. Joglekar, H. Francis Song, William T. Newsome, and Xiao-Jing Wang. ‘Task Representations in Neural Networks Trained to Perform Many Cognitive Tasks’. Nature Neuroscience 22, no. 2 (February 2019): 297–306.