import wget, os, gzip, pickle, random, re, os, gzip, re, string

from tqdm import trange
from collections import Counter

import torch

import random
from random import choice

PAD, START, END, UNK = '.pad', '.start', '.end', '.unk'

SENT = '_s'
TOY = {
    '_s': ['_s _adv', '_np _vp', '_np _vp _prep _np', '_np _vp ( _prep _np )', '_np _vp _con _s','_np _vp ( _con _s )'],
    '_adv': ['briefly', 'quickly', 'impatiently'],
    '_np': ['a _noun', 'the _noun', 'a _adj _noun', 'the _adj _noun'],
    '_prep': ['on', 'with', 'to', 'for', 'at'],
    '_con': ['while', 'but'],
    '_noun': ['mouse', 'bunny', 'cat', 'dog', 'man', 'woman', 'person', 'bear', 'koala', 'judge', 'businessman',
        'businesswoman', 'lawyer', 'teacher', 'engineer'],
    '_vp': ['walked', 'walks', 'ran', 'runs', 'goes', 'went', 'hiked'],
    '_adj': ['short', 'quick', 'busy', 'nice', 'gorgeous', 'spectacular', 'reluctant', 'systematic', 'willowy', 'engaged', 'synthetic']
}

def t(blist):
    return torch.tensor([int(b) for b in blist], dtype=torch.uint8)

def gen_sentence(sent=SENT, g=TOY):

    symb = '_[a-z]*'

    while True:

        match = re.search(symb, sent)
        if match is None:
            return sent

        s = match.span()
        sent = sent[:s[0]] + random.choice(g[sent[s[0]:s[1]]]) + sent[s[1]:]

def load_toy(ntrain=100_000, ntest=20_000, to_torch=True, final=False, seed=0):
    """
    Generates language from a toy grammar.
    :param ntrain:
    :param ntest:
    :param to_torch: Whether to return torch tensors (if false, returns python lists)
    :param final: Whether to return the test set or the validation set (True for test)
    :return:
    """

    random.seed(seed)

    train, test = '', ''
    while len(train) < ntrain:
        train += gen_sentence() + ' . '

    random.seed(seed if final else seed + 1)
    # -- change the seed so we get different test/val sets depending on `final`

    while len(test) < ntest:
        test += gen_sentence() + ' . '

    ctr = Counter(train + test)
    i2t = [PAD, START, END, UNK] + [t for t, _ in ctr.most_common()]
    t2i = { w : i for  i, w in enumerate(i2t)}

    train = [t2i[t] for t in train]
    test  = [t2i[t] for t in test]
    
    if to_torch:
        return (t(train), t(test)), (i2t, t2i)

    return (train, test), (i2t, t2i)
