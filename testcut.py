# -*- coding: utf-8 -*-

"""code from https://colab.research.google.com/drive/1hdtmwTXHLrqNmDhDqHnTQGpDVy1aJc4t#scrollTo=MZul6NbWHqit https://docs.google.com/presentation/d/1zsn-DqoWm2HNPxjuJvjPB3qS892QmNko2Sjr8ZwFmOc/edit#slide=id.g3c6ecf03c1_0_162"""


from itertools import chain
import sklearn_crfsuite
# Character types adapted from Haruechaiyasak et al. 2008.

# Character that can be the final consonant in a word
chartype_c = '\u0e01\u0e02\u0e03\u0e04\u0e06\u0e07\u0e08\u0e0a\u0e0b\u0e0d\u0e0e\u0e0f' + \
  '\u0e10\u0e11\u0e12\u0e13\u0e14\u0e15\u0e16\u0e17\u0e18\u0e19\u0e1a\u0e1b\u0e1e\u0e1f' + \
  '\u0e20\u0e21\u0e22\u0e23\u0e24\u0e25\u0e26\u0e27\u0e28\u0e29\u0e2a\u0e2c\u0e2d'

# Character that cannot be the final consonant in a word
chartype_n = '\u0e05\u0e09\u0e0c\u0e1c\u0e1d\u0e2b\u0e2e'

# Vowel that cannot begin a word
chartype_v = '\u0e30\u0e31\u0e32\u0e33\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39\u0e45\u0e47'

# Vowel that can begin a word
chartype_w = '\u0e40\u0e41\u0e42\u0e43\u0e44'

# Combining symbol
chartype_s = '\u0e3a\u0e4c\u0e4d\u0e4e'

# Standalone symbol
chartype_a = '\u0e2f\u0e46\u0e4f\u0e5a\u0e5b'

# Tone marks
chartype_t = '\u0e48\u0e49\u0e4a\u0e4b'

# Digit character
chartype_d = '0123456789\u0e50\u0e51\u0e52\u0e53\u0e54\u0e55\u0e56\u0e57\u0e58\u0e59'

# Currency character
chartype_b = '$฿'

# Quote character
chartype_q = '\'\"'

# Other character
chartype_o = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Space character inside a word
# chartype_p

# Space character
chartype_z = ' \u00a0'

# Undefined
# chartype_x

tags = [
    ('c', chartype_c),
    ('n', chartype_n),
    ('v', chartype_v),
    ('w', chartype_w),
    ('s', chartype_s),
    ('a', chartype_a),
    ('t', chartype_t),
    ('d', chartype_d),
    ('b', chartype_b),
    ('q', chartype_q),
    ('o', chartype_o),
    ('z', chartype_z)
]

for tag in tags:
    print("Type {}: {}".format(tag[0], tag[1]))


from typing import *
# Get only labels (use in evaluation)
def get_labels(doc) -> List[str]:
    return [label for (c, label) in doc]

# Character type
def get_ctype(c: str) -> str:
    for tag in tags:
        if c in tag[1]:
            return tag[0]
    return 'x'
def char2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'char': word,
        'ctype': get_ctype(word)
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
           '-1:char': word1,
            '-1:ctype': get_ctype(word1)
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:char': word1,
            '+1:ctype': get_ctype(word1)
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [char2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token,label in sent]


def cutok(txt):
 t=crf2.predict([sent2features(txt)])[0]
 listtext=list(txt)
 strdata=""
 for j in list(zip(listtext,t)):
  if j[1]=="B" and strdata!="":
   strdata+="|"+j[0]
  strdata+=j[0]
 return strdata
crf2 = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=500,
    all_possible_transitions=True,
    model_filename="full500.model0"
)
