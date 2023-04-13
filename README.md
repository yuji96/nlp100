# Requirements

- [brew](https://brew.sh/)

# Setup

## 04. 形態素解析

```zsh
brew install mecab mecab-ipadic
```
```zsh
# usage
mecab < neko.txt > neko.txt.mecab
```

## 05. 係り受け解析

```zsh
brew install crf++ cabocha graphviz
pip install beautifulsoup4 lxml graphviz
```
```zsh
# usage
cabocha < ai.ja.txt > ai.ja.txt.parsed
```

## 06. 機械学習
```zsh
pip install gensim
```