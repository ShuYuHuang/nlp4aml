
import hanlp
import opencc
import re
import pandas as pd

recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)


def is_aml_sentence(sen):
    KWS = ["假交易", "掏空", "人头", "吸金", "卷款", "落跑", "炒房", "遭指控", "遭控", "贿选", "毁约", "贿款", "起诉", "涉案", "串证", "声押", "羁押", "逃漏税", "逃税", "侵占", "犯罪", "涉嫌"
           "弊案", "行贿", "犯罪", "拐走", "骗子", "贪污", "传唤", "约谈", "诈骗", "诈贷", "毒品", "嫌犯", "主嫌"]

    pattern = '|'.join(KWS)
    m = re.search(pattern, sen)
    if m != None:
        return True
    return False


def get_aml_names(sen):
    l = recognizer(list(sen))
    names = set()
    for t in l:
        if t[1] == 'NR' and len(t[0]) > 1:
            converter = opencc.OpenCC('s2tw.json')
            names.add(converter.convert(t[0]))
            # names.add(t[0])
    return names


def query(news):
    converter = opencc.OpenCC('tw2s.json')
    print(news)
    news = converter.convert(news)
    print(news)
    ss = news.split('。')
    names = set()
    for s in ss:
        if(is_aml_sentence(s)):
            n = get_aml_names(s)
            names = names.union(n)
            # names.update(get_aml_names(s))
    return list(names)
