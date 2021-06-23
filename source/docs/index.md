# 台灣升大學考試國寫情意題佳作詞頻相關研究

+ 第三組 想不到組名
+ 組員：盧德原、朱修平、楊舒晴、陳宛瑩

## Python Project


```python
import numpy as np
import pandas as pd
import gdown
import os
import re
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf

import torch
import transformers

import CwnSenseTagger
#CwnSenseTagger.download()

import CwnGraph
#CwnGraph.download()

from CwnGraph import CwnBase
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

#import DistilTag
#DistilTag.download()
```


```python
ws = WS(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + '/data')
pos = POS(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + '/data')
ner = NER(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + '/data')
```

    /opt/anaconda3/lib/python3.8/site-packages/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py:909: UserWarning: `tf.nn.rnn_cell.LSTMCell` is deprecated and will be removed in a future version. This class is equivalent as `tf.keras.layers.LSTMCell`, and will be replaced by that in Tensorflow 2.0.
      warnings.warn("`tf.nn.rnn_cell.LSTMCell` is deprecated and will be "
    /opt/anaconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/base_layer_v1.py:1700: UserWarning: `layer.add_variable` is deprecated and will be removed in a future version. Please use `layer.add_weight` method instead.
      warnings.warn('`layer.add_variable` is deprecated and '


### 讀取與整理資料集


```python
all_f = []

for file in os.listdir("data_set/"):
    if file.endswith(".txt"): all_f.append(file)

'''
print("list length = ", len(all_f))
print(all_f[2])
print(len(all_f[2]))
'''
```




    '\nprint("list length = ", len(all_f))\nprint(all_f[2])\nprint(len(all_f[2]))\n'




```python
sentence_list = []
sentence_list_type = []
sentence_list_year = []

for i in all_f:
    f = open("data_set/" + i)
    sentence_list.append(f.read())
    
    if "GSAT" in i:
        sentence_list_type.append("GSAT")
    else:
        sentence_list_type.append("AST")
        
    year = re.search("\_(...|..)\_", i).group(1)
    sentence_list_year.append(year)
    
all_list = pd.DataFrame({'type': sentence_list_type, 'year': sentence_list_year, 'sentence': sentence_list})
all_list
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>year</th>
      <th>sentence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GSAT</td>
      <td>105</td>
      <td>這時，人也只能笑了。\n蘇迪勒的狂風橫掃全臺，它破壞了，也創造了。臺北市的兩個郵筒在一夜肆虐...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GSAT</td>
      <td>107</td>
      <td>秋天總給人一種蕭瑟之感，但我卻對秋天情有獨鍾。\n秋風颯颯，捲起了千堆落葉，也捲起了眾多遷客...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AST</td>
      <td>101</td>
      <td>它，蜿蜒過土壤的縫隙；它，寄身於大海的湛藍。它，摒持著儒者「原泉滾滾，不含晝夜。」的精神奔流...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GSAT</td>
      <td>101</td>
      <td>人生是一條長河，唯有堅硬的卵石才能激盪出美麗的水花，也唯有一份鍥而不舍的真情才能拓展生命的寬...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AST</td>
      <td>103</td>
      <td>最貧窮的人不是沒有錢財，而是沒有夢想。有了夢想就像是有了羅盤的航行；就像是有了心中的羅馬，只...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>203</th>
      <td>GSAT</td>
      <td>110</td>
      <td>如果我有一座新冰箱，那想必是清新可人，條理分明。因為冰藏在其間的，不僅是食物本身，更蘊含背後...</td>
    </tr>
    <tr>
      <th>204</th>
      <td>AST</td>
      <td>101</td>
      <td>人總免不了自己一個人的。有人怕寂寞，說：「寂寞，難耐。」他們怕孤獨，孤獨讓他們惶恐、害怕，感...</td>
    </tr>
    <tr>
      <th>205</th>
      <td>AST</td>
      <td>103</td>
      <td>我的祖父已經高齡九十二歲，他身邊的人正一個個離他而去。有一次和我聊天的時候，我從他口中委婉的...</td>
    </tr>
    <tr>
      <th>206</th>
      <td>AST</td>
      <td>105</td>
      <td>「順風可以航行，逆風可以飛行。」生命，是一連串的功課，其中之一，便是「舉重若輕」的學問。順境...</td>
    </tr>
    <tr>
      <th>207</th>
      <td>GSAT</td>
      <td>102</td>
      <td>音樂，飄揚於空中，那是聽覺的愉快；酸甜苦辣，撞擊味蕾，那是味覺的愉快；「沙鷗翔集，錦鱗游泳」...</td>
    </tr>
  </tbody>
</table>
<p>208 rows × 3 columns</p>
</div>



### ckiptagger


```python
word_sentence_list = ws(sentence_list)

pos_sentence_list = pos(word_sentence_list)

entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
```

### 斷詞結果顯示函式


```python
def print_word_pos_sentence (word_sentence, pos_sentence):
    assert len(word_sentence) == len(pos_sentence)
    for word, pos in zip(word_sentence, pos_sentence):
        print(f"{word}({pos})", end="\u3000")
    print()
    return

"""
for i, sentence in enumerate(sentence_list):
    print(f"'{sentence}'")
    print_word_pos_sentence(word_sentence_list[i],  pos_sentence_list[i])
    for entity in sorted(entity_sentence_list[i]):
        print(entity)
"""
```




    '\nfor i, sentence in enumerate(sentence_list):\n    print(f"\'{sentence}\'")\n    print_word_pos_sentence(word_sentence_list[i],  pos_sentence_list[i])\n    for entity in sorted(entity_sentence_list[i]):\n        print(entity)\n'



### 詞性頻率分析


```python
Vlist = []
Nalist = []

for i in range(len(all_list)):
    pos_df = pd.Series(pos_sentence_list[i]).value_counts().sort_index().rename_axis('CKIP_POS').reset_index(name = 'frequency')
    
    pos_cnt_V = pos_df[pos_df.CKIP_POS != 'V_2']
    pos_cnt_V = pos_cnt_V.loc[pos_cnt_V['CKIP_POS'].str.contains('V')]
    Vlist.append(pos_cnt_V.sum(numeric_only=True).sum())
    
    pos_cnt_N = pos_df.loc[pos_df['CKIP_POS'].str.contains('Na')]
    Nalist.append(pos_cnt_N.sum(numeric_only=True).sum())

all_list['V_cnt'] = Vlist
all_list['Na_cnt'] = Nalist
all_list['V_to_Na_ratio'] = all_list.V_cnt.div(Nalist)

all_list.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>year</th>
      <th>sentence</th>
      <th>V_cnt</th>
      <th>Na_cnt</th>
      <th>V_to_Na_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GSAT</td>
      <td>105</td>
      <td>這時，人也只能笑了。\n蘇迪勒的狂風橫掃全臺，它破壞了，也創造了。臺北市的兩個郵筒在一夜肆虐...</td>
      <td>103</td>
      <td>72</td>
      <td>1.430556</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GSAT</td>
      <td>107</td>
      <td>秋天總給人一種蕭瑟之感，但我卻對秋天情有獨鍾。\n秋風颯颯，捲起了千堆落葉，也捲起了眾多遷客...</td>
      <td>75</td>
      <td>43</td>
      <td>1.744186</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AST</td>
      <td>101</td>
      <td>它，蜿蜒過土壤的縫隙；它，寄身於大海的湛藍。它，摒持著儒者「原泉滾滾，不含晝夜。」的精神奔流...</td>
      <td>83</td>
      <td>83</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GSAT</td>
      <td>101</td>
      <td>人生是一條長河，唯有堅硬的卵石才能激盪出美麗的水花，也唯有一份鍥而不舍的真情才能拓展生命的寬...</td>
      <td>80</td>
      <td>56</td>
      <td>1.428571</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AST</td>
      <td>103</td>
      <td>最貧窮的人不是沒有錢財，而是沒有夢想。有了夢想就像是有了羅盤的航行；就像是有了心中的羅馬，只...</td>
      <td>99</td>
      <td>62</td>
      <td>1.596774</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GSAT</td>
      <td>98</td>
      <td>現在的我正攀爬著，在一片看不見頂端的山壁努力往上爬，我的雙頰充滿了認真而辛勤的色彩，不斷地有...</td>
      <td>116</td>
      <td>55</td>
      <td>2.109091</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GSAT</td>
      <td>103</td>
      <td>人生如寄，歲月如梭。我們總盼望於人生茫茫大海中，拾得內心底處最渴望的晶瑩珍珠。陶淵明所喜愛的...</td>
      <td>102</td>
      <td>74</td>
      <td>1.378378</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GSAT</td>
      <td>103</td>
      <td>古人謂：「天若有情，天亦老」，何況為我們奉獻近乎半生的父母？然而課業的繁重，與自我的迷茫叛逆...</td>
      <td>122</td>
      <td>62</td>
      <td>1.967742</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GSAT</td>
      <td>98</td>
      <td>每個人都有夢，一個從小到大細心呵護的夢，也許很多人親眼看見達到里程碑，但有更多人的夢永遠都只...</td>
      <td>70</td>
      <td>67</td>
      <td>1.044776</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AST</td>
      <td>103</td>
      <td>「將手中的燈提高一些吧！才能照亮後面的人。」海倫‧凱勒曾有此言。圓自己一個夢想，是我提著手中...</td>
      <td>123</td>
      <td>103</td>
      <td>1.194175</td>
    </tr>
  </tbody>
</table>
</div>



欄位依序表示為：指考或學測(type)、年份(year)、文章內容(sentence)、除V_2外所有動詞數量(V_cnt)、普通名詞數量(Na_cnt)、動詞對名詞的比例(V_to_Na_ratio)


```python
vn_df = all_list[['type', 'V_cnt', 'Na_cnt']]
xname = 'Na_cnt'
yname = 'V_cnt'
vn_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>V_cnt</th>
      <th>Na_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GSAT</td>
      <td>103</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GSAT</td>
      <td>75</td>
      <td>43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AST</td>
      <td>83</td>
      <td>83</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GSAT</td>
      <td>80</td>
      <td>56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AST</td>
      <td>99</td>
      <td>62</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>203</th>
      <td>GSAT</td>
      <td>119</td>
      <td>80</td>
    </tr>
    <tr>
      <th>204</th>
      <td>AST</td>
      <td>108</td>
      <td>55</td>
    </tr>
    <tr>
      <th>205</th>
      <td>AST</td>
      <td>108</td>
      <td>66</td>
    </tr>
    <tr>
      <th>206</th>
      <td>AST</td>
      <td>112</td>
      <td>60</td>
    </tr>
    <tr>
      <th>207</th>
      <td>GSAT</td>
      <td>130</td>
      <td>84</td>
    </tr>
  </tbody>
</table>
<p>208 rows × 3 columns</p>
</div>



#### 動詞對名詞散佈圖


```python
_ = sns.lmplot(x = xname, y = yname, data = vn_df, ci = None, hue = 'type')
```


    
![png](output_16_0.png)
    


由上述結果可以發現動詞普遍較名詞多的情況，而指考在這個現象的趨勢更為顯著。

#### 動詞對名詞簡單線性回歸

Let $y_1$ be the number of V words and $x_1$ be the number of N_a words in GSAT data set.<br>
The proposed model is,<br>
$y_1 = \beta_{10} + \beta_{11} x_1 + \epsilon_1$
<br><br>
Let $y_2$ be the number of V words and $x_2$ be the number of N_a words in AST data set.<br>
The proposed model is,<br>
$y_2 = \beta_{20} + \beta_{21} x_2 + \epsilon_2$<br><br>


```python
vn_df_gsat = vn_df[vn_df.type == 'GSAT']
vn_df_ast = vn_df[vn_df.type == 'AST']

print("--- GSAT ---")
result1 = smf.ols(yname + '~ ' + xname, data = vn_df_gsat).fit()
print(result1.summary())

b1_1 = result1.params[1]
b0_1 = result1.params[0]
print(f"Estimated model: y1 = {b0_1:.4f} + {b1_1:.4f} x1")

print("\n\n--- AST ---")
result2 = smf.ols(yname + '~ ' + xname, data = vn_df_ast).fit()
print(result2.summary())

b1_2 = result2.params[1]
b0_2 = result2.params[0]
print(f"Estimated model: y2 = {b0_2:.4f} + {b1_2:.4f} x2")
```

    --- GSAT ---
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  V_cnt   R-squared:                       0.068
    Model:                            OLS   Adj. R-squared:                  0.060
    Method:                 Least Squares   F-statistic:                     8.212
    Date:                Thu, 17 Jun 2021   Prob (F-statistic):            0.00496
    Time:                        13:05:05   Log-Likelihood:                -488.55
    No. Observations:                 115   AIC:                             981.1
    Df Residuals:                     113   BIC:                             986.6
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     84.8264      7.885     10.758      0.000      69.205     100.448
    Na_cnt         0.3055      0.107      2.866      0.005       0.094       0.517
    ==============================================================================
    Omnibus:                        1.252   Durbin-Watson:                   2.118
    Prob(Omnibus):                  0.535   Jarque-Bera (JB):                1.226
    Skew:                          -0.135   Prob(JB):                        0.542
    Kurtosis:                       2.573   Cond. No.                         366.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    Estimated model: y1 = 84.8264 + 0.3055 x1
    
    
    --- AST ---
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  V_cnt   R-squared:                       0.505
    Model:                            OLS   Adj. R-squared:                  0.500
    Method:                 Least Squares   F-statistic:                     92.93
    Date:                Thu, 17 Jun 2021   Prob (F-statistic):           1.45e-15
    Time:                        13:05:05   Log-Likelihood:                -375.58
    No. Observations:                  93   AIC:                             755.2
    Df Residuals:                      91   BIC:                             760.2
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     46.2938      6.124      7.559      0.000      34.129      58.459
    Na_cnt         0.8510      0.088      9.640      0.000       0.676       1.026
    ==============================================================================
    Omnibus:                        5.029   Durbin-Watson:                   1.645
    Prob(Omnibus):                  0.081   Jarque-Bera (JB):                5.690
    Skew:                           0.250   Prob(JB):                       0.0581
    Kurtosis:                       4.104   Cond. No.                         295.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    Estimated model: y2 = 46.2938 + 0.8510 x2


對指考與學測的動詞與名詞分佈分別進行簡單線性迴歸分析：<br>
可以發現學測資料的 $R^2=0.068$ 顯示該模型解釋力不足。<br>
兩資料的模型F檢定與參數t檢定皆為顯著。<br>
殘值分析等暫且忽略。<br>

#### 動詞對名詞比例Histogram


```python
plot = all_list.hist(column = 'V_to_Na_ratio')
print("ratio mean = ",all_list.V_to_Na_ratio.mean())
```

    ratio mean =  1.5512316568733755



    
![png](output_23_1.png)
    


畫出Histogram後可以發現比例的平均值為1.551左右，眾數也位在相近的位置，整體分佈呈現bell-shape。

### 單詞詞意計算函式


```python
cwn = CwnBase()
    
def all_sense_tree (word, verbal = False):
    cnt = 0
    for i in range(len(word)):
        snese_tree = word[i].senses
        cnt += len(word[i].senses)
        if(verbal == True): print(snese_tree)
    
    if(verbal == True): print("total senses = ", cnt)
    return cnt
```


```python
'''
_word = word_sentence_list[108][54]
word = cwn.find_lemma("^" + _word + "$")
print("word: ", _word)
all_sense_tree(word, verbal = True)
'''
```




    '\n_word = word_sentence_list[108][54]\nword = cwn.find_lemma("^" + _word + "$")\nprint("word: ", _word)\nall_sense_tree(word, verbal = True)\n'



### 單詞詞意量計算


```python
pun_set = {"COLONCATEGORY", "COMMACATEGORY", "DASHCATEGORY", "ETCCATEGORY", "EXCLAMATIONCATEGORY", "PARENTHESISCATEGORY",
          "PAUSECATEGORY", "PERIODCATEGORY", "QUESTIONCATEGORY", "SEMICOLONCATEGORY", "SPCHANGECATEGORY"}

all_senses_list = list()
all_senses_list_sum = list()

for i in range(all_list.shape[0]):
    senses_list = list()
    arr = pd.Series(word_sentence_list[i])
    ttl = 0
    for j in range(len(arr)):
        if(pos_sentence_list[i][j] not in pun_set):
            _word = arr[j]
            word = cwn.find_lemma("^" + _word + "$")
            sense_cnt = all_sense_tree(word)
            senses_list.append((arr[j], pos_sentence_list[i][j], sense_cnt))
            
    tp = sentence_list_type[i]
    year = sentence_list_year[i]
    all_senses_list.append((tp, year, senses_list))
    
    tmp_df = pd.DataFrame(senses_list, columns = ['tagged_word', 'CKIP_POS', 'sense_cnt'])
    words_cnt = tmp_df.shape[0]
    tmp_df_ = tmp_df[tmp_df['sense_cnt'] != 0]
    words_cnt_ = tmp_df_.shape[0]
    
    all_senses_list_sum.append((tp, year, words_cnt, tmp_df.sense_cnt.mean(), words_cnt_, tmp_df_.sense_cnt.mean()))

all_senses_df = pd.DataFrame(all_senses_list, columns = ['type', 'year', 'tagged_words'])
all_senses_df_sum = pd.DataFrame(all_senses_list_sum, columns = ['type', 'year', 'words_cnt', 'avg_sense_all', 'words_cnt_nonzero' ,'avg_sense_nonzero'])
all_senses_df_sum['zero_sense_ratio'] = all_senses_df_sum.words_cnt_nonzero.div(words_cnt)

all_senses_df_sum = all_senses_df_sum.assign(zero_sense_ratio = 1 - all_senses_df_sum['zero_sense_ratio'])
display(all_senses_df_sum.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>year</th>
      <th>words_cnt</th>
      <th>avg_sense_all</th>
      <th>words_cnt_nonzero</th>
      <th>avg_sense_nonzero</th>
      <th>zero_sense_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GSAT</td>
      <td>105</td>
      <td>399</td>
      <td>5.100251</td>
      <td>300</td>
      <td>6.783333</td>
      <td>0.353448</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GSAT</td>
      <td>107</td>
      <td>268</td>
      <td>5.313433</td>
      <td>193</td>
      <td>7.378238</td>
      <td>0.584052</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AST</td>
      <td>101</td>
      <td>345</td>
      <td>6.191304</td>
      <td>242</td>
      <td>8.826446</td>
      <td>0.478448</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GSAT</td>
      <td>101</td>
      <td>345</td>
      <td>6.011594</td>
      <td>264</td>
      <td>7.856061</td>
      <td>0.431034</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AST</td>
      <td>103</td>
      <td>394</td>
      <td>5.581218</td>
      <td>305</td>
      <td>7.209836</td>
      <td>0.342672</td>
    </tr>
  </tbody>
</table>
</div>


#### 平均詞意量Histogram


```python
print("average sense from all words = ",all_senses_df_sum.avg_sense_all.mean())
print("average sense from non-zero-sensed words = ",all_senses_df_sum.avg_sense_nonzero.mean())

bins = np.linspace(4, 11, 35)

plt.hist(all_senses_df_sum.avg_sense_all, bins, alpha=0.7, label='avg sense - all words')
plt.hist(all_senses_df_sum.avg_sense_nonzero, bins, alpha=0.7, label='avg sense - non-zero-sensed words')
plt.legend(loc = 'upper right', bbox_to_anchor=(1.7, 1))
plt.grid(True)
plt.show()
```

    average sense from all words =  5.9117085939980205
    average sense from non-zero-sensed words =  8.013096983316954



    
![png](output_31_1.png)
    


經過計算後可以得知在所有資料中，平均辭意為5.91；扣除辭意為0的單詞後平均辭意為8.01。

#### 冷僻詞使用分析

若我們假設詞意量為零之單詞是冷僻詞，經過計算可以得到每篇文章使用冷僻詞的比率：


```python
plot = all_senses_df_sum.hist(column = 'zero_sense_ratio')
print("ratio mean = ",all_senses_df_sum.zero_sense_ratio.mean())
```

    ratio mean =  0.3614161969496021



    
![png](output_35_1.png)
    


可見在所有資料集中，冷僻字的使用平均比例為36.14%。

### 特定詞性單詞出現頻率


```python
tmp_list = list()

for i in range(all_senses_df.shape[0]):
    tmp_list += all_senses_df.tagged_words[i]

senses_df = pd.DataFrame(tmp_list, columns = ['tagged_word', 'CKIP_POS', 'sense_cnt'])
senses_df_ = senses_df[senses_df['CKIP_POS'] != 'WHITESPACE']  #delete \n
senses_df_ = senses_df_[senses_df_['tagged_word'] != '。\n'] #delete 。\n

#senses_df_.head()
```


```python
words_cnt_ = pd.DataFrame(senses_df_.value_counts())
print("Head of frequency of all words")
display(words_cnt_.head(10))

words_cnt_Na = senses_df_[senses_df_.CKIP_POS == 'Na']
words_cnt_Na = pd.DataFrame(words_cnt_Na.value_counts())
print("\n\nHead of frequency of Na(普通名詞) words")
display(words_cnt_Na.head(10))

words_cnt_Nb = senses_df_[senses_df_.CKIP_POS == 'Nb']
words_cnt_Nb = pd.DataFrame(words_cnt_Nb.value_counts())
print("\n\nHead of frequency of Nb(專有名稱) words")
display(words_cnt_Nb.head(10))

words_cnt_V = senses_df_[senses_df_.CKIP_POS != 'V_2']
words_cnt_V = words_cnt_V.loc[words_cnt_V['CKIP_POS'].str.contains('V')]
words_cnt_V = pd.DataFrame(words_cnt_V.value_counts())
print("\n\nHead of frequency of V words (excluding V_2)")
display(words_cnt_V.head(10))

words_cnt_VA = senses_df_[senses_df_.CKIP_POS == 'VA']
words_cnt_VA = pd.DataFrame(words_cnt_VA.value_counts())
print("\n\nHead of frequency of VA(動作不及物動詞) words")
display(words_cnt_VA.head(10))
```

    Head of frequency of all words



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>0</th>
    </tr>
    <tr>
      <th>tagged_word</th>
      <th>CKIP_POS</th>
      <th>sense_cnt</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>的</th>
      <th>DE</th>
      <th>16</th>
      <td>7052</td>
    </tr>
    <tr>
      <th>我</th>
      <th>Nh</th>
      <th>3</th>
      <td>2705</td>
    </tr>
    <tr>
      <th>一</th>
      <th>Neu</th>
      <th>10</th>
      <td>1454</td>
    </tr>
    <tr>
      <th>是</th>
      <th>SHI</th>
      <th>9</th>
      <td>1362</td>
    </tr>
    <tr>
      <th>在</th>
      <th>P</th>
      <th>10</th>
      <td>1229</td>
    </tr>
    <tr>
      <th>了</th>
      <th>Di</th>
      <th>5</th>
      <td>836</td>
    </tr>
    <tr>
      <th>不</th>
      <th>D</th>
      <th>3</th>
      <td>699</td>
    </tr>
    <tr>
      <th>著</th>
      <th>Di</th>
      <th>8</th>
      <td>652</td>
    </tr>
    <tr>
      <th>自己</th>
      <th>Nh</th>
      <th>5</th>
      <td>601</td>
    </tr>
    <tr>
      <th>而</th>
      <th>Cbb</th>
      <th>10</th>
      <td>569</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Head of frequency of Na(普通名詞) words



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>0</th>
    </tr>
    <tr>
      <th>tagged_word</th>
      <th>CKIP_POS</th>
      <th>sense_cnt</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>人</th>
      <th>Na</th>
      <th>11</th>
      <td>481</td>
    </tr>
    <tr>
      <th>心</th>
      <th>Na</th>
      <th>13</th>
      <td>247</td>
    </tr>
    <tr>
      <th>生命</th>
      <th>Na</th>
      <th>8</th>
      <td>234</td>
    </tr>
    <tr>
      <th>人生</th>
      <th>Na</th>
      <th>1</th>
      <td>206</td>
    </tr>
    <tr>
      <th>夢想</th>
      <th>Na</th>
      <th>0</th>
      <td>107</td>
    </tr>
    <tr>
      <th>學生</th>
      <th>Na</th>
      <th>2</th>
      <td>86</td>
    </tr>
    <tr>
      <th>逆境</th>
      <th>Na</th>
      <th>0</th>
      <td>83</td>
    </tr>
    <tr>
      <th>夢</th>
      <th>Na</th>
      <th>3</th>
      <td>81</td>
    </tr>
    <tr>
      <th>郵筒</th>
      <th>Na</th>
      <th>0</th>
      <td>78</td>
    </tr>
    <tr>
      <th>生活</th>
      <th>Na</th>
      <th>3</th>
      <td>78</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Head of frequency of Nb(專有名稱) words



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>0</th>
    </tr>
    <tr>
      <th>tagged_word</th>
      <th>CKIP_POS</th>
      <th>sense_cnt</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>孔子</th>
      <th>Nb</th>
      <th>3</th>
      <td>13</td>
    </tr>
    <tr>
      <th>項羽</th>
      <th>Nb</th>
      <th>0</th>
      <td>11</td>
    </tr>
    <tr>
      <th>李白</th>
      <th>Nb</th>
      <th>2</th>
      <td>11</td>
    </tr>
    <tr>
      <th>蘇迪勒</th>
      <th>Nb</th>
      <th>0</th>
      <td>10</td>
    </tr>
    <tr>
      <th>蘇軾</th>
      <th>Nb</th>
      <th>0</th>
      <td>8</td>
    </tr>
    <tr>
      <th>柳宗元</th>
      <th>Nb</th>
      <th>0</th>
      <td>8</td>
    </tr>
    <tr>
      <th>柯麥隆</th>
      <th>Nb</th>
      <th>0</th>
      <td>8</td>
    </tr>
    <tr>
      <th>陶淵明</th>
      <th>Nb</th>
      <th>0</th>
      <td>7</td>
    </tr>
    <tr>
      <th>李安</th>
      <th>Nb</th>
      <th>0</th>
      <td>7</td>
    </tr>
    <tr>
      <th>史記</th>
      <th>Nb</th>
      <th>1</th>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Head of frequency of V words (excluding V_2)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>0</th>
    </tr>
    <tr>
      <th>tagged_word</th>
      <th>CKIP_POS</th>
      <th>sense_cnt</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>讓</th>
      <th>VL</th>
      <th>6</th>
      <td>183</td>
    </tr>
    <tr>
      <th>使</th>
      <th>VL</th>
      <th>13</th>
      <td>181</td>
    </tr>
    <tr>
      <th>說</th>
      <th>VE</th>
      <th>16</th>
      <td>132</td>
    </tr>
    <tr>
      <th>大</th>
      <th>VH</th>
      <th>28</th>
      <td>120</td>
    </tr>
    <tr>
      <th>面對</th>
      <th>VC</th>
      <th>2</th>
      <td>119</td>
    </tr>
    <tr>
      <th>沒有</th>
      <th>VJ</th>
      <th>6</th>
      <td>107</td>
    </tr>
    <tr>
      <th>想</th>
      <th>VE</th>
      <th>9</th>
      <td>106</td>
    </tr>
    <tr>
      <th>深</th>
      <th>VH</th>
      <th>17</th>
      <td>88</td>
    </tr>
    <tr>
      <th>看</th>
      <th>VC</th>
      <th>13</th>
      <td>88</td>
    </tr>
    <tr>
      <th>愉快</th>
      <th>VH</th>
      <th>1</th>
      <td>87</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Head of frequency of VA(動作不及物動詞) words



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>0</th>
    </tr>
    <tr>
      <th>tagged_word</th>
      <th>CKIP_POS</th>
      <th>sense_cnt</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>應變</th>
      <th>VA</th>
      <th>0</th>
      <td>45</td>
    </tr>
    <tr>
      <th>來</th>
      <th>VA</th>
      <th>19</th>
      <td>44</td>
    </tr>
    <tr>
      <th>通關</th>
      <th>VA</th>
      <th>0</th>
      <td>39</td>
    </tr>
    <tr>
      <th>度人</th>
      <th>VA</th>
      <th>0</th>
      <td>36</td>
    </tr>
    <tr>
      <th>存在</th>
      <th>VA</th>
      <th>3</th>
      <td>32</td>
    </tr>
    <tr>
      <th>笑</th>
      <th>VA</th>
      <th>2</th>
      <td>30</td>
    </tr>
    <tr>
      <th>歪腰</th>
      <th>VA</th>
      <th>0</th>
      <td>29</td>
    </tr>
    <tr>
      <th>站</th>
      <th>VA</th>
      <th>7</th>
      <td>23</td>
    </tr>
    <tr>
      <th>走</th>
      <th>VA</th>
      <th>20</th>
      <td>21</td>
    </tr>
    <tr>
      <th>坐</th>
      <th>VA</th>
      <th>12</th>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>


以上我們統計了幾個特定詞性在所有資料集的單詞出現頻率。

## 連結

+ 回到[R Project 研究結果頁面](https://derekdylu.github.io/LING5505-Final-Project-Group3/source/web/index.html)
+ 回到[入口頁面](https://derekdylu.github.io/LING5505-Final-Project-Group3/index.html)
