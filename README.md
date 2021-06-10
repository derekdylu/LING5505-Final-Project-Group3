<h1 align='center'>
 LING 5505 Final Project
</h1>

<h2 align='center'>
    大考作文佳作分析
</h2>

<h3 align="center">Group 3</h3>

<h5 align="center">
    小組名單
</h5>

## Notes

+ [Proposal](https://docs.google.com/document/d/1DvcNkbKt71XfwLikUxRwCfoIL6N7rdd8p-fO2EZian4/edit#heading=h.66htjfnm6yv3)
+ 208篇學測與指考作文佳作儲存於data_set資料夾
+ 使用R與Python製作

### Things required for python project

  You have to download all the potentially required package according to the commands below,

  **(1) Install required package**
  ```py
    !pip install gdown
    !pip install tensorflow
    !pip install torch
    !pip install transformers
    !pip install CwnSenseTagger
    !pip install CwnGraph
    !pip install ckiptagger
    !pip install CwnSenseTagger.download()
    !pip install CwnGraph.download()
  ```

  **(2) Download required file for ckiptagger**: The commands below let you to save the downloaded file outside the folder that connected to GitHub repository since the file is quite big (about 2GB)
  ```py
    import os
    data_utils.download_data_url(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)))
  ```

  **(3) Import them**
  ```py
    import numpy as np
    import pandas as pd
    import gdown
    
    import torch
    import transformers
    
    import CwnSenseTagger
    CwnSenseTagger.download()

    import CwnGraph
    CwnGraph.download()
    
    from CwnGraph import CwnBase
    from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
  ```
  **(4) Set the ckiptagger worker**
  ```py
    ws = WS(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + '/data')
    pos = POS(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + '/data')
    ner = NER(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + '/data')
  ```

  **(5) For more information, visit [ckiplab/ckiptagger](https://github.com/ckiplab/ckiptagger)**

## Milestones

