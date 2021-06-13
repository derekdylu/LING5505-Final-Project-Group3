<h1 align='center'>
 LING 5505 Final Project
</h1>

<h2 align='center'>
    台灣升大學考試國寫情意題佳作研究
</h2>

<h3 align="center">3. 想不到組名</h3>

<h5 align="center">
    朱修平、盧德原、楊舒晴、陳宛瑩
</h5>

## Quick start

+ [Proposal](https://docs.google.com/document/d/1DvcNkbKt71XfwLikUxRwCfoIL6N7rdd8p-fO2EZian4/edit#heading=h.66htjfnm6yv3)
+ [Slides]
+ [Written Report]
+ [Demo site]
+ [data_set](https://github.com/derekdylu/LING5505-Final-Project-Group3/tree/main/data_set) folder for 208 raw materials
+ All materials of the demo site are stored in _[site](https://github.com/derekdylu/LING5505-Final-Project-Group3/tree/main/web/_site) folder
+ The project contains `.Rmd` and `.ipynb` files
  + make sure to install the required packages by following the instructions below before running the `.ipynb` file 
  + for `.Rmd` file, open it in [RStudio](https://www.rstudio.com) and run by following the instructions in the software

### Things required for .ipynb file

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
    CwnSenseTagger.download()
    CwnGraph.download()
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

