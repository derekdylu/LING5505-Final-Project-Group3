<h1 align='center'>
 LING 5505 Final Project
</h1>

<h2 align='center'>
    大考作文佳作分析
</h2>

<h3 align="center">Group 3</h3>

<!-- <div align="center">
  <a href="https://icheft.github.io/MGT2001-Final-Project/">
    <img src="https://img.shields.io/badge/MGT2001-網頁版請點此-green?style=for-the-badge&logo=python">
  </a>
</div> -->

<h5 align="center">
小組名單
</h5>


## Notes

+ All files should be named `Stat_2020_b_2.xxx`
+ Brainstorming 在 [Google Spreadsheet](https://docs.google.com/spreadsheets/d/109Qs8u3Vit_T6Y6TnVSWRkA9TWvFKvzJEdva9StlEfE/edit#gid=0)

### Things to Do Before Geting Started

1. Everyone should have their local Python packages identical

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
  **(3) Set the ckiptagger worker**
  ```py
    ws = WS(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + '/data')
    pos = POS(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + '/data')
    ner = NER(os.path.abspath(os.path.join(os.getcwd(), os.path.pardir)) + '/data')
  ```
  
  **(4) For more information, visit [ckiplab/ckiptagger](https://github.com/ckiplab/ckiptagger)

## Milestones

