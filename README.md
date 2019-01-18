# interp_gan

### 環境
python==3.5.6  
keras==2.2.0  
matplotlib==2.2.2  
numpy==1.14.3  
scipy==1.1.0  
tensorflow==1.12.0  
Pillow==5.1.0

### 設定
confing.pyにデータが入ったディレクトリまでのパスを設定する。  
"SAVE_PATH"を設定しmain.pyを動かすと、そのディレクトリ以下に/completion_net , /discriminator  
のディレクトリが作成される


### 実行方法

生成器のみ学習
```
python main.py -c
```

識別器のみ学習
```
python main.py -d
```

生成器、識別器同時に学習
```
python main.py -cd
```

テスト用データの補間
```
python main.py -t
```

実際に線検出したデータの補間
```
python main.py -i
```
