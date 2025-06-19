```
git clone https://github.com/Pozovi23/for_ML_contest.git
cd for_ML_contest
```

создаете и активируете виртуальное окружение 
(там кто как его создает, но у меня так: 
```
python3 -m venv venv
source venv/bin/activate
```
)

и в виртуальном окружении ставите зависимости
```pip install -r requirements.txt```

Пример запуска на одной фотке
```python3 main.py "test/train_BLA_0006_JPG.rf.994cd3b5e58adc3c8a7659979139b424.jpg" --file_dir_out="./output" --filename_out="out_sahi" --device="cpu" --save=True```

------------------------------------------------------------------------------------------------
но вообще функция для инференса лежит в inference.py и называется inference_sahi, там же и описание аргументов
