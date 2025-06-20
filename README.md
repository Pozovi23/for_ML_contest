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

для виндоус юзеров создание и активация идет вот так вроде бы
```
python3 -m venv venv
venv\Scripts\activate.bat
```
)

и в виртуальном окружении ставите зависимости
```pip install -r requirements.txt```

Пример запуска на всех фотках из папки test
```python3 main.py "test" --file_dir_out="./output" --device="cuda:0"```

