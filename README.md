функция для инференса

inference_sahi(filepath_src, file_dir_out, filename_out, device = "cuda:0", save=False, hide_labels=True, hide_conf=True)

аргументы, которые вам нужно установить самостоятельно

filepath_src - строка содержащая путь до файла

file_dir_out - строка содержащая путь до папки, в которую положиться картинка с детекцией

filename_out - строка с названием выходного файла (наша картинка с детекцией)

device - ну или "cuda:0" (в дефолте) или 'cpu'

save - сохранять ли картинку в file_dir_out/filename_out

ну и доп аргументы

hide_labels - скрывать название класса

hide_conf - скрывать confidence

-------------------------------------------------

`git clone https://github.com/Pozovi23/for_ML_contest.git`

создаете виртуальное окружение (там кто как его создает, но у меня просто `python -m venv venv`)

и в виртуальном окружении ставите зависимости
`pip install -r requirements.txt`

потом уже используете функцию

---------------------------------------

в main.py пример
