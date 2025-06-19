функция для инференса

inference_sahi(filepath_src, file_dir_out, filename_out, device = "cuda:0", save=False, hide_labels=True, hide_conf=True)

аргументы, которые вам нужно установить самостоятельно

-------------------------------------------------

`git clone https://github.com/Pozovi23/for_ML_contest.git`

создаете виртуальное окружение (там кто как его создает, но у меня просто `python -m venv venv`)

и в виртуальном окружении ставите зависимости
`pip install -r requirements.txt`

необходимая вам функция лежит в inference.py и называется inference_sahi

в main.py представлен пример её использования
