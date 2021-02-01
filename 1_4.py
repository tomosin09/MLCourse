import pandas as pd
import numpy as np

# Чтение датафрейма
students_perfomance = pd.read_csv('StudentsPerformance.csv')

# '''Найти количество студентов
# со значением free/reduced в столбце Lunch'''
# reduced = (students_perfomance['lunch'] == 'free/reduced')
# print(f'Доля студентов с урезанным ланчем {reduced.mean()}')

# '''Как различается среднее
# и дисперсия оценок по предметам у групп студентов
# со стандартным или урезанным ланчем?'''
# standart = (students_perfomance['lunch'] == 'standard')
# print(f'Оценки студентов с урезанным ланчем\n{students_perfomance[reduced].describe()}'
#       f'\n\nОценки студентов со стандартным ланчем\n{students_perfomance[standart].describe()}')

# Переименование колонок
# print(students_perfomance.columns)
# students_perfomance = students_perfomance \
#     .rename(columns={'parental level of education': 'parental_level_of_education',
#                      'test preparation course': 'test_preparation_course',
#                      'math score': 'math_score',
#                      'reading score': 'reading_score',
#                      'writing score': 'writing_score'})

# Изучение функции query
# print(students_perfomance.query('writing_score > 74'))
# print(students_perfomance.query('gender == "female" & writing_score > 78'))

# Фильтры по колонкам
# Вывод колонок
score_columns = [i for i in list(students_perfomance) if 'score' in i]
score_columns2 = students_perfomance.filter(like='score')
print(score_columns2.head())
