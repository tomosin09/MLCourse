import pandas as pd
import numpy as np

# Чтение датафрейма
students_perfomance = pd.read_csv('StudentsPerformance.csv')

'''Найти количество студентов
со значением free/reduced в столбце Lunch'''


def share(dataFrame):
    return f'Доля студентов с урезанным ланчем {dataFrame.mean()}'


# print(lunch((students_perfomance['lunch'] == 'free/reduced')))

'''Как различается среднее
и дисперсия оценок по предметам у групп студентов
со стандартным или урезанным ланчем?'''
standart = (students_perfomance['lunch'] == 'standard')
# print(f'Оценки студентов с урезанным ланчем\n{students_perfomance[reduced].describe()}'
#       f'\n\nОценки студентов со стандартным ланчем\n{students_perfomance[standart].describe()}')

# Переименование колонок
students_perfomance = students_perfomance \
    .rename(columns={'parental level of education': 'parental_level_of_education',
                     'test preparation course': 'test_preparation_course',
                     'math score': 'math_score',
                     'reading score': 'reading_score',
                     'writing score': 'writing_score'})

# Изучение функции query
# print(students_perfomance.query('writing_score > 74'))
# print(students_perfomance.query('gender == "female" & writing_score > 78'))
# Фильтры по колонкам
# Вывод колонок
score_columns = [i for i in list(students_perfomance) if 'score' in i]
score_columns2 = students_perfomance.filter(like='score')

# Группировка
mean_scores = students_perfomance.groupby(['gender', 'race/ethnicity'], as_index=False) \
    .aggregate({'math_score': 'mean', 'reading_score': 'mean'}) \
    .rename(columns={'math_score': 'mean_math_score', 'reading_score': 'mean_reading_score'})
# print(mean_scores)
sort = students_perfomance.sort_values(['gender', 'math_score'], ascending=False) \
    .groupby(['gender', 'math_score']) \
    .head(10)
# print(sort.loc[:,['gender', 'math_score']])
# print(students_perfomance.sort_values(['gender', 'math_score'], ascending=False)\
#       .groupby(['gender', 'math_score']).head(5))
students_perfomance['total_score'] = students_perfomance.math_score \
                                     + students_perfomance.reading_score \
                                     + students_perfomance.writing_score
# print(students_perfomance['total_score'].head())
students_perfomance = students_perfomance.assign(total_score_log=
                                                 np.log(students_perfomance.total_score))
students_perfomance = students_perfomance.drop(['total_score', 'lunch'], axis=1)

''' Пересчитаем число ног у героев игры Dota2! Сгруппируйте героев
из датасэта по числу их ног (колонка legs),
и заполните их число в задании ниже.
'''
dota_hero = pd.read_csv('dota_hero_stats.csv')
count_legs = dota_hero.groupby(['legs'], as_index=False) \
    .aggregate({'id': 'count'}) \
    .rename(columns={'id': 'quantity'})
# print(count_legs)

'''К нам поступили данные из бухгалтерии о заработках Лупы и Пупы за разные задачи! 
Посмотрите у кого из них больше средний заработок в различных категориях (колонка Type) 
и заполните таблицу, указывая исполнителя с большим заработком в каждой из категорий.'''
accountancy = pd.read_csv('accountancy.csv')
# print(accountancy.groupby(['Executor','Type']).aggregate({'Salary':'mean'}))

''' Продолжим исследование героев Dota2. 
Сгруппируйте по колонкам attack_type и primary_attr
и выберите самый распространённый набор характеристик.
'''
# print(dota_hero.loc[:,['attack_type', 'primary_attr']])
count_attributes = dota_hero.groupby(['attack_type', 'primary_attr'], as_index=False) \
    .aggregate({'id': 'count'}).rename(columns={'id': 'quantity'})

'''
Пользуясь предыдущими данными, 
укажите через пробел (без запятых) чему равны минимальная, 
средняя и максимальная концентрации аланина (alanin) среди видов рода Fucus. 
Округлите до 2-ого знака, десятичным разделителем является точка.
'''
concentration = pd.read_csv('algae.csv')
# mean_concentrations = concentration.groupby('genus').mean()
# print(mean_concentrations)
# ПЕРЕДЕЛАТЬ
concentration_fucus = concentration.query('genus == "Fucus"').describe()[('alanin')]
concentration_group = concentration.groupby('group') \
    .aggregate({'species': 'count', 'citrate': 'var'}) \
    .rename(columns={'species': 'quantity',
                     'citrate': 'var_citrate'})
concentration_group['percentile_sucrose'] = concentration['sucrose'].max() - concentration['sucrose'].min()
print(concentration_group)