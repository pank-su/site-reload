import json

import numpy as np
from flask import Flask, render_template, url_for, redirect, request, render_template_string
from data import db_session
from data.tasks import Task


# Решение задачи MH методом и возврат значений токов
def MH_method(legs: list) -> dict:
    dict_of_i = {}
    [dict_of_i.update({el + 1: []}) for el in range(len(legs))]
    for el in range(len(legs)):
        if legs[el][1] == 'R':
            minus_or_plus = -1
        else:
            minus_or_plus = 1
        r1 = round(sum([1 / el_2[0] for index, el_2 in enumerate(legs) if el != index]) ** -1, 2)
        i = round(legs[el][2] / (r1 + legs[el][0]), 2)
        dict_of_i[el + 1].append(i * minus_or_plus)
        uab = round(r1 * i, 2)
        for index, el_3 in enumerate([el_2[0] for el_2 in legs]):
            if index != el:
                dict_of_i[index + 1].append(round(uab / el_3 * -1 * minus_or_plus, 2))
    return dict_of_i


# Тоже самое, только для красивого вывода(для вывода решения)
def MH_method_for_out(legs: list):
    dict_of_i = {}
    [dict_of_i.update({el + 1: []}) for el in range(len(legs))]
    for el in range(len(legs)):
        if legs[el][1] == 'R':
            minus_or_plus = -1
        else:
            minus_or_plus = 1
        r1 = round(sum([1 / el_2[0] for index, el_2 in enumerate(legs) if el != index]) ** -1, 2)
        if r1.is_integer():
            r1 = int(r1)
        list_of_r1 = [el_2[0] for index, el_2 in enumerate(legs) if el != index]
        str_r1 = ''
        if len(list_of_r1) > 2:
            str_r1 += '('
            str_r1 += '+'.join(list(map(lambda a: f'1/{a}', list_of_r1)))
            str_r1 += ') ^ -1'
        elif len(list_of_r1) == 2:
            one, two = list_of_r1
            str_r1 += f'{one} * {two}/({one} + {two})'
        yield f'{str_r1} = {r1}'
        i = round(legs[el][2] / (r1 + legs[el][0]), 2)
        dict_of_i[el + 1].append(i * minus_or_plus)
        yield f'{r1} + {legs[el][0]} = {r1 + legs[el][0]}'
        yield str(el + 1), str(legs[el][2]) + '/' + str(r1 + legs[el][0])
        yield str(i * minus_or_plus)
        uab = round(r1 * i, 2)
        yield f'{r1} * {i} = {uab}'
        for index, el_3 in enumerate([el_2[0] for el_2 in legs]):
            if index != el:
                dict_of_i[index + 1].append(round(uab / el_3 * -1 * minus_or_plus, 2))
                yield str(index + 1), str(uab) + '/' + str(el_3)
                yield str(round(uab / el_3 * -1 * minus_or_plus, 2))
    yield dict_of_i


# Решение задачи MУH методом и возврат значений токов
def MYH_method(legs: list) -> dict:
    dict_of_i = {}
    gs = [round(1 / el[0], 3) for el in legs]
    edss = [-el[2] if el[1] == 'R' else el[2] for el in legs]
    uab = round(sum([gs[i] * edss[i] for i in range(len(legs))]) / sum(gs), 2)
    for i in range(len(legs)):
        dict_of_i[i + 1] = (edss[i] - uab) * gs[i]
    return dict_of_i


# Тоже самое, только для красивого вывода(для вывода решения)
def MYH_method_for_out(legs: list):
    dict_of_i = {}
    gs = [round(1 / el[0], 3) for el in legs]
    yield gs
    edss = [-el[2] if el[1] == 'R' else el[2] for el in legs]
    uab = round(sum([gs[i] * edss[i] for i in range(len(legs))]) / sum(gs), 2)
    str_uab = '('
    for i in range(len(legs)):
        if i == 0:
            str_uab += f'{edss[i]} * ' + f'{gs[i]}'
        else:
            str_uab += (' + ' if edss[i] > 0 else ' - ') + f'{abs(edss[i])} * ' + f'{gs[i]}'
    str_uab += f') / ({"+".join(map(str, gs))}) = {round(sum([gs[i] * edss[i] for i in range(len(legs))]), 3)} / {sum(gs)}'
    yield f'{str_uab} = {uab}'
    for i in range(len(legs)):
        dict_of_i[i + 1] = (edss[i] - uab) * gs[i]
        if uab > 0:
            yield f'I{i + 1} = ({edss[i]} - {uab}) * {gs[i]} = {round((edss[i] - uab) * gs[i], 4)}'
        else:
            yield f'I{i + 1} = ({edss[i]} + {uab}) * {gs[i]} = {round((edss[i] - uab) * gs[i], 4)}'
    yield dict_of_i


# Это вспомогательная функция для МУКУ метода,
# чтобы получить определённый порядок для создания системы уравнения
def MYKY_help_1(num: int) -> list:
    result = []
    x = 0
    for i in range(1, num + 1):
        if i == 1:
            result.append([i])
        elif i == num:
            result[x].append(i)
        else:
            result[x].append(i)
            x += 1
            result.append([i])
    return result


# Решение задачи MУКУ методом и возврат значений токов
def MYKY_method(legs: list):
    array = MYKY_help_1(len(legs))
    b = [0]
    for g in range(len(array)):
        one = legs[array[g][0] - 1][2] * (-1 if legs[array[g][0] - 1][1] == 'L' else 1)
        second = legs[array[g][1] - 1][2] * (-1 if legs[array[g][1] - 1][1] == 'R' else 1)
        b.append(one + second)
    a = [[1] * len(legs)]
    for g in range(len(array)):
        a.append(
            [(legs[i - 1][0] if array[g].index(i) == 1 else -legs[i - 1][0]) if i in array[g] else 0
             for i in
             range(1, len(legs) + 1)])  # заполнение матрицы
    a = np.array(a)
    b = np.array(b)
    x = np.linalg.solve(a, b)
    return x


# Тоже самое, только для красивого вывода(для вывода решения)
def MYKY_method_for_out(legs: list):
    array = MYKY_help_1(len(legs))
    b = [0]
    for g in range(len(array)):
        one = legs[array[g][0] - 1][2] * (-1 if legs[array[g][0] - 1][1] == 'L' else 1)
        second = legs[array[g][1] - 1][2] * (-1 if legs[array[g][1] - 1][1] == 'R' else 1)
        b.append(one + second)
    a = [[1] * len(legs)]
    for g in range(len(array)):
        a.append(
            [(legs[i - 1][0] if array[g].index(i) == 1 else -legs[i - 1][0]) if i in array[g] else 0
             for i in
             range(1, len(legs) + 1)])  # заполнение матрицы
    for el, el_2 in zip(a, b):
        yield '[' + ' '.join(list(map(str, el))) + ']' + '  ' + '[' + str(el_2) + ']'
    for el, el_2 in zip(a, b):
        str_ = ''
        for index, el_ in enumerate(el):
            if el_ == 0:
                continue

            if str_ == '':
                if el_ == 1:
                    str_ += f'I{len(legs) - index}'
                    continue
                str_ += f'{el_} * I{len(legs) - index}'
            else:
                if el_ == 1:
                    str_ += (' - ' if el_ < 0 else ' + ') + f'I{len(legs) - index}'
                    continue
                str_ += (' - ' if el_ < 0 else ' + ') + f'{abs(el_)} * I{len(legs) - index}'
        yield f'{str_} = {el_2}'
    a = np.array(a)
    b = np.array(b)
    x = np.linalg.solve(a, b)
    yield x

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ХАХАХАХХАХАХАХА ЭТО СЕКРЕТНЫЙ КЛЮЧ'


@app.route('/phys/task_1', methods=['GET', 'POST'])
def task_1():
    if request.method == 'GET':
        return render_template('first_task.html')
    elif request.method == 'POST':
        param = []
        decide = True
        legs = []
        for i in range(1, int(request.form.get('lines')) + 1):
            param.append({})
            legs.append([])
            if request.form.get(f'{i}R') == 0:
                decide = False
            legs[i - 1].append(int(request.form.get(f'{i}R')))
            param[-1]['name'] = f'{i}R'
            param[-1]['content'] = str(request.form.get(f'{i}R'))
            param.append({})
            if request.form.get(f'{i}D') == 'Направление':
                decide = False
            legs[i - 1].append(request.form.get(f'{i}D'))
            param[-1]['name'] = f'{i}D'
            param[-1]['content'] = str(request.form.get(f'{i}D'))
            param.append({})
            if request.form.get(f'{i}V') == 0:
                decide = False
            legs[i - 1].append(int(request.form.get(f'{i}V')))
            param[-1]['name'] = f'{i}V'
            param[-1]['content'] = str(request.form.get(f'{i}V'))
        if not decide:
            return render_template('first_task_for_edit.html', ran=list(range(1, int(request.form.get('lines')) + 1)), elems=param, lines=int(request.form.get('lines')))
        print(legs)
        return json.dumps(legs)


@app.route('/phys/task/<int:task_id>', methods=['GET'])
def get_task(task_id):
    print(task_id)
    return redirect('/phys/task_1')

@app.route('/')
def lol():
    return redirect('/phys/task_1')





if __name__ == '__main__':
    db_session.global_init("/home/pankov/PycharmProjects/site-reload/data/db/tasks.db")
    app.run(port=8080, host='127.0.0.1')