import json
import os

import numpy as np
import sympy
from flask import Flask, render_template, redirect, request, send_file
from sympy import Poly, solve, oo
from sympy.abc import x

from data import db_session
from data.beautiful_links import Link
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


def getDeepDotQuality(func, arg, val, n = 3):
    dy = func.diff(arg)
    dyn = dy.subs(arg, val)
    if (dyn == 0):
        return getDeepDotQuality(dy, arg, val, n+1)
    elif (n % 2 == 1):
        return 'has an inflection point'
    elif (dyn > 0):
        return 'min'
    else:
        return 'max'
    return 'aaaaaa'


def getDotQuality(func, arg, val):
    dy = func.subs(arg, val)
    if (dy > 0):
        return 'min'
    elif (dy < 0):
        return 'max'
    else:
        return getDeepDotQuality(func, arg, val)


def findExtremums(func, arg):
    dy = func.diff(arg)
    ddy = dy.diff(arg)
    extremums = solve(dy, arg)

    for val in extremums:
        yield '{} = ({}, {})'.format(getDotQuality(ddy, arg, val), val, func.subs(x, val))




@app.route('/math/task_1', methods=['GET', 'POST'])
def math_task_1():
    if request.method == 'GET':
        return render_template('second_task.html')
    elif request.method == 'POST':
        result = '$$1) D{f} = '
        x = sympy.Symbol("x")
        R = sympy.S.Reals
        f = sympy.parse_expr(request.form.get('line'))
        result += sympy.latex(sympy.calculus.util.continuous_domain(f, x, R)) + '$$\n'
        result += '$$2) K{f} = ' + sympy.latex(sympy.solve(f)) + '$$\n'
        result += "$$3) f'(x) = " + sympy.latex(sympy.diff(f)) + '$$\n'
        print(f)

        try:
            result_ = []
            result_2 = []
            some = sympy.solvers.inequalities.solve_poly_inequality(Poly(sympy.diff(f), x, domain='ZZ'),
                                                             '!=')
            some_minus = sympy.solvers.inequalities.solve_poly_inequality(
                Poly(sympy.diff(f), x, domain='ZZ'),
                '<')
            some_plus = sympy.solvers.inequalities.solve_poly_inequality(
                Poly(sympy.diff(f), x, domain='ZZ'),
                '>')
            for el in some:
                if el in some_minus:
                    result_.append('-')
                    result_2.append(r'\downarrow')
                elif el in some_plus:
                    result_.append('+')
                    result_2.append(r'\uparrow')
            print(result_)
            result += '$$' + sympy.latex(some) + '$$\n'
            result += '$$' + '\hspace{33pt}'.join(result_) + '$$\n'
            result += '$$' + '\hspace{0pt}'.join(result_2) + '$$'
            result += '$$' + '\hspace{3pt}'.join(list(findExtremums(f, x))) + '$$'
        except Exception:
            result += '$$-$$\n'
        result += r'$$ \lim_{x \to \infty} ' + sympy.latex(f) + ' = ' + sympy.latex(sympy.limit(f, x, oo)) + '$$'
        new_task = Task()
        if db_sess.query(Task).first() is None:
            new_task.id = 1
        else:
            new_task.id = db_sess.query(Task).order_by(Task.id.desc()).first().id + 1
        for el in db_sess.query(Task).filter(Task.type == 2):
            if el.info == [str(f)]:
                return redirect(f'/task/{el.id}')
        new_task.info = [str(f)]
        new_task.type = 2
        result_dict = {'result': result}
        with open(fr'static/files/{new_task.id}.json', 'w') as file:
            file.write(json.dumps(result_dict))
        new_task.solution_path = fr'static/files/{new_task.id}.json'
        graph = sympy.plot(f, show=False, addaptive=False, nb_of_points=400)
        graph.save(f'static/images/{new_task.id}.png')
        db_sess.add(new_task)
        db_sess.commit()
        return redirect(f'/task/{new_task.id}')
        # return render_template('second_task_solution.html', solution=result)

@app.route('/get_image/<int:task_id>')
def get_image(task_id):
    return send_file(f'static/images/{task_id}.png', mimetype='image/gif')


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
            if request.form.get(f'{i}R') == '':
                decide = False
            try:
                legs[i - 1].append(int(request.form.get(f'{i}R')))
            except ValueError:
                decide = False
            param[-1]['name'] = f'{i}R'
            param[-1]['content'] = str(request.form.get(f'{i}R'))
            param.append({})
            if request.form.get(f'{i}D') == 'Направление':
                decide = False
            try:
                legs[i - 1].append(request.form.get(f'{i}D'))
            except ValueError:
                decide = False
            param[-1]['name'] = f'{i}D'
            param[-1]['content'] = str(request.form.get(f'{i}D'))
            param.append({})
            if request.form.get(f'{i}V') == '':
                decide = False
            try:
                legs[i - 1].append(int(request.form.get(f'{i}V')))
            except ValueError:
                decide = False
            param[-1]['name'] = f'{i}V'
            param[-1]['content'] = str(request.form.get(f'{i}V'))
        if not decide:
            return render_template('first_task_for_edit.html',
                                   ran=list(range(1, int(request.form.get('lines')) + 1)),
                                   elems=param, lines=int(request.form.get('lines')))
        result_dict = {}
        result = ''
        MH_gen = MH_method_for_out(legs)
        for i in range(len(legs)):
            result += str(i + 1) + '.' + '\n'
            result += 'R1 = ' + str(next(MH_gen)) + '\n'
            result += 'Rэ = ' + str(next(MH_gen)) + '\n'
            result += 'I' + ' = '.join(next(MH_gen)) + ' = ' + next(MH_gen) + '\n'
            result += 'Uab = ' + str(next(MH_gen)) + '\n'
            for g in range(len(legs) - 1):
                result += 'I' + ' = '.join(next(MH_gen)) + ' = ' + next(MH_gen) + '\n'
        dict_ = next(MH_gen)
        result += '\n\nTrue currents: \n'
        for i in range(1, len(legs) + 1):
            sum_i = ''
            for index, el in enumerate(dict_[i]):
                if index == 0:
                    sum_i += str(el)
                else:
                    sum_i += (' - ' if el < 0 else ' + ') + str(abs(el))

            result += 'I{} = {} = {}'.format(str(i), sum_i,
                                             str(round(float(sum(dict_[i])), 2))) + '\n'
        result_dict['MH'] = result

        result = ''
        MYH_gen = MYH_method_for_out(legs)
        gs = next(MYH_gen)
        for index, g in enumerate(gs):
            result += f'  g{index + 1} = 1/{legs[index][0]} = {g} Cм \n'
        result += f'  Uab = {next(MYH_gen)} \n'
        for i in range(len(legs)):
            result += '  ' + next(MYH_gen) + '\n'
        result_dict['MYH'] = result
        result = 'Матрица: \n'
        MYKY_gen = MYKY_method_for_out(legs)
        for i in range(len(legs)):
            result += next(MYKY_gen) + '\n'
        result += '\nСистема: \n'
        for i in range(len(legs)):
            result += next(MYKY_gen) + '\n'
        i = next(MYKY_gen)
        result += '\n\n'
        for g in range(len(legs)):
            result += f'I{g + 1} = ' + str(i[g]) + '\n'
        result_dict['MYKY'] = result
        cols = [f'I{i}' for i in range(1, len(legs) + 1)]
        nl = '\n      '
        html = f'''
                <table class="table">
                  <thead>
                    <tr>
                      <th scope="col">Метод</th>
                      {nl.join([f'<th scope="col">{el}</th>' for el in cols])}
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th scope="row">МУН</th>
                      {nl.join([f'<td>{round(el, 4)}</td>' for el in MYH_method(legs).values()])}
                    </tr>
                    <tr>
                      <th scope="row">MH</th>
                      {nl.join([f'<td>{round(sum(el), 4)}</td>' for el in MH_method(legs).values()])}
                    </tr>
                    <tr>
                      <th scope="row">МКУ(МУКУ)</th>
                      {nl.join([f'<td>{round(el, 4)}</td>' for el in MYKY_method(legs)])}
                    </tr>
                  </tbody>
                </table>
                        '''
        result_dict['html'] = html
        new_task = Task()
        if db_sess.query(Task).first() is None:
            new_task.id = 1
        else:
            new_task.id = db_sess.query(Task).order_by(Task.id.desc()).first().id + 1
        for el in db_sess.query(Task).filter(Task.type == 1):
            if el.info == legs:
                return redirect(f'/task/{el.id}')
        new_task.info = legs
        new_task.type = 1
        with open(fr'static/files/{new_task.id}.json', 'w') as file:
            file.write(json.dumps(result_dict))
        new_task.solution_path = fr'static/files/{new_task.id}.json'
        db_sess.add(new_task)
        db_sess.commit()
        return redirect(f'/task/{new_task.id}')


@app.route('/task/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = db_sess.query(Task).filter(Task.id == task_id).first()
    if task.type == 1:
        legs = task.info
        param = []
        for i in range(len(legs)):
            param.append({})
            param[-1]['name'] = f'{i + 1}R'
            param[-1]['content'] = str(legs[i][0])
            param.append({})
            param[-1]['name'] = f'{i + 1}D'
            param[-1]['content'] = str(legs[i][1])
            param.append({})
            param[-1]['name'] = f'{i + 1}V'
            param[-1]['content'] = str(legs[i][2])

        if len(db_sess.query(Link).filter(Link.task_id == task_id).all()) == 1:
            return render_template('first_task_solution.html',
                                   ran=list(range(1, len(legs) + 1)),
                                   elems=param, lines=len(legs), task_id=task_id,
                                   link=db_sess.query(Link).filter(
                                       Link.task_id == task_id).first().link,
                                   beauti='true')

        return render_template('first_task_solution.html',
                               ran=list(range(1, len(legs) + 1)),
                               elems=param, lines=len(legs), task_id=task_id, link='',
                               beauti='false')
    elif task.type == 2:
        with open(f'static/files/{task_id}.json', 'r') as file:
            dict_ = json.loads(file.read())
        return render_template('second_task_solution.html', line=task.info[0], solution=dict_['result'], task_id=task_id)


@app.route('/')
def lol():
    return redirect('/phys/task_1')


@app.route('/get_json_task/<int:task_id>')
def get_json_task(task_id):
    with open(rf'static/files/{task_id}.json', 'r') as file:
        return file.read()


@app.route('/check_url', methods=['POST'])
def check_url():
    link = request.data.decode('utf-8')
    if len(db_sess.query(Link).filter(Link.link == link).all()) == 1:
        return 'False'
    return 'True'


@app.route('/add_url', methods=['POST'])
def add_url():
    data = request.data.decode('utf-8')
    new_link = Link()
    if db_sess.query(Link).first() is None:
        new_link.id = 1
    else:
        new_link.id = db_sess.query(Link).order_by(Link.id.desc()).first().id + 1
    new_link.link, new_link.task_id = data.split('୪')[0], int(data.split('୪')[1])
    db_sess.add(new_link)
    db_sess.commit()
    return 'ok'


@app.route('/t/<string:name>')
def beauty(name):
    task_id = db_sess.query(Link).filter(Link.link == name).first().task_id
    task = db_sess.query(Task).filter(Task.id == task_id).first()
    print(task.type)
    if task.type == 1:
        legs = task.info
        param = []
        for i in range(len(legs)):
            param.append({})
            param[-1]['name'] = f'{i + 1}R'
            param[-1]['content'] = str(legs[i][0])
            param.append({})
            param[-1]['name'] = f'{i + 1}D'
            param[-1]['content'] = str(legs[i][1])
            param.append({})
            param[-1]['name'] = f'{i + 1}V'
            param[-1]['content'] = str(legs[i][2])

        return render_template('first_task_solution.html',
                               ran=list(range(1, len(legs) + 1)),
                               elems=param, lines=len(legs), task_id=task_id, link=name,
                               beauti='true')



if __name__ == '__main__':
    db_session.global_init(r"data/db/tasks.db")
    db_sess = db_session.create_session()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=1)
