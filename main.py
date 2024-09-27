import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
import copy
import time
start_time = time.time()
warnings.filterwarnings('ignore')


def standardize_persian_text(text):
    return text.replace('ي', 'ی').replace('ك', 'ک').replace('?', 'ی')


def trim_query(query):
    query_split = query.split(' ')
    query_drop_space = [i for i in query_split if i]

    cleaned_data = []
    for i in range(len(query_drop_space)):
        if not (query_drop_space[i] == 'از' and i > 0 and query_drop_space[i - 1] == 'از'):
            cleaned_data.append(query_drop_space[i])

    return cleaned_data


def convert_day_to_int(query):
    map_day = {
        'شنبه': 0,
        'یکشنبه': 1,
        'دوشنبه': 2,
        'سه': 3,
        'چهارشنبه': 4,
        'پنج': 5
    }
    query_edit = []
    for i in query:
        if i in map_day:
            query_edit.append(map_day[i])
        else:
            query_edit.append(i)
    if query_edit[1] == 0:
        del query_edit[1]
    if len(query_edit) > 6:
        if query_edit[6] == 0:
            del query_edit[6]
    # p
    return query_edit


def convert_time_to_int(time_str):
    # print(time_str)
    # print(time_str)
    hour = time_str.split(':')[0]
    hour_int = int(hour)
    # print(hour_int)
    # input()
    return hour_int


def time_series(query):
    if len(query) > 8:
        index = [2, 4, 7, 9]
    else:
        index = [2, 4]
    sample = []

    for i in index:
        sample.append(convert_time_to_int(query[i]))

    return sample


def check_class(query, hours):
    # print(query)
    # print(hours)
    # input()
    if len(query) > 8:
        day = [query[0], query[5]]
    else:
        day = [query[0]]
    for i in range(0, len(hours), 2):

        if hours[i + 1] - hours[i] == 2:
            q = (hours[i] - 8) / 2
            if base_table[day[0]][int(q)] + 1 == 1:
                day.pop(0)
                continue
            else:
                return False
        else:
            q = (hours[i] - 8) / 2
            if hours[i] % 2 == 0:
                if base_table[day[0]][int(q)] != 0.25 and base_table[day[0]][int(q)] + 0.25 <= 1:
                    day.pop(0)
                    continue
                else:
                    return False
            else:
                if base_table[day[0]][int(q)] != 0.75 and base_table[day[0]][int(q)] + 0.75 <= 1:
                    day.pop(0)
                    continue
                else:
                    return False
    return True


def trim_three_hours(start_time, day):
    # print(start_time)
    if start_time % 2 == 0:
        return [day, 'از', convert_to_time_format(start_time), 'تا', convert_to_time_format(start_time+2), day, 'از', convert_to_time_format(start_time+2), 'تا', convert_to_time_format(start_time+4)]
    return [day, 'از', convert_to_time_format(start_time-1), 'تا', convert_to_time_format(start_time), day, 'از', convert_to_time_format(start_time), 'تا', convert_to_time_format(start_time+2)]


def convert_to_time_format(hour):
    return str(hour) + ":00"


def tick_base_table(query, hours):
    if len(query) > 8:
        day = [query[0], query[5]]
    else:
        day = [query[0]]

    if hours[1] - hours[0] == 3:
        query = trim_three_hours(hours[0], query[0])
    if check_class(query, hours):
        for i in range(0, len(hours), 2):
            if hours[i + 1] - hours[i] == 2:
                q = (hours[i] - 8) / 2
                base_table[day[0]][int(q)] += 1
            else:
                q = (hours[i] - 8) / 2
                if hours[i] % 2 == 0:
                    base_table[day[0]][int(q)] += 0.25
                else:
                    base_table[day[0]][int(q)] += 0.75
            day.pop(0)
        return True
    else:
        return False


def pipline(title):
    title = standardize_persian_text(title)
    # print(title)
    title_edit = trim_query(title)
    title_edit = convert_day_to_int(title_edit)
    # print(title_edit)
    title_hour = time_series(title_edit)
    if tick_base_table(title_edit, title_hour):
        return True
    return False


def selector(class_list, i=0, selected=None):
    global base_table, df_matrix, c, df_matrix_course

    if selected is None:
        selected = []

    index_class = list(class_list.keys())

    if i >= len(index_class):
        dt_temp = pd.DataFrame(base_table)
        dt_temp_course = pd.DataFrame(selected, columns=['Course', 'Schedule'])
        dt_temp['id'] = c + 1
        dt_temp_course['id'] = c + 1
        c += 1
        df_matrix = pd.concat([df_matrix, dt_temp])
        df_matrix_course = pd.concat([df_matrix_course, dt_temp_course])
        
        return

    for j in range(len(class_list[index_class[i]])):
        base_table_copy = copy.deepcopy(base_table)
        selected_copy = selected.copy()

        flag = pipline(class_list[index_class[i]][j])

        if flag:
            selected_copy.append((index_class[i], class_list[index_class[i]][j]))
            selector(class_list, i + 1, selected_copy)

        base_table = base_table_copy


def index_trim(indx):
    index_edit = []
    for i in indx:
        index_edit.append(i.replace('?', 'ی'))
    return index_edit


def fitness_function(matrx):

    val_base = [(0.25, 0.75), (0.75, 0.25)]
    fitness_scores_list = []
    for i in val_base:
        matrix_copy = copy.deepcopy(matrx)
        matrix_copy = edd_oven(matrix_copy, i[0], i[1])

        days_present = np.sum(np.any(matrix_copy > 0, axis=1))
        gaps = 0

        for row in range(matrix_copy.shape[0]):
            class_indices = np.nonzero(matrix_copy.loc[row])[0]
            if len(class_indices) > 1:
                for k in range(len(class_indices) - 1):
                    gaps += (class_indices[k + 1] - class_indices[k] - 1)
        # sum(matrix_copy.iloc[:, 4:7].sum())
        # sum_evening_class = sum(matrix_copy.iloc[:, 4:7].sum())

        fitness = days_present + gaps
        fitness_scores_list.append(fitness)
    return np.array(fitness_scores_list).mean()


def edd_oven(matrx, one_val, zero_val):
    matrx[matrx == one_val] = 1
    matrx[matrx == zero_val] = 0
    return matrx


df_matrix = pd.DataFrame()
df_matrix_course = pd.DataFrame()
c = 0
base_table = np.zeros(shape=(6, 7))

dt = pd.read_csv('Book2.csv', encoding='windows-1256')
dt['درس'] = index_trim(dt['درس'])

values_dict = defaultdict(list)
for indx, row in dt.iterrows():
    values_dict[row[0]].append(row[1])

selector(values_dict)
df_matrix.to_csv('test.csv', index=False)
df_matrix_course.to_csv('test2.csv', index=False, encoding='utf-8-sig')
point_list = []
id_table = 0
for table in range(0, df_matrix.shape[0], 6):
    id_table += 1
    point = fitness_function(df_matrix.iloc[table:table + 6, :7])
    point_list.append((id_table, point))
point_list = np.array(point_list)

minimum_id = point_list[np.argmin(point_list[:, 1])][0]
sorted_list = sorted(point_list, key=lambda x: x[1])
print(sorted_list)
print(minimum_id)
# input()
for i in range(4):
    print(sorted_list[i][0])
    minimum_table = df_matrix[df_matrix['id'] == sorted_list[i][0]]
    print(minimum_table.iloc[:, :7])
    course = df_matrix_course[df_matrix_course['id'] == sorted_list[i][0]][['Course', 'Schedule']]

    for detail in course.iloc:
        print(detail[0])
        print(detail[1])
    print()
end_time = time.time()
print(end_time - start_time)
