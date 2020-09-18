import datetime
import socket

import pymysql
import pandas as pd
import re

import time

from multilabel_pipeline import MultiLabelPipeline
from transformers import ElectraTokenizer
from model import ElectraForMultiLabelClassification
import operator
from konlpy.tag import Okt

# 학습된 멀티라벨 모델 사용하기 위한 준비
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-finetuned-goemotions")
model = ElectraForMultiLabelClassification.from_pretrained(
    "ckpt/koelectra-base-goemotions-ckpt(TRFAABE)/checkpoint-10500")

goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
)

checkTD = {0: 'TV', 1: '로봇청소기', 2: '에어컨', 3: '공기청정기', 4: '냉장고', 5: '전구', 6: '전기'}
checkTI = {7: '전원', 8: '예약', 9: '편성표', 10: '주행', 11: '흡입', 12: '충전', 13: '설정',
           14: '온도', 15: '바람', 16: '모드', 17: '필터', 18: '공기상태', 19: '세기',
           20: '유통기한', 21: '음식', 22: '밝기', 23: '색상', 24: '날짜', 25: '기간', 26: '상태',
           27: '전원제어', 28: '예약제어', 29: '모드제어', 30: '세기제어', 31: '온도제어', 32: '청소제어',
           33: '바람제어', 34: '밝기제어', 35: '색깔제어'}

day_expression_list = ['오늘', '내일', '모레']
time_expression_list = ['오전', '오후', '아침', '저녁']

# 서버로서 기능하기 위한 준비
HOST = ''
PORT = 6942

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))

# mysqlDB 사용을 위한 준비
host_name = "127.0.0.1"
username = "root"
password = "wooseong24599!"
database_name = "smarthome2020"

db = pymysql.connect(
    host=host_name,
    port=3306,
    user=username,
    passwd=password,
    db=database_name,
    charset='utf8'
)


def run_model(text: str):
    input_texts = [text]
    checkResults, checkLabels, checkScores = goemotions(input_texts)

    checkTDresults = []
    checkTIresults = []
    TD = {'TV': 0, '로봇청소기': 0, '에어컨': 0, '공기청정기': 0, '냉장고': 0, '전구': 0, '전기': 0}
    TI = {'전원': 0, '예약': 0, '편성표': 0, '주행': 0, '흡입': 0, '충전': 0, '설정': 0,
          '온도': 0, '바람': 0, '모드': 0, '필터': 0, '공기상태': 0, '세기': 0, '유통기한': 0,
          '음식': 0, '밝기': 0, '색상': 0, '날짜': 0, '기간': 0, '상태': 0,
          '전원제어': 0, '예약제어': 0, '모드제어': 0, '세기제어': 0, '온도제어': 0,
          '청소제어': 0, '바람제어': 0, '밝기제어': 0, '색깔제어': 0}
    label = checkLabels[0]
    score = checkScores[0]
    for i in range(len(label)):
        if label[i] in TD:
            TD[label[i]] = score[i]
        if label[i] in TI:
            TI[label[i]] = score[i]
    checkTDresults.append(max(TD.items(), key=operator.itemgetter(1))[0])
    checkTIresults.append(max(TI.items(), key=operator.itemgetter(1))[0])

    out_list = [checkTDresults[0], checkTIresults[0]]

    return out_list


def Query(sentence: str, labels: list):
    okt = Okt()
    print(f'{okt.morphs(sentence)}')
    print(labels[0], labels[1])
    sentence_morphs = okt.morphs(sentence)
    day2value = {'일': 0, '월': 1, '화': 2, '수': 3, '목': 4, '금': 5, '토': 6}
    result = '실패다 이말이야'
    # 일부 해야함
    if labels[0] == 'TV':
        if labels[1] == '전원':
            # 나중에 예, 아니오 구현할 수 있으면 구현하자.
            int2kor_power = {1: '켜진 상태', 0: '꺼진 상태'}
            sql = """
            select power from tv
            """
            df = pd.read_sql(sql, db)
            power_state = int2kor_power[int(df['power'][0])]
            result = f'TV 전원은 현재 {power_state} 입니다.'

        # 해야함
        elif labels[1] == '예약':
            query_sql = 'select * from tv_guide'
            df_tv_guide = pd.read_sql(query_sql, db)
            today = datetime.datetime.now()
            hour_int = today.hour
            if '오후' in sentence or '저녁' in sentence or '밤' in sentence or '낮' in sentence:
                hour_set = f'{hour_int + 12}:00:00'
            else:
                hour_set = f'{hour_int if hour_int > 9 else f"0{hour_int}"}:00:00'

            if '오늘' in sentence_morphs:
                if '오전' in sentence_morphs or '아침' in sentence_morphs:
                    sql = f'select * from tv_reservation where day = {today.weekday()} and time < "12:00:00"'
                    df = pd.read_sql(sql, db)
                    data = ''
                    if len(df) > 0:
                        for time_data, title in zip(df['time'], df['title']):
                            time_ = str(time_data).split()[-1][:-3]
                            data += f'\t{time_},{title}'
                    if data != '':
                        result = '오늘 오전 TV에 예약된 프로그램들은 다음과 같습니다.' + data
                    else:
                        result = '오늘 오전 TV에 예약된 프로그램은 없습니다.'
                elif '오후' in sentence_morphs or '저녁' in sentence_morphs:
                    sql = f'select * from tv_reservation where day = {today.weekday()} and time >= "12:00:00"'
                    df = pd.read_sql(sql, db)
                    data = ''
                    if len(df) > 0:
                        for time_data, title in zip(df['time'], df['title']):
                            time_ = str(time_data).split()[-1][:-3]
                            data += f'\t{time_},{title}'
                    if data != '':
                        result = '오늘 오후 TV에 예약된 프로그램들은 다음과 같습니다.' + data
                    else:
                        result = '오늘 오후 TV에 예약된 프로그램은 없습니다.'

                else:
                    sql = f'select * from tv_reservation where day = {today.weekday()}'
                    df = pd.read_sql(sql, db)
                    data = ''
                    if len(df) > 0:
                        for time_data, title in zip(df['time'], df['title']):
                            time_ = str(time_data).split()[-1][:-3]
                            data += f'\t{time_},{title}'
                    if data != '':
                        result = '오늘 TV에 예약된 프로그램들은 다음과 같습니다.' + data
                    else:
                        result = '오늘 TV에 예약된 프로그램은 없습니다.'

            elif '모레' in sentence_morphs:
                days_after_tomorrow = today + datetime.timedelta(days=2)
                if '오전' in sentence_morphs or '아침' in sentence_morphs:
                    sql = f'select * from tv_reservation where day = {days_after_tomorrow.weekday()} and time < "12:00:00"'
                    df = pd.read_sql(sql, db)
                    data = ''
                    if len(df) > 0:
                        for time_data, title in zip(df['time'], df['title']):
                            time_ = str(time_data).split()[-1][:-3]
                            data += f'\t{time_},{title}'
                    if data != '':
                        result = '모레 오전 TV에 예약된 프로그램들은 다음과 같습니다.' + data
                    else:
                        result = '모레 오전 TV에 예약된 프로그램은 없습니다.'
                elif '오후' in sentence_morphs or '저녁' in sentence_morphs:
                    sql = f'select * from tv_reservation where day = {days_after_tomorrow.weekday()}'
                    df = pd.read_sql(sql, db)
                    data = ''
                    if len(df) > 0:
                        for time_data, title in zip(df['time'], df['title']):
                            time_ = str(time_data).split()[-1][:-3]
                            data += f'\t{time_},{title}'
                    if data != '':
                        result = '모레 오후 TV에 예약된 프로그램들은 다음과 같습니다.' + data
                    else:
                        result = '모레 오후 TV에 예약된 프로그램은 없습니다.'
                else:
                    sql = f'select * from tv_reservation where day = {days_after_tomorrow.weekday()}'
                    df = pd.read_sql(sql, db)
                    data = ''
                    if len(df) > 0:
                        for time_data, title in zip(df['time'], df['title']):
                            time_ = str(time_data).split()[-1][:-3]
                            data += f'\t{time_},{title}'
                    if data != '':
                        result = '모레 TV에 예약된 프로그램들은 다음과 같습니다.' + data
                    else:
                        result = '모레 TV에 예약된 프로그램은 없습니다.'

            elif '내일' in sentence_morphs:
                tomorrow = today + datetime.timedelta(days=1)
                if '오전' in sentence_morphs or '아침' in sentence_morphs:
                    sql = f'select * from tv_reservation where day = {tomorrow.weekday()} and time < "12:00:00"'
                    df = pd.read_sql(sql, db)
                    data = ''
                    if len(df) > 0:
                        for time_data, title in zip(df['time'], df['title']):
                            time_ = str(time_data).split()[-1][:-3]
                            data += f'\t{time_},{title}'
                    if data != '':
                        result = '내일 오전 TV에 예약된 프로그램들은 다음과 같습니다.' + data
                    else:
                        result = '내일 오전 TV에 예약된 프로그램은 없습니다.'
                elif '오후' in sentence_morphs or '저녁' in sentence_morphs:
                    sql = f'select * from tv_reservation where day = {tomorrow.weekday()}'
                    df = pd.read_sql(sql, db)
                    data = ''
                    if len(df) > 0:
                        for time_data, title in zip(df['time'], df['title']):
                            time_ = str(time_data).split()[-1][:-3]
                            data += f'\t{time_},{title}'
                    if data != '':
                        result = '내일 오후 TV에 예약된 프로그램들은 다음과 같습니다.' + data
                    else:
                        result = '내일 오후 TV에 예약된 프로그램은 없습니다.'
                else:
                    sql = f'select * from tv_reservation where day = {tomorrow.weekday()}'
                    df = pd.read_sql(sql, db)
                    data = ''
                    if len(df) > 0:
                        for time_data, title in zip(df['time'], df['title']):
                            time_ = str(time_data).split()[-1][:-3]
                            data += f'\t{time_},{title}'
                    if data != '':
                        result = '내일 TV에 예약된 프로그램들은 다음과 같습니다.' + data
                    else:
                        result = '내일 TV에 예약된 프로그램은 없습니다.'
            elif '요일' in sentence:
                p = re.compile('[월화수목금토일]요일')
                m = p.match(sentence)
                if m:
                    day2int = {f'{day}요일': i for i, day in enumerate(['월', '화', '수', '목', '금', '토', '일'])}
                    result = f'TV {m.group()} 예약 문장입니다.'
                    if '오전' in sentence_morphs or '아침' in sentence_morphs:
                        sql = f'select * from tv_reservation where day = {day2int[m.group()]} and time < "12:00:00"'
                        df = pd.read_sql(sql, db)
                        data = ''
                        if len(df) > 0:
                            for time_data, title in zip(df['time'], df['title']):
                                time_ = str(time_data).split()[-1][:-3]
                                data += f'\t{time_},{title}'
                        if data != '':
                            result = f'{m.group()} 오전 TV에 예약된 프로그램들은 다음과 같습니다.' + data
                        else:
                            result = f'{m.group()} 오전 TV에 예약된 프로그램은 없습니다.'
                    elif '오후' in sentence_morphs or '저녁' in sentence_morphs:
                        sql = f'select * from tv_reservation where day = {day2int[m.group()]} and time >= "12:00:00"'
                        df = pd.read_sql(sql, db)
                        data = ''
                        if len(df) > 0:
                            for time_data, title in zip(df['time'], df['title']):
                                time_ = str(time_data).split()[-1][:-3]
                                data += f'\t{time_},{title}'
                        if data != '':
                            result = f'{m.group()} 오후 TV에 예약된 프로그램들은 다음과 같습니다.' + data
                        else:
                            result = f'{m.group()} 오후 TV에 예약된 프로그램은 없습니다.'

                    else:
                        sql = f'select * from tv_reservation where day = {day2int[m.group()]}'
                        df = pd.read_sql(sql, db)
                        data = ''
                        if len(df) > 0:
                            for time_data, title in zip(df['time'], df['title']):
                                time_ = str(time_data).split()[-1][:-3]
                                data += f'\t{time_},{title}'
                        if data != '':
                            result = f'{m.group()} TV에 예약된 프로그램들은 다음과 같습니다.' + data
                        else:
                            result = f'{m.group()} TV에 예약된 프로그램은 없습니다.'
                else:
                    result = '정확한 발음으로 말해주길 바랍니다.'
            else:
                # 제목으로 검색
                title_list = get_title_set(df_tv_guide)

                target_title = ''.join(
                    re.compile('(이번 주)*[에의]?[ ]?[하는]{0,2}[ ]?(.*) 예약').search(sentence).group(2).split())
                result_title = ''
                # 유사도가 가장 높은 대상을 돌려준다.
                for title in title_list:
                    title_join = ''.join(title.split())
                    if target_title == title_join:
                        result_title = title.strip()
                        break
                    elif target_title in title_join:
                        result_title = title.strip()
                        break

                if result_title == '':
                    result = '찾고자하는 프로그램이 편성표상에 없습니다.'
                else:
                    insert_title = ''
                    for title in df_tv_guide['title']:
                        if result_title in title:
                            insert_title = title
                            break

                    if insert_title != '':
                        query_sql = f'select * from tv_reservation where title = "{insert_title}"'
                        df_reserved = pd.read_sql(query_sql, db)
                        if len(df_reserved) > 0:
                            result = f'이번주에 하는 {result_title} 프로그램은 예약되었습니다.'
                        else:
                            result = f'이번주에 하는 {result_title} 프로그램은 예약되지 않았습니다.'

        elif labels[1] == '편성표':
            # 나중에 어떻게 편성표를 넘겨줄지 고민을 해보자.
            date_list = ['월', '화', '수', '목', '금', '토', '일']
            if '오늘' in sentence_morphs:
                time_sentence = '오늘'
                date = date_list[time.localtime().tm_wday]
                if '오전' in sentence_morphs or '아침' in sentence_morphs:
                    time_sentence += ' 오전의'
                    sql = f"""
                    select time, title from tv_guide where day = {time.localtime().tm_wday}
                    and time < "12:00:00"
                    """
                elif '오후' in sentence_morphs or '저녁' in sentence_morphs:
                    time_sentence += ' 오후의'
                    sql = f"""
                    select time, title from tv_guide where day = {time.localtime().tm_wday}
                    and time >= "12:00:00"
                    """
                else:
                    time_sentence += '의'
                    sql = f"""
                    select time, title from tv_guide where day = {time.localtime().tm_wday}
                    """
            elif '모레' in sentence_morphs:
                time_sentence = '모레'
                after_tomorrow = datetime.datetime.now() + datetime.timedelta(days=2)
                if '오전' in sentence_morphs or '아침' in sentence_morphs:
                    time_sentence += ' 오전의'
                    sql = f"""
                    select time, title from tv_guide where day = {after_tomorrow.weekday()}
                    and time < "12:00:00"
                    """
                elif '오후' in sentence_morphs or '저녁' in sentence_morphs:
                    time_sentence += ' 오후의'
                    sql = f"""
                    select time, title from tv_guide where day = {after_tomorrow.weekday()}
                    and time >= "12:00:00"
                    """
                else:
                    time_sentence += '의'
                    sql = f"""
                    select time, title from tv_guide where day = {after_tomorrow.weekday()}
                    """
            elif '내일' in sentence_morphs:
                time_sentence = '내일'
                tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
                if '오전' in sentence_morphs or '아침' in sentence_morphs:
                    time_sentence += ' 오전의'
                    sql = f"""
                    select time, title from tv_guide where day = {tomorrow.weekday()}
                    and time < "12:00:00"
                    """
                elif '오후' in sentence_morphs or '저녁' in sentence_morphs:
                    time_sentence += ' 오후의'
                    sql = f"""
                    select time, title from tv_guide where day = {tomorrow.weekday()}
                    and time >= "12:00:00"
                    """
                else:
                    time_sentence += '의'
                    sql = f"""
                    select time, title from tv_guide where day = {tomorrow.weekday()}
                    """
            else:
                p = re.compile('[월화수목금토일]요일')
                m = p.search(sentence)
                if m:
                    date2int = {date: i for i, date in enumerate(['월', '화', '수', '목', '금', '토', '일'])}
                    p = re.compile('[월화수목금토일]')
                    m = p.search(sentence)
                    time_sentence = m.group() + '요일'

                    sql = f"""
                    select time, title from tv_guide where day = {date2int[time_sentence]}
                    """
            df = pd.read_sql(sql, db)
            data = ''
            for t, title in zip(df['time'], df['title']):
                data += f'\t{str(t).split()[-1][:-3]},{title}'
            print(data)

            # 안드로이드에 표 만드는 기능 추가해야함.
            result = f'{time_sentence} TV 편성표는 다음과 같습니다.' + data

        elif labels[1] == '전원제어':
            int2kor_power = {1: '켜진 상태', 0: '꺼진 상태'}
            if '켜' in sentence or '킬' in sentence:
                result = 'TV 전원을 켰습니다.'
                sql = 'update tv set power = 1'
            else:
                result = 'TV 전원을 껏습니다.'
                sql = 'update tv set power = 0'

            with db.cursor() as curs:
                curs.execute(sql)
                db.commit()

        elif labels[1] == '예약제어':
            query_sql = 'select * from tv_guide'
            df_tv_guide = pd.read_sql(query_sql, db)
            title_list = get_title_set(df_tv_guide)
            regex_hour = re.compile('[0-9]+시')
            regex_day = re.compile('[월화수목금토일]요일')
            regex_date = re.compile('[0-9]+일')
            hour_match = regex_hour.search(sentence)
            day_match = regex_day.search(sentence)
            date_match = regex_date.search(sentence)
            day2int = {f'{day}요일': i for i, day in enumerate(['월', '화', '수', '목', '금', '토', '일'])}

            # 6시 내고향이라는 예외적인 프로그렘을 거르기 위함
            if hour_match and (day_match or date_match):
                hour = hour_match.group()
                hour_int = int(re.compile('[0-9]+').search(hour).group())
                if '오후' in sentence or '저녁' in sentence or '밤' in sentence or '낮' in sentence:
                    hour_set = f'{hour_int + 12}:00:00'
                else:
                    hour_set = f'{hour_int if hour_int > 9 else f"0{hour_int}"}:00:00'

                if day_match:
                    day = day_match.group()
                    day_int = day2int[day]
                    query_sql = f'select * from tv_guide where day = {day_int} and time = "{hour_set}"'
                    df = pd.read_sql(query_sql, db)
                    if len(df) > 0:
                        sql = 'insert into tv_reservation(day, time, title, genre) ' \
                                f'values({df["day"][0]}, "{str(df["time"][0]).split()[-1][:-3]}", "{df["title"][0]}", "{df["genre"][0]}")'
                        with db.cursor() as curs:
                            curs.execute(sql)
                        db.commit()
                        result = f'{day} {hour}에 하는 {df["title"][0]} 방송을 예약했습니다.'

                    else:
                        result = '해당 시간에 하는 방송은 없습니다.'
                elif date_match:
                    # 날짜를 요일로 변환.
                    today = datetime.datetime.now()
                    target_day = int(re.compile('[0-9]+').search(date_match.group()).group())

                    day_int = datetime.datetime(year=today.year, month=today.month, day=target_day).weekday()

                    query_sql = f'select * from tv_guide where day = {day_int} and time = "{hour_set}"'
                    df = pd.read_sql(query_sql, db)
                    if len(df) > 0:
                        sql = 'insert into tv_reservation(day, time, title, genre) ' \
                              f'values({df["day"][0]}, "{str(df["time"][0]).split()[-1][:-3]}", "{df["title"][0]}", "{df["genre"][0]}")'
                        with db.cursor() as curs:
                            curs.execute(sql)
                        db.commit()
                        result = f'{date_match.group()} {hour}에 하는 {df["title"][0]} 방송을 예약했습니다.'

                    else:
                        result = '해당 시간에 하는 방송은 없습니다.'
            else:
                # 제목으로 제어
                target_title = ''.join(re.compile('(이번 주)*[에의]?[ ]?[하는]{0,2}[ ]?(.*) 예약').search(sentence).group(2).split())
                result_title = ''
                # 유사도가 가장 높은 대상을 돌려준다.
                for title in title_list:
                    title_join = ''.join(title.split())
                    if target_title == title_join:
                        result_title = title.strip()
                        break
                    elif target_title in title_join:
                        result_title = title.strip()
                        break

                if result_title == '':
                    result = '찾고자하는 프로그램이 편성표상에 없습니다.'
                else:
                    insert_title = ''
                    for title in df_tv_guide['title']:
                        if result_title in title:
                            insert_title = title
                            break

                    if insert_title != '':
                        query_sql = f'select * from tv_reservation where title = "{insert_title}"'
                        df_reserved = pd.read_sql(query_sql, db)
                        if len(df_reserved) < 1:
                            df_target_program = df_tv_guide[df_tv_guide['title'] == insert_title]
                            day = list(df_target_program['day'])[0]
                            time_ = str(list(df_target_program['time'])[0]).split()[-1]
                            genre = list(df_target_program['genre'])[0]

                            sql = f'insert into tv_reservation(day, time, title, genre) ' \
                                  f'values ({day}, "{time_}", "{insert_title}", "{genre}")'

                            with db.cursor() as curs:
                                curs.execute(sql)
                            db.commit()

                            result = f'이번주에 하는 {result_title} 프로그램을 예약했습니다.'
                        else:
                            result = f'이번주에 하는 {result_title} 프로그램은 이미 예약되었습니다.'

        else:
            result = '다른 표현으로 말해주길 바랍니다.'

    elif labels[0] == '로봇청소기':
        if labels[1] == '상태':
            int2kor_state = {1: '청소', 0: '충전'}
            sql = 'select state from robot'
            df = pd.read_sql(sql, db)
            state_int = df['state'][0]

            result = f'현재 로봇청소기의 상태는 {int2kor_state[state_int]} 중 입니다.'

        elif labels[1] == '예약':
            sql = 'select reservation from robot'
            df = pd.read_sql(sql, db)
            reserved_time_list = df['reservation'][0].split(':')

            result = f'로봇청소기의 청소가 예약된 시간은 {reserved_time_list[0]}시 {reserved_time_list[1]}분 입니다.'

        elif labels[1] == '주행':
            sql = 'select movement_mode from robot'
            df = pd.read_sql(sql, db)
            check = df['movement_mode'][0]

            result = f'로봇청소기의 주행모드는 {check} 입니다.'

        elif labels[1] == '흡입':
            sql = 'select suction_mode from robot'
            df = pd.read_sql(sql, db)
            suction_mode = df['suction_mode'][0]

            result = f'로봇청소기의 흡입모드는 {suction_mode} 입니다.'

        elif labels[1] == '모드':
            sql = 'select movement_mode, suction_mode from robot'
            df = pd.read_sql(sql, db)
            check = df['movement_mode'][0]
            suction_mode = df['suction_mode'][0]

            result = f'로봇청소기의 주행모드는 {check}이고 흡입모드는 {suction_mode} 입니다.'

        elif labels[1] == '설정':
            sql = 'select reservation, movement_mode, suction_mode from robot'
            df = pd.read_sql(sql, db)
            reserved_time = df['reservation'][0][-3]
            movement_mode = df['movement_mode'][0]
            suction_mode = df['suction_mode'][0]

            data = f'\t예약시간,{reserved_time}' + f'\t이동모드,{movement_mode}' + f'\f흡입모드,{suction_mode}'
            result = '로봇청소기 설정은 다음과 같습니다.' + data

        elif labels[1] == '충전':
            sql = 'select battery from robot'
            df = pd.read_sql(sql, db)
            battery = df['battery'][0]

            result = f'로봇청소기 충전률은 {battery}% 입니다.'

        elif labels[1] == '예약제어':
            regex_hour = re.compile('[0-9]+시[간]{0,1}[ ]*[반]{0,1}')
            regex_minute = re.compile('[0-9]+분')
            regex_number = re.compile('[0-9]+')
            hour_match = regex_hour.search(sentence)
            minute_match = regex_minute.search(sentence)
            time_set = ''

            int2str_time = {i: str_i for i, str_i in
                            zip(range(1, 60), ['0' + str(i) if i < 10 else str(i) for i in range(1, 60)])}
            if hour_match and minute_match:
                # (오전, 오후) 1시 30분으로 바꿔줘, 1시간 30분 뒤, 앞로 바꿔줘
                hour = regex_number.search(hour_match.group()).group()
                minute = regex_number.search(minute_match.group()).group()

                if '오전' in sentence:
                    time_set = time.strptime(f'{int2str_time[int(hour)]}:{int2str_time[int(minute)]}', '%H:%M')
                    time_set = time.strftime('%H:%M', time_set)

                elif '오후' in sentence:
                    time_set = time.strptime(f'{int(hour) + 12 % 24}:{int2str_time[int(minute)]}', '%H:%M')
                    time_set = time.strftime('%H:%M', time_set)

                elif '앞' in sentence or '전' in sentence:
                    # 현재 시간 구하기, 기준 1
                    now = datetime.datetime.now()
                    # 예약 시간이 기준인지 확인, 기준 2, 이거 정규식 고민해서 수정해야함 나중에
                    regex_reserved = re.compile('[이기예과지][전존거난]')
                    regex_reserved_match = regex_reserved.search(sentence)
                    if regex_reserved_match:
                        sql = 'select reservation from robot'
                        df = pd.read_sql(sql, db)
                        reserved_time = str(df['reservation'][0]).split()[-1][:-3]
                        time_delta = datetime.datetime.strptime(reserved_time, '%H:%M') - datetime.datetime.strptime(
                            f'{int2str_time[int(hour)]}:{int(minute)}', '%H:%M')
                        # datetime.timedelta 에서 string으로 전환하는 법
                        time_set = str(time_delta)[:-3]

                    else:
                        time_delta = datetime.datetime.strptime(f'{now.hour}:{now.minute}', '%H:%M') \
                                     - datetime.datetime.strptime(
                            f'{int2str_time[int(hour)]}:{int2str_time[int(minute)]}', '%H:%M')
                        time_set = str(time_delta)[:-3]

                    print(time_set)

                elif '뒤' in sentence or '후' in sentence:
                    # 현재 시간 구하기, 기준 1
                    now = datetime.datetime.now()
                    # 예약 시간이 기준인지 확인, 기준 2, 이거 정규식 고민해서 수정해야함 나중에
                    regex_reserved = re.compile('[이기예과지][전존거난]')
                    regex_reserved_match = regex_reserved.search(sentence)
                    if regex_reserved_match:
                        sql = 'select reservation from robot'
                        df = pd.read_sql(sql, db)
                        reserved_time = str(df['reservation'][0]).split()[-1][:-3]
                        time_delta = datetime.datetime.strptime(reserved_time, '%H:%M') \
                                     + datetime.timedelta(hours=int(hour), minutes=int(minute))
                        # datetime.timedelta 에서 string으로 전환하는 법
                        time_set = time_delta.strftime('%H:%M')

                    else:
                        time_delta = datetime.datetime.strptime(f'{now.hour}:{now.minute}', '%H:%M') \
                                     + datetime.timedelta(hours=int(hour), minutes=int(minute))
                        time_set = time_delta.strftime('%H:%M')
                else:
                    time_set = time.strptime(f'{int2str_time[int(hour)]}:{int2str_time[int(minute)]}', '%H:%M')
                    time_set = time.strftime('%H:%M', time_set)

            elif hour_match:
                hour = regex_number.search(hour_match.group()).group()
                minute = '30' if '반' in hour_match.group() else '00'

                if '오전' in sentence:
                    time_set = time.strptime(f'{int2str_time[int(hour)]}:{minute}', '%H:%M')
                    time_set = time.strftime('%H:%M', time_set)

                elif '오후' in sentence:
                    time_set = time.strptime(f'{int(hour) + 12 % 24}:{minute}', '%H:%M')
                    time_set = time.strftime('%H:%M', time_set)

                elif '앞' in sentence or '전' in sentence:
                    # 현재 시간 구하기, 기준 1
                    now = datetime.datetime.now()
                    # 예약 시간이 기준인지 확인, 기준 2, 이거 정규식 고민해서 수정해야함 나중에
                    regex_reserved = re.compile('[이기예과지][전존거난]')
                    regex_reserved_match = regex_reserved.search(sentence)
                    if regex_reserved_match:
                        sql = 'select reservation from robot'
                        df = pd.read_sql(sql, db)
                        reserved_time = str(df['reservation'][0]).split()[-1][:-3]
                        time_delta = datetime.datetime.strptime(reserved_time,
                                                                '%H:%M') - datetime.datetime.strptime(
                            f'{int2str_time[int(hour)]}:{minute}', '%H:%M')
                        # datetime.timedelta 에서 string으로 전환하는 법
                        time_set = str(time_delta)[:-3]

                    else:
                        time_delta = datetime.datetime.strptime(f'{now.hour}:{now.minute}', '%H:%M') \
                                     - datetime.datetime.strptime(f'{int2str_time[int(hour)]}:{minute}', '%H:%M')
                        time_set = str(time_delta)[:-3]

                    print(time_set)

                elif '뒤' in sentence or '후' in sentence:
                    # 현재 시간 구하기, 기준 1
                    now = datetime.datetime.now()
                    # 예약 시간이 기준인지 확인, 기준 2, 이거 정규식 고민해서 수정해야함 나중에
                    regex_reserved = re.compile('[이기예과지][전존거난]')
                    regex_reserved_match = regex_reserved.search(sentence)
                    if regex_reserved_match:
                        sql = 'select reservation from robot'
                        df = pd.read_sql(sql, db)
                        reserved_time = str(df['reservation'][0]).split()[-1][:-3]
                        time_delta = datetime.datetime.strptime(reserved_time, '%H:%M') \
                                     + datetime.timedelta(hours=int(hour), minutes=int(minute))
                        # datetime.timedelta 에서 string으로 전환하는 법
                        time_set = time_delta.strftime('%H:%M')

                    else:
                        time_delta = datetime.datetime.strptime(f'{now.hour}:{now.minute}', '%H:%M') \
                                     + datetime.timedelta(hours=int(hour), minutes=int(minute))
                        # 점검대상
                        time_set = time_delta.strftime('%H:%M')

                    print(time_set)
                else:
                    time_set = time.strptime(f'{int2str_time[int(hour)]}:{minute}', '%H:%M')
                    time_set = time.strftime('%H:%M', time_set)

            elif minute_match:
                minute = int(regex_number.search(minute_match.group()).group())
                # ???분 후로, 전으로 지금 시간에서 30분 후로로
                if '앞' in sentence or '전' in sentence:
                    # 현재 시간 구하기, 기준 1
                    now = datetime.datetime.now()
                    # 예약 시간이 기준인지 확인, 기준 2, 이거 정규식 고민해서 수정해야함 나중에
                    regex_reserved = re.compile('[이기예과지][전존거난]')
                    regex_reserved_match = regex_reserved.search(sentence)
                    if regex_reserved_match:
                        sql = 'select reservation from robot'
                        df = pd.read_sql(sql, db)
                        reserved_time = str(df['reservation'][0]).split()[-1][:-3]
                        time_delta = datetime.datetime.strptime(reserved_time,
                                                                '%H:%M') - datetime.datetime.strptime(
                            f'00:{minute}', '%H:%M')
                        # datetime.timedelta 에서 string으로 전환하는 법
                        time_set = str(time_delta)[:-3]

                    else:
                        time_delta = datetime.datetime.strptime(f'{now.hour}:{now.minute}', '%H:%M') \
                                     - datetime.datetime.strptime(f'00:{minute}', '%H:%M')
                        time_set = str(time_delta)[:-3]

                    print(time_set)

                elif '뒤' in sentence or '후' in sentence:
                    # 현재 시간 구하기, 기준 1
                    now = datetime.datetime.now()
                    # 예약 시간이 기준인지 확인, 기준 2, 이거 정규식 고민해서 수정해야함 나중에
                    regex_reserved = re.compile('[이기예과지][전존거난]')
                    regex_reserved_match = regex_reserved.search(sentence)
                    if regex_reserved_match:
                        sql = 'select reservation from robot'
                        df = pd.read_sql(sql, db)
                        reserved_time = str(df['reservation'][0]).split()[-1][:-3]
                        time_delta = datetime.datetime.strptime(reserved_time, '%H:%M') \
                                     + datetime.timedelta(minutes=int(minute))
                        # datetime.timedelta 에서 string으로 전환하는 법
                        time_set = time_delta.strftime('%H:%M')

                    else:
                        time_delta = datetime.datetime.strptime(f'{now.hour}:{now.minute}', '%H:%M') \
                                     + datetime.timedelta(minutes=int(minute))
                        time_set = time_delta.strftime('%H:%M')

                    print(time_set)

            sql = f'update robot set reservation = "{time_set}:00"'
            with db.cursor() as curs:
                curs.execute(sql)
            db.commit()

            result = f'로봇청소기의 변경된 예약시간은 {time_set} 입니다.'

        elif labels[1] == '모드제어':
            movement_mode_list = ['지그재그', '꼼꼼', '집중', '반복', '영역']
            suction_mode_list = ['강', '중', '약', '터보']
            target_location_list = ['거실', '안방', '화장실', '작은 방', '큰 방', '주방', '다목적 실', '다목적실', '베란다']
            check2move = {check: mode for check, mode
                          in zip(movement_mode_list, ['지그재그', '꼼꼼청소', '집중청소', '반복청소', '지정영역청소'])}

            movement = ''
            suction = ''
            location = ''
            for check in movement_mode_list:
                if check in sentence:
                    movement = check2move[check]
                    break

            for suction_mode in suction_mode_list:
                if suction_mode in sentence:
                    suction = suction_mode
                    break

            if movement == '' or movement == '지정영역청소':
                for target_location in target_location_list:
                    if target_location in sentence:
                        movement = '영역'
                        location = target_location
                        break

            if movement != '':
                sql = f'update robot set movement_mode = "{movement}"'
                with db.cursor() as curs:
                    curs.execute(sql)
                db.commit()

            if suction != '':
                sql = f'update robot set suction_mode = "{suction}"'
                with db.cursor() as curs:
                    curs.execute(sql)
                db.commit()

            if location != '':
                sql = f'update robot set location = "{location}"'
                with db.cursor() as curs:
                    curs.execute(sql)
                db.commit()

            if movement != '' and suction != '':
                if location != '':
                    result = f'로봇청소기의 지정영역청소 위치와 흡입모드는 {location}, {suction}' \
                             f'{"으로" if suction != "터보" else "로"} 수정했습니다.'
                else:
                    result = f'로봇청소기의 흡입모드와 이동모드는 {suction}, {movement}로 수정했습니다.'

            elif movement != '':
                result = f'로봇청소기의 이동모드는 {movement}로 수정했습니다.'
            else:
                result = f'로봇청소기의 흡입모드는 {suction}{"으로" if suction != "터보" else "로"} 수정했습니다.'

        elif labels[1] == '청소제어':
            if '청소' in sentence or '깨끗' in sentence:
                sql = 'update robot set state = 1'
                result = '로봇청소기 청소를 시작합니다.'
            else:
                sql = 'update robot set state = 0'
                result = '로봇청소기 충전을 시작합니다.'

            with db.cursor() as curs:
                curs.execute(sql)
            db.commit()

        else:
            result = '다른 표현으로 말해주길 바랍니다.'

    elif labels[0] == '에어컨':
        if labels[1] == '전원':
            int2kor_power = {1: '켜진', 0: '꺼진'}
            sql = 'select power from air_conditioner'
            df = pd.read_sql(sql, db)
            power = int2kor_power[df['power'][0]]
            result = f'에어컨 전원은 현재 {power} 상태 입니다.'

        elif labels[1] == '예약':
            sql = 'select str_time, duration from air_conditioner'
            df = pd.read_sql(sql, db)
            str_time = str(df['str_time'][0]).split()[-1][:-3]
            duration = get_air_conditioner_reservation_sentence_part(df['duration'][0])
            result = f'에어컨 예약시간은 {str_time}이고 동작지속시간은 {duration} 입니다.'

        elif labels[1] == '온도':
            sql = 'select temperature from air_conditioner'
            df = pd.read_sql(sql, db)
            temperature = df['temperature'][0]
            result = f'에어컨 온도는 {temperature}도 입니다.'

        elif labels[1] == '바람':
            int2kor_wind_intensity = {3: '강풍', 2: '중풍', 1: '약풍', 0: '무풍'}
            sql = 'select wind_intensity, wind_direction from air_conditioner'
            df = pd.read_sql(sql, db)
            wind_intensity = int2kor_wind_intensity[df['wind_intensity'][0]]
            wind_direction = df['wind_direction'][0]
            result = f'에어컨 바람 세기는 현재 {wind_intensity}이고 방향은 {wind_direction} 입니다.'

        elif labels[1] == '모드':
            sql = 'select mode from air_conditioner'
            df = pd.read_sql(sql, db)
            mode = df['mode'][0]
            result = f'에어컨 작동 모드는 현재 {mode}입니다.'

        elif labels[1] == '설정':
            int2kor_power = {1: '켜진', 0: '꺼진'}
            int2kor_wind_intensity = {3: '강풍', 2: '중풍', 1: '약풍', 0: '무풍'}
            int2kor_filter = {1: "깨끗한", 0: "더러운"}

            sql = 'select * from air_conditioner'
            df = pd.read_sql(sql, db)

            power = int2kor_power[df['power'][0]]
            str_time = str(df['str_time'][0]).split()[-1][:-3]
            duration = get_air_conditioner_reservation_sentence_part(df['duration'][0])
            temperature = df['temperature'][0]
            wind_intensity = int2kor_wind_intensity[df['wind_intensity'][0]]
            wind_direction = df['wind_direction']
            mode = df['mode'][0]
            air_filter = int2kor_filter[df['filter'][0]]

            data = f'\t전원,{power}' + f'\t예약시간,{str_time}' + f'\t지속시간,{duration}' \
                   + f'\t온도,{temperature}' + f'\t바람세기,{wind_intensity}' + f'\t바람방향,{wind_direction}' \
                   + f'\t작동모드,{mode}' + f'\t필터,{air_filter}'
            result = '에어컨 설정은 다음과 같습니다.' + data

        elif labels[1] == '필터':
            int2kor_filter = {1: "깨끗한", 0: "더러운"}

            sql = 'select filter from air_conditioner'
            df = pd.read_sql(sql, db)
            air_filter = int2kor_filter[df['filter'][0]]

            result = f'에어컨 필터상태는 현재 {air_filter} 상태입니다.'

        elif labels[1] == '전원제어':
            regex_power = re.compile('[켜키킬]')
            power_match = regex_power.search(sentence)

            sql = ''
            query_sql = 'select power from air_conditioner'

            df = pd.read_sql(query_sql, db)
            power = df['power'][0]

            if power_match:
                if power == 1:
                    result = '에어컨은 이미 켜져있습니다.'
                else:
                    sql = 'update air_conditioner set power = 1'
                    result = '에어컨을 켰습니다.'
            else:
                if power == 0:
                    result = '에어컨은 이미 꺼져있습니다.'
                else:
                    sql = 'update air_conditioner set power = 0'
                    result = '에어컨을 껐습니다.'

            if sql != '':
                with db.cursor() as curs:
                    curs.execute(sql)
                db.commit()

        elif labels[1] == '예약제어':
            regex_hour = re.compile('[0-9]+시[간]{0,1}[ ]*[반]{0,1}')
            regex_minute = re.compile('[0-9]+분')
            regex_number = re.compile('[0-9]+')
            hour_match = regex_hour.search(sentence)
            minute_match = regex_minute.search(sentence)
            time_set = ''

            int2str_time = {i: str_i for i, str_i in
                            zip(range(1, 60), ['0' + str(i) if i < 10 else str(i) for i in range(1, 60)])}
            if hour_match and minute_match:
                # (오전, 오후) 1시 30분으로 바꿔줘, 1시간 30분 뒤, 앞로 바꿔줘
                hour = regex_number.search(hour_match.group()).group()
                minute = regex_number.search(minute_match.group()).group()

                if '오전' in sentence:
                    time_set = time.strptime(f'{int2str_time[int(hour)]}:{int2str_time[int(minute)]}', '%H:%M')
                    time_set = time.strftime('%H:%M', time_set)

                elif '오후' in sentence:
                    time_set = time.strptime(f'{int(hour) + 12 % 24}:{int2str_time[int(minute)]}', '%H:%M')
                    time_set = time.strftime('%H:%M', time_set)

                elif '앞' in sentence or '전' in sentence:
                    # 현재 시간 구하기, 기준 1
                    now = datetime.datetime.now()
                    # 예약 시간이 기준인지 확인, 기준 2, 이거 정규식 고민해서 수정해야함 나중에
                    regex_reserved = re.compile('[이기예과지][전존거난]')
                    regex_reserved_match = regex_reserved.search(sentence)
                    if regex_reserved_match:
                        sql = 'select reservation from air_conditioner'
                        df = pd.read_sql(sql, db)
                        reserved_time = str(df['reservation'][0]).split()[-1][:-3]
                        time_delta = datetime.datetime.strptime(reserved_time, '%H:%M') - datetime.datetime.strptime(
                            f'{int2str_time[int(hour)]}:{int(minute)}', '%H:%M')
                        # datetime.timedelta 에서 string으로 전환하는 법
                        time_set = str(time_delta)[:-3]

                    else:
                        time_delta = datetime.datetime.strptime(f'{now.hour}:{now.minute}', '%H:%M') \
                                     - datetime.datetime.strptime(
                            f'{int2str_time[int(hour)]}:{int2str_time[int(minute)]}', '%H:%M')
                        time_set = str(time_delta)[:-3]

                    print(time_set)

                elif '뒤' in sentence or '후' in sentence:
                    # 현재 시간 구하기, 기준 1
                    now = datetime.datetime.now()
                    # 예약 시간이 기준인지 확인, 기준 2, 이거 정규식 고민해서 수정해야함 나중에
                    regex_reserved = re.compile('[이기예과지][전존거난]')
                    regex_reserved_match = regex_reserved.search(sentence)
                    if regex_reserved_match:
                        sql = 'select reservation from air_conditioner'
                        df = pd.read_sql(sql, db)
                        reserved_time = str(df['reservation'][0]).split()[-1][:-3]
                        time_delta = datetime.datetime.strptime(reserved_time, '%H:%M') \
                                     + datetime.timedelta(hours=int(hour), minutes=int(minute))
                        # datetime.timedelta 에서 string으로 전환하는 법
                        time_set = time_delta.strftime('%H:%M')

                    else:
                        time_delta = datetime.datetime.strptime(f'{now.hour}:{now.minute}', '%H:%M') \
                                     + datetime.timedelta(hours=int(hour), minutes=int(minute))
                        time_set = time_delta.strftime('%H:%M')
                else:
                    time_set = time.strptime(f'{int2str_time[int(hour)]}:{int2str_time[int(minute)]}', '%H:%M')
                    time_set = time.strftime('%H:%M', time_set)

            elif hour_match:
                hour = regex_number.search(hour_match.group()).group()
                minute = '30' if '반' in hour_match.group() else '00'

                if '오전' in sentence:
                    time_set = time.strptime(f'{int2str_time[int(hour)]}:{minute}', '%H:%M')
                    time_set = time.strftime('%H:%M', time_set)

                elif '오후' in sentence:
                    time_set = time.strptime(f'{int(hour) + 12 % 24}:{minute}', '%H:%M')
                    time_set = time.strftime('%H:%M', time_set)

                elif '앞' in sentence or '전' in sentence:
                    # 현재 시간 구하기, 기준 1
                    now = datetime.datetime.now()
                    # 예약 시간이 기준인지 확인, 기준 2, 이거 정규식 고민해서 수정해야함 나중에
                    regex_reserved = re.compile('[이기예과지][전존거난]')
                    regex_reserved_match = regex_reserved.search(sentence)
                    if regex_reserved_match:
                        sql = 'select reservation from air_conditioner'
                        df = pd.read_sql(sql, db)
                        reserved_time = str(df['reservation'][0]).split()[-1][:-3]
                        time_delta = datetime.datetime.strptime(reserved_time,
                                                                '%H:%M') - datetime.datetime.strptime(
                            f'{int2str_time[int(hour)]}:{minute}', '%H:%M')
                        # datetime.timedelta 에서 string으로 전환하는 법
                        time_set = str(time_delta)[:-3]

                    else:
                        time_delta = datetime.datetime.strptime(f'{now.hour}:{now.minute}', '%H:%M') \
                                     - datetime.datetime.strptime(f'{int2str_time[int(hour)]}:{minute}', '%H:%M')
                        time_set = str(time_delta)[:-3]

                    print(time_set)

                elif '뒤' in sentence or '후' in sentence:
                    # 현재 시간 구하기, 기준 1
                    now = datetime.datetime.now()
                    # 예약 시간이 기준인지 확인, 기준 2, 이거 정규식 고민해서 수정해야함 나중에
                    regex_reserved = re.compile('[이기예과지][전존거난]')
                    regex_reserved_match = regex_reserved.search(sentence)
                    if regex_reserved_match:
                        sql = 'select reservation from air_conditioner'
                        df = pd.read_sql(sql, db)
                        reserved_time = str(df['reservation'][0]).split()[-1][:-3]
                        time_delta = datetime.datetime.strptime(reserved_time, '%H:%M') \
                                     + datetime.timedelta(hours=int(hour), minutes=int(minute))
                        # datetime.timedelta 에서 string으로 전환하는 법
                        time_set = time_delta.strftime('%H:%M')

                    else:
                        time_delta = datetime.datetime.strptime(f'{now.hour}:{now.minute}', '%H:%M') \
                                     + datetime.timedelta(hours=int(hour), minutes=int(minute))
                        # 점검대상
                        time_set = time_delta.strftime('%H:%M')

                    print(time_set)
                else:
                    time_set = time.strptime(f'{int2str_time[int(hour)]}:{minute}', '%H:%M')
                    time_set = time.strftime('%H:%M', time_set)

            elif minute_match:
                minute = int(regex_number.search(minute_match.group()).group())
                # ???분 후로, 전으로 지금 시간에서 30분 후로로
                if '앞' in sentence or '전' in sentence:
                    # 현재 시간 구하기, 기준 1
                    now = datetime.datetime.now()
                    # 예약 시간이 기준인지 확인, 기준 2, 이거 정규식 고민해서 수정해야함 나중에
                    regex_reserved = re.compile('[이기예과지][전존거난]')
                    regex_reserved_match = regex_reserved.search(sentence)
                    if regex_reserved_match:
                        sql = 'select reservation from air_conditioner'
                        df = pd.read_sql(sql, db)
                        reserved_time = str(df['reservation'][0]).split()[-1][:-3]
                        time_delta = datetime.datetime.strptime(reserved_time,
                                                                '%H:%M') - datetime.datetime.strptime(
                            f'00:{minute}', '%H:%M')
                        # datetime.timedelta 에서 string으로 전환하는 법
                        time_set = str(time_delta)[:-3]

                    else:
                        time_delta = datetime.datetime.strptime(f'{now.hour}:{now.minute}', '%H:%M') \
                                     - datetime.datetime.strptime(f'00:{minute}', '%H:%M')
                        time_set = str(time_delta)[:-3]

                    print(time_set)

                elif '뒤' in sentence or '후' in sentence:
                    # 현재 시간 구하기, 기준 1
                    now = datetime.datetime.now()
                    # 예약 시간이 기준인지 확인, 기준 2, 이거 정규식 고민해서 수정해야함 나중에
                    regex_reserved = re.compile('[이기예과지][전존거난]')
                    regex_reserved_match = regex_reserved.search(sentence)
                    if regex_reserved_match:
                        sql = 'select reservation from air_conditioner'
                        df = pd.read_sql(sql, db)
                        reserved_time = str(df['reservation'][0]).split()[-1][:-3]
                        time_delta = datetime.datetime.strptime(reserved_time, '%H:%M') \
                                     + datetime.timedelta(minutes=int(minute))
                        # datetime.timedelta 에서 string으로 전환하는 법
                        time_set = time_delta.strftime('%H:%M')

                    else:
                        time_delta = datetime.datetime.strptime(f'{now.hour}:{now.minute}', '%H:%M') \
                                     + datetime.timedelta(minutes=int(minute))
                        time_set = time_delta.strftime('%H:%M')

                    print(time_set)

            sql = f'update air_conditioner set reservation = "{time_set}:00"'
            with db.cursor() as curs:
                curs.execute(sql)
            db.commit()

            result = f'에어컨의 변경된 예약시간은 {time_set} 입니다.'

        elif labels[1] == '모드제어':
            mode_list = ['냉방', '제습', '송풍', '자동']
            for mode in mode_list:
                if mode in sentence:
                    target_mode = mode
                    break
            sql = f'update air_conditioner set mode = "{target_mode}"'
            with db.cursor() as curs:
                curs.execute(sql)
            db.commit()

            result = f'에어컨 모드를 {target_mode}으로 바꾸었습니다.'

        elif labels[1] == '바람제어':
            int2kor_wind_intensity = {3: '강풍', 2: '중풍', 1: '약풍', 0: '무풍'}
            kor2int_wind_intensity = {int2kor_wind_intensity[num]: num for num in range(4)}
            for val, key in enumerate(['없', '약', '중', '강']):
                kor2int_wind_intensity[key] = val
            wind_intensity_regex = re.compile('[강중약무없][풍]*')
            wind_direction_list = ['좌우', '상하', '고정', '표준']
            wind_intensity_match = wind_intensity_regex.search(sentence)
            target_direction = ''

            for direction in wind_direction_list:
                if direction in sentence:
                    target_direction = direction
                    break

            intensity_sql = f'update air_conditioner ' \
                            f'set wind_intensity = {kor2int_wind_intensity[wind_intensity_match.group()] if wind_intensity_match else " "} '
            direction_sql = f'update air_conditioner set wind_direction = "{target_direction}"'

            if wind_intensity_match and target_direction != '':
                with db.cursor() as curs:
                    curs.execute(intensity_sql)
                    curs.execute(direction_sql)
                db.commit()
                result = f'에어컨 바람세기는 ' \
                         f'{wind_intensity_match.group() if "없" != wind_intensity_match.group() else "무풍"}으로 설정했고' \
                         f' 방향을 {target_direction}' \
                         f'{"으로" if "상하" != target_direction and "좌우" != target_direction else "로"} 바꾸었습니다.'

            elif wind_intensity_match:
                with db.cursor() as curs:
                    curs.execute(intensity_sql)
                db.commit()
                result = f'에어컨 바람세기는 ' \
                         f'{wind_intensity_match.group() if "없" != wind_intensity_match.group() else "무풍"}으로 ' \
                         f'설정했습니다.'

            elif wind_direction_list != '':
                with db.cursor() as curs:
                    curs.execute(direction_sql)
                db.commit()
                result = f'에어컨 방향을 {target_direction}' \
                         f'{"으로" if "상하" != target_direction and "좌우" != target_direction else "로"} 바꾸었습니다.'
            else:
                result = '정확한 발음으로 말해주세요'

        elif labels[1] == '온도제어':
            regex_number = re.compile('[0-9]+')
            temperature_find = regex_number.findall(sentence)
            if len(temperature_find) > 0:
                result = f'에어컨 온도를 {temperature_find[-1]}도로 바꿨습니다.'
                sql = f'update air_conditioner set temperature = {temperature_find[-1]}'
                with db.cursor() as curs:
                    curs.execute(sql)
                db.commit()

            else:
                result = '정확한 문장을 말해주시길 바랍니다.'

        else:
            result = '다른 표현으로 말해주길 바랍니다.'

    elif labels[0] == '공기청정기':
        if labels[1] == '전원':
            int2kor_power = {1: '켜진', 0: '꺼진'}
            sql = 'select power from air_purifier'
            df = pd.read_sql(sql, db)
            power = int2kor_power[df['power'][0]]
            result = f'공기청정기의 전원은 현재 {power} 상태입니다.'

        elif labels[1] == '필터':
            int2kor_filter = {1: '깨끗한', 0: '더러운'}
            sql = 'select filter from air_purifier'
            df = pd.read_sql(sql, db)
            air_filter = int2kor_filter[df['filter'][0]]
            result = f'공기청정기의 필터는 현재 {air_filter} 상태입니다.'

        elif labels[1] == '공기상태':
            int2kor_condition = {3: '매우나쁨', 2: '나쁨', 1: '보통', 0: '좋음'}
            sql = 'select air_condition from air_purifier'
            df = pd.read_sql(sql, db)
            air_condition = int2kor_condition[df['air_condition'][0]]
            result = f'공기의 상태는 현재 {air_condition} 입니다.'

        elif labels[1] == '세기':
            int2kor_intensity = {2: '강', 1: '중', 0: '약'}
            sql = 'select purification_intensity from air_purifier'
            df = pd.read_sql(sql, db)
            intensity = int2kor_intensity[df['purification_intensity'][0]]
            result = f'공기청정기의 청정 세기는 {intensity}입니다.'

        elif labels[1] == '모드':
            sql = 'select mode from air_purifier'
            df = pd.read_sql(sql, db)
            mode = df['mode'][0]
            result = f'공기청정기 청정 모드는 현재 {mode} 입니다.'

        elif labels[1] == '설정':
            int2kor_power = {1: '켜진', 0: '꺼진'}
            int2kor_filter = {1: '깨끗한', 0: '더러운'}
            int2kor_condition = {3: '매우나쁨', 2: '나쁨', 1: '보통', 0: '좋음'}
            int2kor_intensity = {2: '강', 1: '중', 0: '약'}

            sql = 'select * from air_purifier'
            df = pd.read_sql(sql, db)

            power = int2kor_power[df['power'][0]]
            air_filter = int2kor_filter[df['filter'][0]]
            air_condition = int2kor_condition[df['air_condition'][0]]
            intensity = int2kor_intensity[df['purification_intensity'][0]]
            mode = df['mode'][0]

            data = f'\t전원,{power}' + f'\t필터,{air_filter}' + f'\t공기상태,{air_condition}' \
                   + f'\t청정세기,{intensity}' + f'\t청정모드,{mode}'

            result = '공기청정기 설정 정보는 다음과 같습니다.' + data

        elif labels[1] == '전원제어':
            query_sql = 'select power from air_purifier'
            sql = ''
            df = pd.read_sql(query_sql, db)
            power = df['power'][0]
            if '켜' in sentence or '킬' in sentence or '키' in sentence:
                if power == 1:
                    result = '이미 공기청정기는 켜져있습니다.'
                else:
                    result = '공기청정기를 작동시켰습니다.'
                    sql = 'update air_purifier set power = 1'
            else:
                if power == 0:
                    result = '이미 공기청정기는 꺼져있습니다.'
                else:
                    result = '공기청정기를 작동 중지시켰습니다.'
                    sql = 'update air_purifier set power = 0'

            if sql != '':
                with db.cursor() as curs:
                    curs.execute(sql)
                db.commit()

        elif labels[1] == '모드제어':
            mode_list = ['일반', '표준', '자동', '수면']
            target_mode = ''
            for mode in mode_list:
                if mode in sentence:
                    target_mode = mode
                    break

            if target_mode != '':
                sql = f'update air_purifier set mode = "{target_mode}"'
                result = f'공기청정기의 모드를 {target_mode}으로 바꾸었습니다.'

                with db.cursor() as curs:
                    curs.execute(sql)
                db.commit()

            else:
                result = '정확한 발음으로 다시 말해주세요.'

        elif labels[1] == '세기제어':
            kor2int_intensity = {'강': 2, '중': 1, '약': 1}
            regex_intensity = re.compile('[강중약]')
            regex_intensity_match = regex_intensity.search(sentence)

            if regex_intensity_match:
                intensity = kor2int_intensity[regex_intensity_match.group()]
                sql = f'update air_purifier set purification_intensity = {intensity}'
                result = f'공기청정기의 청정 강도를 {regex_intensity_match.group()}으로 변경했습니다.'
                with db.cursor() as curs:
                    curs.execute(sql)
                db.commit()

            else:
                result = '정확한 발음으로 다시 말해 주세요.'

        else:
            result = '다른 표현으로 말해주길 바랍니다.'

    elif labels[0] == '냉장고':
        p = re.compile('냉[장동]실')
        m = p.search(sentence)
        if m:
            field = m.group()
        else:
            field = '냉장고'

        if labels[1] == '온도':
            # 냉장고 인지, 냉장실인지 냉동실인지 구분 필요
            sql = 'select * from refrigerator'
            df = pd.read_sql(sql, db)

            if field == '냉장실':
                result = f'냉장실의 온도는 {df["refrigerator_room_temperature"][0]}도 입니다.'

            elif field == '냉동실':
                result = f'냉동실의 온도는 {df["freezer_temperature"][0]}도 입니다.'

            else:
                result = f'냉장고의 온도는 냉장실, 냉동실 각각 {df["refrigerator_room_temperature"][0]}도, ' \
                         f'{df["freezer_temperature"][0]}도 입니다.'

        elif labels[1] == '유통기한':
            # 냉장고 인지, 냉장실인지 냉동실인지 구분 필요
            # 지났는지, 남았는지, 있는 전체인지.
            refrigerator_sql = 'select name, expiration_date from refrigerator_room'
            freezer_sql = 'select name, expiration_date from freezer'
            df_refrigerator_room = pd.read_sql(refrigerator_sql, db)
            df_freezer = pd.read_sql(freezer_sql, db)
            determine = get_expiration_date_determination(sentence)
            today = datetime.datetime.now().date()

            if field == '냉장실':
                if determine == '지난':
                    is_out_of_date = df_refrigerator_room['expiration_date'] < today
                    out_of_date_list = df_refrigerator_room[is_out_of_date]

                    data = ''
                    for name, date in zip(out_of_date_list['name'], out_of_date_list['expiration_date']):
                        data += f'\t{name}, {date}'

                    if data == '':
                        result = '냉장실에 유통기한이 지난 음식은 없습니다.'
                    else:
                        result = '냉장실에 유통기한이 지난 음식들은 다음과 같습니다.' + data

                elif determine == '임박한':
                    datetime.timedelta(days=7)
                    condition_under = (df_refrigerator_room['expiration_date'] > today)
                    condition_upper_than = (
                            df_refrigerator_room['expiration_date'] <= (today + datetime.timedelta(days=7)))
                    close_list = df_refrigerator_room[condition_under & condition_upper_than]

                    data = ''
                    for name, date in zip(close_list['name'], close_list['expiration_date']):
                        data += f'\t{name}, {date}'

                    if data == '':
                        result = '냉장실에 유통기한이 7일 이하로 남은 음식은 없습니다.'
                    else:
                        result = '냉장실에 유통기한이 7일 이하로 남은 음식들은 다음과 같습니다.' + data

                elif determine == '남은':
                    is_past = df_refrigerator_room['expiration_date'] > today
                    past_list = df_refrigerator_room[is_past]

                    data = ''
                    for name, date in zip(past_list['name'], past_list['expiration_date']):
                        data += f'\t{name}, {date}'

                    if data == '':
                        result = '냉장실에 유통기한이 남은 음식은 없습니다.'
                    else:
                        result = '냉장실에 유통기한이 남은 음식들은 다음과 같습니다.' + data

                else:
                    data = ''
                    for name, date in zip(df_refrigerator_room['name'], df_refrigerator_room['expiration_date']):
                        data += f'\t{name}, {date}'

                    if data == '':
                        result = '냉장실의 음식들은 없습니다.'
                    else:
                        result = '냉장실의 음식들의 유통기한은 다음과 같습니다.' + data

            elif field == '냉동실':
                if determine == '지난':
                    is_out_of_date = df_freezer['expiration_date'] < today
                    out_of_date_list = df_freezer[is_out_of_date]

                    data = ''
                    for name, date in zip(out_of_date_list['name'], out_of_date_list['expiration_date']):
                        data += f'\t{name}, {date}'

                    if data == '':
                        result = '냉장실에 유통기한이 지난 음식은 없습니다.'
                    else:
                        result = '냉장실에 유통기한이 지난 음식들은 다음과 같습니다.' + data

                elif determine == '임박한':
                    datetime.timedelta(days=7)
                    condition_under = (df_freezer['expiration_date'] > today)
                    condition_upper_than = (
                            df_freezer['expiration_date'] <= (today + datetime.timedelta(days=7)))
                    close_list = df_freezer[condition_under & condition_upper_than]

                    data = ''
                    for name, date in zip(close_list['name'], close_list['expiration_date']):
                        data += f'\t{name}, {date}'

                    if data == '':
                        result = '냉장실에 유통기한이 7일 이하로 남은 음식은 없습니다.'
                    else:
                        result = '냉장실에 유통기한이 7일 이하로 남은 음식들은 다음과 같습니다.' + data

                elif determine == '남은':
                    is_past = df_refrigerator_room['expiration_date'] > today
                    past_list = df_refrigerator_room[is_past]

                    data = ''
                    for name, date in zip(past_list['name'], past_list['expiration_date']):
                        data += f'\t{name}, {date}'

                    if data == '':
                        result = '냉장실에 유통기한이 남은 음식은 없습니다.'
                    else:
                        result = '냉장실에 유통기한이 남은 음식들은 다음과 같습니다.' + data

                else:
                    data = ''
                    for name, date in zip(df_freezer['name'], df_freezer['expiration_date']):
                        data += f'\t{name}, {date}'

                    if data == '':
                        result = '냉장실의 음식들은 없습니다.'
                    else:
                        result = '냉장실의 음식들의 유통기한은 다음과 같습니다.' + data

            else:
                if determine == '지난':
                    is_out_of_date_freezer = df_freezer['expiration_date'] < today
                    out_of_date_list_freezer = df_freezer[is_out_of_date_freezer]
                    is_out_of_date_refrigerator = df_refrigerator_room['expiration_date'] < today
                    out_of_date_list_refrigerator = df_refrigerator_room[is_out_of_date_refrigerator]

                    data = ''
                    for name, date in zip(out_of_date_list_freezer['name'],
                                          out_of_date_list_freezer['expiration_date']):
                        data += f'\t{name}, {date}'

                    for name, date in zip(out_of_date_list_refrigerator['name'],
                                          out_of_date_list_refrigerator['expiration_date']):
                        data += f'\t{name}, {date}'

                    if data == '':
                        result = '냉장고에 유통기한이 지난 음식은 없습니다.'
                    else:
                        result = '냉장고에 유통기한이 지난 음식들은 다음과 같습니다.' + data

                elif determine == '임박한':
                    datetime.timedelta(days=7)
                    condition_under_freezer = (df_freezer['expiration_date'] > today)
                    condition_upper_than_freezer = (
                            df_freezer['expiration_date'] <= (today + datetime.timedelta(days=7)))
                    close_list_freezer = df_freezer[condition_under_freezer & condition_upper_than_freezer]

                    condition_under_refrigerator = (df_refrigerator_room['expiration_date'] > today)
                    condition_upper_than_refrigerator = (
                            df_refrigerator_room['expiration_date'] <= (today + datetime.timedelta(days=7)))
                    close_list_refrigerator_room = df_refrigerator_room[
                        condition_under_refrigerator & condition_upper_than_refrigerator]

                    data = ''
                    for name, date in zip(close_list_freezer['name'],
                                          close_list_freezer['expiration_date']):
                        data += f'\t{name}, {date}'

                    for name, date in zip(close_list_refrigerator_room['name'],
                                          close_list_refrigerator_room['expiration_date']):
                        data += f'\t{name}, {date}'

                    if data == '':
                        result = '냉장고에 유통기한이 7일 이하로 남은 음식은 없습니다.'
                    else:
                        result = '냉장고에 유통기한이 7일 이하로 남은 음식들은 다음과 같습니다.' + data

                elif determine == '남은':
                    is_past_freezer = df_freezer['expiration_date'] > today
                    past_list_freezer = df_freezer[is_past_freezer]

                    is_past_refrigerator = df_refrigerator_room['expiration_date'] > today
                    past_list_refrigerator = df_refrigerator_room[is_past_refrigerator]

                    data = ''
                    for name, date in zip(past_list_freezer['name'], past_list_freezer['expiration_date']):
                        data += f'\t{name}, {date}'

                    for name, date in zip(past_list_refrigerator['name'], past_list_refrigerator['expiration_date']):
                        data += f'\t{name}, {date}'

                    if data == '':
                        result = '냉장고에 유통기한이 남은 음식은 없습니다.'
                    else:
                        result = '냉장고에 유통기한이 남은 음식들은 다음과 같습니다.' + data

                else:
                    data = ''
                    for name, date in zip(df_freezer['name'], df_freezer['expiration_date']):
                        data += f'\t{name}, {date}'

                    for name, date in zip(df_refrigerator_room['name'], df_refrigerator_room['expiration_date']):
                        data += f'\t{name}, {date}'

                    if data == '':
                        result = '냉장고에 음식들은 없습니다.'
                    else:
                        result = '냉장고에 있는 음식들의 유통기한은 다음과 같습니다.' + data

        elif labels[1] == '음식':
            refrigerator_sql = 'select name, category from refrigerator_room'
            freezer_sql = 'select name, category from freezer'
            df_refrigerator_room = pd.read_sql(refrigerator_sql, db)
            df_freezer = pd.read_sql(freezer_sql, db)

            category_list = list(set(df_freezer['category'])) + list(set(df_refrigerator_room['category']))
            target_category = ''
            for category in category_list:
                if category in sentence:
                    target_category = category
                    break

            if field == '냉장실':
                if target_category != '':
                    is_category_refrigerator = df_refrigerator_room['category'] == target_category

                    category_refrigerator = df_refrigerator_room[is_category_refrigerator]

                    data = ''
                    for food in category_refrigerator['name']:
                        data += f'\t{food}'

                    result = f'냉장실의 {target_category} 목록은 다음과 같습니다.' + data
                else:
                    if '음식' in sentence or '식품' in sentence or '먹' in sentence:
                        data = ''
                        for food in df_refrigerator_room['name']:
                            data += f'\t{food}'

                        result = '냉장실의 음식들은 다음과 같습니다.' + data
                    else:
                        result = '원하시는 종류의 음식은 냉장실에 존재하지 않습니다.'

            elif field == '냉동실':
                if target_category != '':
                    is_category_freezer = df_freezer['category'] == target_category

                    category_freezer = df_freezer[is_category_freezer]

                    data = ''
                    for food in category_freezer['name']:
                        data += f'\t{food}'

                    result = f'냉동실의 {target_category} 목록은 다음과 같습니다.' + data
                else:
                    if '음식' in sentence or '식품' in sentence or '먹' in sentence:
                        data = ''
                        for food in df_freezer['name']:
                            data += f'\t{food}'

                        result = '냉동실의 음식들은 다음과 같습니다.' + data
                    else:
                        result = '원하시는 종류의 음식은 냉동실에 존재하지 않습니다.'

            else:
                if target_category != '':
                    is_category_freezer = df_freezer['category'] == target_category
                    is_category_refrigerator = df_refrigerator_room['category'] == target_category

                    category_freezer = df_freezer[is_category_freezer]
                    category_refrigerator = df_refrigerator_room[is_category_refrigerator]

                    data = ''
                    for food in category_freezer['name']:
                        data += f'\t{food}'

                    for food in category_refrigerator['name']:
                        data += f'\t{food}'

                    result = f'냉장고의 {target_category} 목록은 다음과 같습니다.' + data
                else:
                    if '음식' in sentence or '식품' in sentence or '먹' in sentence:
                        data = ''
                        for food in df_refrigerator_room['name']:
                            data += f'\t{food}'

                        for food in df_freezer['name']:
                            data += f'\t{food}'

                        result = '냉장고의 음식들은 다음과 같습니다.' + data
                    else:
                        result = '원하시는 종류의 음식은 냉장고에 존재하지 않습니다.'

        elif labels[1] == '온도제어':
            regex_number = re.compile('[0-9]+')
            number_match = regex_number.search(sentence)
            if field == '냉장실':
                if number_match:
                    sql = f'update refrigerator set refrigerator_room_temperature = {number_match.group()}'
                    result = f'냉장실의 온도를 {number_match.group()}도로 설정했습니다.'
                    with db.cursor() as curs:
                        curs.execute(sql)
                    db.commit()

            elif field == '냉동실':
                if number_match:
                    sql = f'update refrigerator set freezer_temperature = -{number_match.group()}'
                    result = f'냉동실의 온도를 -{number_match.group()}도로 설정했습니다.'
                    with db.cursor() as curs:
                        curs.execute(sql)
                    db.commit()
            else:
                if number_match:
                    if '-' in sentence or '영하' in sentence:
                        sql = f'update refrigerator set freezer_temperature = -{number_match.group()}'
                        result = f'냉동실의 온도를 -{number_match.group()}도로 설정했습니다.'
                        with db.cursor() as curs:
                            curs.execute(sql)
                        db.commit()
                    else:
                        sql = f'update refrigerator set refrigerator_room_temperature = {number_match.group()}'
                        result = f'냉장실의 온도를 {number_match.group()}도로 설정했습니다.'
                        with db.cursor() as curs:
                            curs.execute(sql)
                        db.commit()
                else:
                    result = '정확한 발음으로 다시 말해주세요.'
        else:
            result = '다른 표현으로 말해주길 바랍니다.'

    elif labels[0] == '전구':
        bulb_location_list = ['거실', '안방', '화장실', '작은 방', '큰 방', '주방', '다목적 실', '다목적실', '베란다']
        target_location = ''
        if labels[1] == '전원':
            for location in bulb_location_list:
                if location in sentence:
                    target_location = ''.join(location.split(' '))
                    break

            print(f'target location = {target_location}')

            if target_location != '':
                sql = f'select power from bulb where location = "{target_location}"'
            else:
                sql = f'select location, power from bulb'

            df = pd.read_sql(sql, db)
            int2kor_power = {1: '켜져있습니다', 0: '꺼져있습니다'}
            int2kor_power2 = {1: '켜짐', 0: '꺼짐'}
            if target_location != '':
                result = f'현재 {target_location} 전구 전원은 {int2kor_power[df["power"][0]]}'
            else:
                data = ''
                for location, color in zip(df['location'], df['power']):
                    data += f'\t{location},{int2kor_power2[color]}'

                result = '현재 집안의 전구들의 전원은 다음과 같습니다.' + data

        elif labels[1] == '밝기':
            for location in bulb_location_list:
                if location in sentence:
                    target_location = ''.join(location.split(' '))
                    break

            if target_location != '':
                sql = f'select light from bulb where location = "{target_location}"'
            else:
                sql = f'select location, light from bulb'

            df = pd.read_sql(sql, db)

            if target_location != '':
                result = f'현재 {target_location} 전구의 밝기는 {df["light"][0]} 입니다.'
            else:
                data = ''
                for location, color in zip(df['location'], df['light']):
                    data += f'\t{location},{color}'
                result = '현재 전구들의 밝기 입니다.' + data

        elif labels[1] == '색상':
            for location in bulb_location_list:
                if location in sentence:
                    target_location = ''.join(location.split(' '))
                    break

            if target_location != '':
                sql = f'select color from bulb where location = "{target_location}"'
            else:
                sql = f'select location, color from bulb'

            df = pd.read_sql(sql, db)

            if target_location != '':
                result = f'현재 {target_location} 전구의 색상은 {df["color"][0]} 입니다.'
            else:
                data = ''
                for location, color in zip(df['location'], df['color']):
                    data += f'\t{location},{color}'
                result = '현재 전구들의 색상정보 입니다.' + data

        elif labels[1] == '설정':
            for location in bulb_location_list:
                if location in sentence:
                    target_location = ''.join(location.split(' '))
                    break

            if target_location != '':
                sql = f'select * from bulb where location = "{target_location}"'
            else:
                sql = f'select * from bulb'

            df = pd.read_sql(sql, db)

            int2kor_power2 = {1: '켜짐', 0: '꺼짐'}
            if target_location != '':
                data = f'\t전원,{int2kor_power2[df["power"][0]]}' + f'\t빛,{df["light"][0]}' \
                       + f'\t색상,{df["color"][0]}'
                result = f'현재 {target_location} 전구의 설정은 다음과 같습니다.' + data
            else:
                data = ''
                for location, power, light, color in zip(df['location'], df['power'], df["light"], df['color']):
                    data += f'\t{location},{int2kor_power2[power]},{light},{color}'
                result = '현재 전구들의 설정정보 입니다.' + data

        elif labels[1] == '전원제어':
            for location in bulb_location_list:
                if location in sentence:
                    target_location = ''.join(location.split(' '))
                    break

            if '켜' in sentence or '킬' in sentence:
                if target_location != '':
                    sql = f'update bulb set power = 1 where location = "{target_location}"'
                    result = f'{target_location}의 전구 전원을 켰습니다.'
                else:
                    sql = 'update bulb set power = 1'
                    result = '모든 전구의 전원을 켰습니다.'
            else:
                if target_location != '':
                    sql = f'update bulb set power = 0 where location = "{target_location}"'
                    result = f'{target_location}의 전구 전원을 껐습니다.'
                else:
                    sql = 'update bulb set power = 0'
                    result = '모든 전구의 전원을 껐습니다.'

            with db.cursor() as curs:
                curs.execute(sql)
                db.commit()

        elif labels[1] == '밝기제어':
            for location in bulb_location_list:
                if location in sentence:
                    target_location = ''.join(location.split(' '))
                    break
            regex_light = re.compile('[강중약]')
            light_match = regex_light.search(sentence)
            if light_match is not None:
                light = light_match.group()
                if target_location != '':
                    sql = f'update bulb set light = "{light}" where location = "{target_location}"'
                    result = f'{target_location}의 전구 밝기를 {light}로 했습니다.'
                else:
                    sql = f'update bulb set light = "{light}"'
                    result = f'모든 전구의 전구 밝기를 {light}로 했습니다.'
            else:
                if '높' in sentence or '위' in sentence or '올' in sentence:
                    if target_location != '':
                        query_sql = f'select light from bulb where location = "{target_location}"'
                        df = pd.read_sql(query_sql, db)
                        light_degree = df['light'][0]

                        if '2단계' in sentence or '두 단계' in sentence:
                            stage = "두 단계"
                            repeat = 2
                        else:
                            stage = "한 단계"
                            repeat = 1

                        for _ in range(repeat):
                            if light_degree == '중':
                                light_degree = '강'
                            elif light_degree == '약':
                                light_degree = '중'

                        sql = f'update bulb set light = "{light_degree}" where location = "{target_location}"'
                        result = f'{target_location}의 전구 밝기를 {stage} 높였습니다.'
                    else:
                        query_sql = f'select location, light from bulb where location = "{target_location}"'
                        df = pd.read_sql(query_sql, db)

                        for light_degree, location in zip(df['light'], df['location']):
                            if '2단계' in sentence or '두 단계' in sentence:
                                stage = "두 단계"
                                repeat = 2
                            else:
                                stage = "한 단계"
                                repeat = 1

                            for _ in range(repeat):
                                if light_degree == '중':
                                    light_degree = '강'
                                elif light_degree == '약':
                                    light_degree = '중'

                            sql = f'update bulb set light = "{light_degree}" where location = "{location}"'

                            with db.cursor() as curs:
                                curs.execute(sql)
                                db.commit()

                        result = f'모든 전구의 전구 밝기를 {stage} 높였습니다.'

                elif '낮' in sentence or '아래' in sentence or '내려' in sentence or '내리' in sentence:
                    if target_location != '':
                        query_sql = f'select light from bulb where location = "{target_location}"'
                        df = pd.read_sql(query_sql, db)
                        light_degree = df['light'][0]

                        if '2단계' in sentence or '두 단계' in sentence:
                            stage = "두 단계"
                            repeat = 2
                        else:
                            stage = "한 단계"
                            repeat = 1

                        for _ in range(repeat):
                            if light_degree == '중':
                                light_degree = '약'
                            elif light_degree == '강':
                                light_degree = '중'

                        sql = f'update bulb set light = "{light_degree}" where location = "{target_location}"'
                        result = f'{target_location}의 전구 밝기를 {stage} 낮췄습니다.'
                    else:
                        query_sql = f'select location, light from bulb where location = "{target_location}"'
                        df = pd.read_sql(query_sql, db)

                        for light_degree, location in zip(df['light'], df['location']):
                            if '2단계' in sentence or '두 단계' in sentence:
                                stage = "두 단계"
                                repeat = 2
                            else:
                                stage = "한 단계"
                                repeat = 1

                            for _ in range(repeat):
                                if light_degree == '중':
                                    light_degree = '약'
                                elif light_degree == '강':
                                    light_degree = '중'

                            sql = f'update bulb set light = "{light_degree}" where location = "{location}"'

                            with db.cursor() as curs:
                                curs.execute(sql)
                                db.commit()

                        result = f'모든 전구의 전구 밝기를 {stage} 낮췄습니다.'
                else:
                    result = '현재의 전구 밝기를 유지합니다.'

            with db.cursor() as curs:
                curs.execute(sql)
                db.commit()

        elif labels[1] == '색깔제어':
            for location in bulb_location_list:
                if location in sentence:
                    target_location = ''.join(location.split(' '))
                    break

            # 나중에 밝은 색, 분위기 있는 색, 따뜻한 색, 차가운 색, 시원한 색, 푸른, 적색, 청색 이런 것 추가하자.
            if '밝은' in sentence or '주광' in sentence:
                color_set = '주광'
            elif '분위기' in sentence or '무드' in sentence:
                color_set = '무드'
            elif '따' in sentence or '웜톤' in sentence:
                color_set = '웜톤'
            elif '차가운' in sentence or '쿨톤' in sentence or '시원' in sentence:
                color_set = '쿨톤'
            elif '푸른' in sentence or '파랑' in sentence or '청' in sentence:
                color_set = '파랑'
            elif '붉은' in sentence or '빨강' in sentence or '적' in sentence:
                color_set = '빨강'

            if target_location != '':

                sql = f'update bulb set color = "{color_set}" where location = "{target_location}"'
                result = f'{target_location}의 전구의 색을 {color_set}으로 설정했습니다.'
            else:
                sql = f'update bulb set color = "{color_set}"'
                result = f'모든 전구의 색을 {color_set}으로 설정했습니다.'

            with db.cursor() as curs:
                curs.execute(sql)
                db.commit()

        else:
            result = '다른 표현으로 말해주길 바랍니다.'

    elif labels[0] == '전기':
        if labels[1] == '날짜':
            result = '전기 날짜'
            regex = re.compile('[0-9]+')
            match_month_int = regex.search(sentence)
            if match_month_int is not None:
                # 3개월 전의,
                today_month = int(time.strftime('%m', time.localtime(time.time())))
                month_int = int(match_month_int.group())
                if '개월 전' in sentence:
                    sql = "select charge, energy from electricity" \
                          f" where month = {abs(today_month - month_int)}"

                    df = pd.read_sql(sql, db)
                    info, values = get_electricity_sentence_part(df, sentence, labels[0])

                    result = f'{month_int}개월 전의 전기{info} {values} 입니다.'
                else:
                    sql = "select charge, energy from electricity" \
                          f" where month = {month_int}"

                    df = pd.read_sql(sql, db)
                    info, values = get_electricity_sentence_part(df, sentence, labels[0])

                    result = f'{month_int}월의 전기{info} {values} 입니다.'

            elif '이번 달' in sentence:
                today_month = int(time.strftime('%m', time.localtime(time.time())))
                sql = "select charge, energy from electricity" \
                      f" where month = {today_month}"

                df = pd.read_sql(sql, db)
                info, values = get_electricity_sentence_part(df, sentence, labels[0])

                result = f'이번 달의 전기{info} {values} 입니다.'

            elif '지난 달' in sentence:
                last_month = int(time.strftime('%m', time.localtime(time.time()))) - 1
                sql = "select charge, energy from electricity" \
                      f" where month = {last_month}"

                df = pd.read_sql(sql, db)
                info, values = get_electricity_sentence_part(df, sentence, labels[0])

                result = f'지난 달의 전기{info} {values} 입니다.'

        elif labels[1] == '기간':
            result = '전기 기간'
            # 최근 3개월, 6월 부터 9월
            regex = [re.compile('[0-9]+월'), re.compile('[0-9]+개월'),
                     re.compile('[두세내다섯여섯일곱덟아홉열 0-9]+?달'), re.compile('[0-9]+?년')]
            regex2 = re.compile('[0-9]+')
            regex3 = re.compile('[두세내다섯여섯일곱덟아홉열]+')
            data_findall = [p.findall(sentence) for p in regex]

            if data_findall[0]:
                # 모월 부터 모월까지
                month = [regex2.match(data).group() for data in data_findall[0]]

                sql = f'SELECT sum(charge), sum(energy) from electricity \
                        where month <= {month[1]} and month >= {month[0]}'

                df = pd.read_sql(sql, db)

                info, values = get_electricity_sentence_part(df, sentence, labels[1])

                result = f'{month[0]}월 부터 {month[1]}월의 전기{info} {values} 입니다.'
                # result = f'{data_findall[0][0]} 부터 {data_findall[0][1]}의 전기 기간 문장입니다.'
            elif data_findall[1]:
                # 개월
                today_month = int(time.strftime('%m', time.localtime(time.time())))
                month = [[regex2.match(data).group()] for data in data_findall[1]]
                sql = f'SELECT sum(charge), sum(energy) from electricity \
                        where month <= {today_month} and month >= {today_month - int(month[0][0])}'

                df = pd.read_sql(sql, db)

                info, values = get_electricity_sentence_part(df, sentence, labels[1])

                result = f'최근 {month[0][0]}개월의 전기{info} {values} 입니다.'
            elif data_findall[2]:
                # 달
                month = [
                    [regex2.search(data_findall[2][0]).group()] \
                        if regex2.search(data_findall[2][0]) is not None else [],
                    [regex3.search(data_findall[2][0]).group()] \
                        if regex3.search(data_findall[2][0]) is not None else []
                ]

                if '열 한 달' in sentence:
                    today_month = int(time.strftime('%m', time.localtime(time.time())))

                    if today_month < 12:
                        sql = f'SELECT sum(charge), sum(energy) from electricity where month <= {today_month}'
                    else:
                        sql = 'SELECT sum(charge), sum(energy) from electricity where month <= 11'

                    df = pd.read_sql(sql, db)

                    info, values = get_electricity_sentence_part(df, sentence, labels[1])

                    result = f'최근 열 한 달의 전기{info} {values} 입니다.'

                elif len(month[0]) == 1:
                    # 숫자로 들어옴
                    print(month[0][0])
                    month_int = int(month[0][0])
                    today_month = int(time.strftime('%m', time.localtime(time.time())))

                    if today_month - month_int < 0:
                        sql = f'SELECT sum(charge), sum(energy) from electricity \
                                where month <= {today_month}'
                    else:
                        sql = f'SELECT sum(charge), sum(energy) from electricity \
                                where month <= {today_month} and month >= {today_month - month_int}'

                    df = pd.read_sql(sql, db)
                    info, values = get_electricity_sentence_part(df, sentence, labels[1])

                    # 보내기 전에 기간을 숫자에서 한국어로 바꾸는게 듣기 좋으니 시간나면 추가하자.
                    result = f'최근 {data_findall[2][0]}의 전기{info} {values} 입니다.'
                elif len(month[1]) == 1:
                    # 한국어로 들어옴
                    kor2int = {'두': 2, '세': 3, '네': 4, '내': 4, '다섯': 5, '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9, '열': 10}
                    int2kor = {2: '두', 3: '세'}
                    today_month = int(time.strftime('%m', time.localtime(time.time())))
                    month_int = kor2int[month[1][0]]

                    if today_month - month_int < 0:
                        sql = f'SELECT sum(charge), sum(energy) from electricity \
                                                    where month <= {today_month}'
                    else:
                        sql = f'SELECT sum(charge), sum(energy) from electricity \
                                                    where month <= {today_month} and month >= {today_month - month_int}'

                    df = pd.read_sql(sql, db)
                    info, values = get_electricity_sentence_part(df, sentence, labels[1])

                    # 보내기 전에 기간을 숫자에서 한국어로 바꾸는게 듣기 좋으니 시간나면 추가하자.
                    result = f'최근 {month_int}달 의 전기{info} {values} 입니다.'

            elif data_findall[3]:
                # 모 년, 하지만 DB에는 1년치만 가지고 있어서. 감안해서 작성
                year = [regex2.match(data).group() for data in data_findall[3]]

                # 어짜피 1년치라 걍 다 하면 된다.
                sql = 'SELECT sum(charge), sum(energy) from electricity'

                df = pd.read_sql(sql, db)
                info, values = get_electricity_sentence_part(df, sentence, labels[1])

                result = f'지난 {year[0]}년의 전기{info} {values} 입니다.'
            elif '올해' in sentence_morphs:
                result = '올해의 전기 기간 문장입니다.'
            else:
                result = '전기 기간 문장입니다.'
        else:
            result = '다른 표현으로 말해주길 바랍니다.'

    print(result)
    # result = '임시 문장입니다.'
    return result


def get_title_set(df_tv_guide):
    title_list = list(set(df_tv_guide['title']))
    title_list_before_remove_sc = list(set(map(lambda title: re.sub('[0-9(]+.*[부회)]+', '', title),
                                               title_list)))
    extracted_title_list = list(map(lambda title: re.sub('[!?]+', '', title),
                                    title_list_before_remove_sc))

    print(extracted_title_list)
    return extracted_title_list


def get_expiration_date_determination(sentence: str):
    result = '전체'
    out_of_date_list = ['지난', '남지 않은', '초과한']
    past_list = ['남은', '지나지 않은', '초과하지 않은']
    close_list = ['얼마 남지 않은', '별로 남지 않은', '그다지 남지 않은', '아직 지나지 않은', '아직 초과하지 않은', '임박']

    for check in close_list:
        if check in sentence:
            result = '임박한'
            break

    if result == '전체':
        for check in out_of_date_list:
            if check in sentence:
                result = '지난'
                break

    if result == '전체':
        for check in past_list:
            if check in sentence:
                result = '남은'
                break

    return result


def set_air_conditioner_reservation_sentence_part(hour, minute):
    if hour > 0 and minute > 0:
        result = f'{hour}시간 {minute}분'
    elif hour > 0:
        result = f'{hour}시간'
    else:
        result = f'{minute}분'

    return result


def get_air_conditioner_reservation_sentence_part(duration: int):
    hour = duration // 60
    minute = duration % 20

    if hour > 0 and minute > 0:
        result = f'{hour}시간 {minute}분'
    elif hour > 0:
        result = f'{hour}시간'
    else:
        result = f'{minute}분'

    return result


def get_electricity_sentence_part(df, sentence, ti_label):
    if '요금' in sentence or '사용료' in sentence or '비용' in sentence:
        if '사용량' in sentence or '사용한 양' in sentence or '사용된' in sentence:
            if ti_label == '기간':
                info = '요금과 사용량의 합계는'
                values = f'{int(df["sum(charge)"][0])}원과 {int(df["sum(energy)"][0])}kWh'
            else:
                info = '요금과 사용량은'
                values = f'{int(df["charge"][0])}원과 {int(df["energy"][0])}kWh'
        else:
            if ti_label == '기간':
                info = '요금의 합계는'
                values = f'{int(df["sum(charge)"][0])}원'
            else:
                info = '요금은'
                values = f'{int(df["charge"][0])}원'
    else:
        if ti_label == '기간':
            info = '사용량의 합계는'
            values = f'{int(df["sum(energy)"][0])}kWh'
        else:
            info = '사용량은'
            values = f'{int(df["energy"][0])}kWh'

    return info, values


def start_QA_system():
    # mecab = Mecab()
    while True:
        # 클라이너트가 보낸 메시지를 수신하기 위해 대기
        print('Starting listening')
        server_socket.listen()
        client_socket, addr = server_socket.accept()
        print(f'Connected by {addr}')
        data = client_socket.recv(1024)

        # 빈 문자열을 수신하면 루프를 중지함
        if not data or data.decode() == '종료':
            break

        # 수신받은 문자열을 출력한다.
        print(f'Received from {addr}\n {data.decode()}')
        labels = run_model(data.decode())
        result = Query(data.decode(), labels)
        # 받은 문자열을 다시 클라이언트로 전송해준다. 에코(메아리)
        client_socket.sendall(result.encode())

    # 소켓을 닫는다.
    client_socket.close()
    server_socket.close()
    db.close()


if __name__ == "__main__":
    # 수행시간 측정, 사용자 경험을 좋게 하기 위함
    start_QA_system()
