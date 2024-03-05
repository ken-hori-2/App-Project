from action_list_app import app
from flask import render_template, request, redirect, url_for
import sqlite3
import csv
import datetime

DATABASE = 'database.db'

# add
# with open('/Users/ken/Desktop/App-Project/action_list_app/input/date_action.csv', 'w') as f: # input/date_action.csv', 'w') as f:
# with open('/home/ubuntu/App-Project/action_list_app/input/date_action.csv', 'w') as f:
# dt_now = datetime.datetime.now()
dt_now_jst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
print(dt_now_jst)
date = dt_now_jst.strftime('%Y%m%d%H%M')
filename = '/home/ubuntu/App-Project/action_list_app/input/date_action_' + date + '.csv'
with open(filename, 'w') as f:
    writer = csv.writer(f)
    # writer.writerows([info["date"], info["action"], info["place"]])
    writer.writerow(['date', 'action', 'place'])


@app.route('/')
def index():
    # info = [
    #     {
    #     'action' : '運動',
    #     'place' : 'ジム',
    #     'date' : '2024/2/14'
    #     },
    #     {
    #     'action' : '仕事',
    #     'place' : '会社',
    #     'date' : '2024/2/14'
    #     }
    # ]

    con = sqlite3.connect(DATABASE)
    db_info = con.execute('SELECT * FROM info').fetchall()
    con.close()

    info = []
    for row in db_info:
        info.append({'action': row[0], 'place': row[1], 'date': row[2]})

    # # add
    # with open(dir_path, 'w') as f: # input/date_action.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     # writer.writerows([info["date"], info["action"], info["place"]])
    #     writer.writerow(['date', 'action', 'place'])

    # with open(dir_path) as f: # data/temp/sample_writer_row.csv') as f:
    #     print("**********")
    #     print(f.read())
    #     print("**********")

    # ########## 2024/02/27 ##########
    # filename = '/home/ubuntu/App-Project/action_list_app/input/date_action_' + date + '.csv'
    # with open(filename, 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerows([['date', 'action', 'place']])
    # ########## 2024/02/27 ##########







    return render_template(
        'index.html',
        info = info
    )

@app.route('/form')
def form():
    return render_template(
        'form.html'
    )


@app.route('/register', methods=['POST'])
def register():
    action = request.form['action']
    place = request.form['place']
    date = request.form['date']

    con = sqlite3.connect(DATABASE)
    con.execute('INSERT INTO info VALUES(?, ?, ?)',
                [action, place, date])
    
    con.commit()
    con.close()

    # add
    ########## 2024/02/27 ##########
    # dt_now = datetime.datetime.now()
    # print(dt_now)
    # # date = dt_now.strftime('%Y年%m月%d日 %H:%M:%S')
    # date = dt_now.strftime('%Y%m%d%H%M') # :%S')
    # print(date)
    dt_now_jst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    print(dt_now_jst)
    date = dt_now_jst.strftime('%Y%m%d%H%M')

    # next_action_list = [
    #     "wakeup",
    #     "go work",
    #     "breakfast",
    #     "coffee",
    #     "working",
    #     "lunch",
    #     "tooth brush",
    #     "going out",
    #     "meeting",
    #     "study(English)",
    #     "gym",
    #     "buy dinner",
    #     "back home",
    #     "bath",
    #     "study(ML)",
    #     "go bed",
    #     "study",
    #     "Movie",
    #     "study(AWS)",
    #     "study(other)",
    #     "TV/Youtube",
    #     "sleep",
    # ]
    next_action_list = {
        "wakeup":0,
        "go work":1,
        "breakfast":2,
        "coffee":3,
        "working":4,
        "lunch":5,
        "tooth brush":6,
        "going out":7,
        "meeting":8,
        "study(English)":9,
        "gym":10,
        "buy dinner":11,
        "back home":12,
        "bath":13,
        "study(ML)":14,
        "go bed":15,
        "study":16,
        "Movie":17,
        "study(AWS)":18,
        "study(other)":19,
        "TV/Youtube":20,
        "sleep":21,
    }
    # 入力された行動を辞書から検索してラベルに変換（今回は配列順なので辞書形式ではなくてもいい）
    try:
        action = next_action_list[action]
    except:
        action = -1 # 入力された行動が辞書内にない場合は-1
    ########## 2024/02/27 ##########

    # with open('/home/ubuntu/App-Project/action_list_app/input/date_action.csv', 'a') as f: # input/date_action.csv', 'w') as f:
    with open(filename, 'a') as f: # input/date_action.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows([[date, action, place]])
        # writer.writerow(['a', 'b', 'c'])

    # with open('/home/ubuntu/App-Project/action_list_app/input/date_action.csv') as f: # data/temp/sample_writer_row.csv') as f:
    with open(filename) as f: # input/date_action.csv', 'w') as f:
        print("**********")
        print(f.read())
        print("**********")
    
    
    return redirect(url_for('index'))


# # from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime
# from io import StringIO
# import pandas as pd
# # @app.route('/download/<obj>/')
# # def download(obj):

# #     f = StringIO()
# #     writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL, lineterminator="\n")

# #     if obj == 'users':
# #         writer.writerow(['id','username','gender','age','created_at'])
# #         for u in User.query.all():
# #             writer.writerow([u.id, u.username,u.gender,u.age,u.created_at])

# #     res = make_response()
# #     res.data = f.getvalue()
# #     res.headers['Content-Type'] = 'text/csv'
# #     res.headers['Content-Disposition'] = 'attachment; filename='+ obj +'.csv'
# #     return res
# @app.route('/')
# def csv_display(): 
#     date_fruit_list = pd.read_csv("./input/date_action.csv").values.tolist()
#     print(date_fruit_list)
#     return render_template('date_fruit.html', title='食べた果物記録', date_fruit_list=date_fruit_list)
# ## おまじない
# if __name__ == "__main__":
#     app.run(debug=True) 