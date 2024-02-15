from action_list_app import app
from flask import render_template, request, redirect, url_for
import sqlite3
import csv
DATABASE = 'database.db'

# add
dir_path = '/Users/ken/Desktop/App-Project/action_list_app/input/date_action.csv'
with open(dir_path, 'w') as f: # input/date_action.csv', 'w') as f:
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
    with open(dir_path, 'a') as f: # input/date_action.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows([[date, action, place]])
        # writer.writerow(['a', 'b', 'c'])

    with open(dir_path) as f: # data/temp/sample_writer_row.csv') as f:
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