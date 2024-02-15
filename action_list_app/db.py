import sqlite3

DATABASE = 'database.db'
def create_info_table():
    con = sqlite3.connect(DATABASE)
    con.execute("CREATE TABLE IF NOT EXISTS info (action, place, date)")
    con.close()