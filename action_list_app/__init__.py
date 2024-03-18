from flask import Flask
app = Flask(__name__)

import action_list_app.resister_main

from action_list_app import db
db.create_info_table()