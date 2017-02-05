from flask import Flask
import datetime

import task
import json

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/<date>")
def main(date):
    date = datetime.datetime.strptime(date, "%d%m%Y").date()
    return json.dumps([task.get_misprice(date), task.get_misplace(date)])


app.run(host='0.0.0.0', port=9000)
