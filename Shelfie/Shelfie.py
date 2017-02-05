from flask import Flask
from flask import render_template
from flask import request
app = Flask(__name__)


@app.route('/')
def hello_world():
    d = {}
    return render_template("cropper.html")

@app.route("/home",methods=["POST","GET"])
def home():
    print("Home")
    if request.method != "GET":
        vals = request.form["params"]
        x,y,w,h = map(float,vals.split(";"))
        #TODO: add functionality
    else:
        pass
    return render_template("dash.html")

if __name__ == '__main__':
    app.run(debug=True)
