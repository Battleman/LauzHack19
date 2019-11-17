from flask import Flask, escape, request, render_template, redirect, url_for, make_response
from werkzeug import secure_filename
import numpy as np
from darknet import Darknet
from predictor import detect_image

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    if request.method == 'POST':
        print(request.files)
        f = request.files['file']
        i = int(request.cookies.get("i", 1))
        fname = "submitted_images/" + secure_filename(f.filename)
        f.save(fname)
        wtype = get_waste_type(fname, i)
    r = make_response(redirect(url_for(".landing", wtype=wtype)))
    r.set_cookie(b'i', "{}".format(i+1))
    return r

@app.route("/result")
def landing():
    wtype = request.args["wtype"]
    print(wtype)
    resources = {
        "Aluminium":{
            "picto": "/static/images/picto/alu.png",
            "bins": ["/static/images/bins/poubelle_alu1.jpg", "/static/images/bins/poubelle_alu2.jpg"],
            "name": "Aluminium"
        },
        "Carton":{
            "picto": "/static/images/picto/carton.png",
            "bins": ["/static/images/bins/poubelle_papier.jpg", "/static/images/bins/poubelle_papier2.jpg"],
            "name": "Cardboard"
        },
        "PET": {
            "picto": "/static/images/picto/pet.jpg",
            "bins": ["/static/images/bins/poubelle_pet.jpeg", "/static/images/bins/poubelle_pet2.png"],
            "name": "PET"
        },
        "Dechet": {
            "picto": "/static/images/picto/dechet.png",
            "bins": ["/static/images/bins/poubelle_general.jpg"],
            "name": "General Trash"
        },
        "Verre_Brun": {
            "picto": "/static/images/picto/verre.png",
            "bins": ["/static/images/bins/poubelle_verre.jpg"],
            "name": "Brown Glass"
        },
        "Verre_Blanc": {
            "picto": "/static/images/picto/verre.png",
            "bins": ["/static/images/bins/poubelle_verre.jpg"],
            "name": "White Glass"
        },
        "Verre_Reste": {
            "picto": "/static/images/picto/verre.png",
            "bins": ["/static/images/bins/poubelle_verre.jpg"],
            "name": "Green or special Glass"
        },
        "Papier": {
            "picto": "/static/images/picto/papier.png",
            "bins": ["/static/images/bins/poubelle_papier.jpg", "/static/images/bins/poubelle_papier2.jpg"],
            "name": "Paper"
        }
    }
    return render_template("profile-page.html", res=resources[wtype], country="Switzerland")

def get_waste_type(fname, i):

    print('Loading network...')
    model = Darknet("darknet/cfg/yolov3-tiny.cfg")
    model.load_weights('darknet/backup/yolov3-tiny_900.weights')

    model.eval()
    print('Network loaded')
    t = detect_image(model, fname, i)
    i+=1
    return t
