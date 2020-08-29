from  flask import Flask,render_template
import pandas as pd
from time import time
app=Flask(__name__)


@app.route("/")
def home():
    news=pd.read_csv("data/train.csv")
    title=news["title"]
    label=news["text"]
    author=news["author"]
    status=news["label"]
    return render_template("Home.html",title=title,label=label,author=author,status=status)

@app.route("/model")
def train_model():
    status="1"
    df=pd.read_csv("data/train.csv")
    length=df.shape
    pos_sample=df[df["label"]==1]
    neg_smaple=df[df["label"]==0]
    train_score=0
    test_score=0
    total_score=0
    status=input("Enter Sta")
    return render_template("trainmodel.html",length=length[0],pos_sample=len(pos_sample),neg_smaple=len(neg_smaple),status=status)

@app.route("/p", methods=["POST"])
def predict():
    return render_template("predict.html")

app.run(debug=True)