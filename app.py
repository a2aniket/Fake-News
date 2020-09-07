from  flask import Flask,render_template,request
import pandas as pd
import pickle
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from Prediction import check 
import model

app=Flask(__name__)
db=SQLAlchemy(app)

app.config['SQLALCHEMY_DATABASE_URI'] ='sqlite:///DataBase.db' #"mysql://root:@localhost/Check"
#if local_server:
    #app.config['SQLALCHEMY_DATABASE_URI'] =paramas['local_uri']
#else:
    #app.config['SQLALCHEMY_DATABASE_URI'] =paramas['prod_uri']

session=False
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    comm = db.Column(db.String(80),nullable=False)
    date = db.Column(db.String(80),nullable=False)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username=db.Column(db.String(80),nullable=False)
    password=db.Column(db.String(80),nullable=False)

db.create_all()

@app.route("/")
def home():
    news=pd.read_csv("data/train.csv")
    title=news["title"]
    label=news["text"]
    author=news["author"]
    status=news["label"]
    return render_template("Home.html",title=title,label=label,author=author,status=status)

@app.route("/model")
def train_model(session=None,train_status="0"):
    if session:
        status="0"
        df=pd.read_csv("data/train.csv")
        length=df.shape
        pos_sample=df[df["label"]==1]
        neg_smaple=df[df["label"]==0]
        shape={
        'length':length[0],
        'pos_sample':len(pos_sample),
        'neg_smaple':len(neg_smaple)
        }
        comment=Comment.query.filter_by().all()
        model = pickle.load(open('data/model details.pkl', 'rb'))
        return render_template("trainmodel.html",model=model,shape=shape,status=train_status,comment=comment)
    else:
        return render_template("Login.html")

@app.route("/train")
def again_train_model():
    model.train_model()
    return train_model()

@app.route("/p", methods=["POST","GET"])
def predict():
    if (request.method == "POST"):
        news =request.form.get("news")
        prediction=check(news)
        print(prediction[0])
    return render_template("predict.html",prediction=prediction,news=news)

@app.route("/comment",methods=["POST","GET"])
def comment():
    if request.method=="POST":
        comment=request.form.get("comment")
        print(comment)
        comments=Comment(comm=comment,date=datetime.now())
        db.session.add(comments)
        db.session.commit()
    return home()

@app.route("/login",methods=["POST","GET"])
def login():
    if request.method=="POST":
        username=request.form.get("username")
        password=request.form.get("password")
        login=User.query.filter_by(username=username,password=password).all()
        if login:
            session=True
            return train_model(session)
        else:
            session=False
            return train_model(session)

@app.route('/adduser',methods=["POST","GET"])
def add_user():
    if request.method=="POST":
        username=request.form.get("username")
        password=request.form.get("password")
        print(username)
        print(password)
        user=User(username=username,password=password)
        db.session.add(user)
        db.session.commit()
        return train_model()


if __name__ == '__main__':
    app.run(debug=True)
