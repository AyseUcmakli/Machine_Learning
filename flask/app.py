from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def hello_world():
    test_np_input = np.array([[1],[2],[17]])
    model = load('model.joblib')
    preds = model.predict(test_np_input)
    preds_as_str = str(preds)
    return preds_as_str'
    
