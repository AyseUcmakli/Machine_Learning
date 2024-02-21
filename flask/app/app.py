from flask import Flask, render_template, request
import numpy as np
from joblib import dump, load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

# Flask uygulamasını oluşturur. __name__, uygulamanın nerede çalıştığını belirtir ve genellikle __main__ olarak ayarlanır.
app = Flask(__name__)

#'/' yolu için bir endpoint (nokta) belirler. Bu endpoint hem GET hem de POST HTTP metodlarını kabul eder.
#Bu, kullanıcının tarayıcı üzerinden sayfayı görüntüleyebilmesi (GET) ve form gönderdiğinde bu formu 
#işleyebilmesi (POST) için tasarlanmış bir endpoint'tir.
@app.route('/', methods = ['GET', 'POST'])

def hello_world():#Endpoint'in fonksiyonu başlar.
    request_type_str = request.method# İstek türünü belirlemek için request objesinden method özelliğini kullanır. 
                                     # Bu, kullanıcının sayfayı görüntüleyip (GET) veya bir form gönderip (POST) göndermediğini kontrol eder.
    if request_type_str == 'GET':# Eğer istek GET ise, sayfayı görüntüleme işlemi gerçekleştirilir.
        return render_template('index.html', href='static/base_pic.svg')#bir HTML sayfasını tarayıcıya gönderir.sayfanın içinde bir SVG resminin yolu bulunabilir.
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex#Rastgele bir dize oluşturur. Bu genellikle dosya adlarına benzersizlik kazandırmak için kullanılır.
        path = "static/" + random_string + ".svg"#Rastgele dizeyi kullanarak her form gönderiminde farklı bir dosya adı elde edilir.
        model = load('model2.joblib')#Önceden eğitilmiş bir modeli yükler ve okur
        np_arr = floats_string_to_np_arr(text)# Formdan alınan metni sayısal bir NumPy dizisine dönüştürür.
        make_picture('AgesAndHeights.pkl', model, np_arr, path)# Modeli ve kullanıcının girdiğini içeren NumPy dizisini kullanarak bir resim oluşturan fonksiyonu çağırır.
        return render_template('index.html', href=path)#Oluşturulan resmin bulunduğu yol ile birlikte, tekrar render_template fonksiyonu
                                                       # aracılığıyla index.html şablonunu kullanarak bir HTML sayfasını tarayıcıya gönderir. 
                                                       # Bu sefer resim içeriğiyle birlikte gönderilir.


def make_picture(training_data_filename, model, new_input_np_arr, output_file):
  data = pd.read_pickle(training_data_filename)
  ages = data['Age']
  data = data[ages > 0]
  ages = data['Age']
  heights = data['Height']
  x_new = np.array(list(range(19))).reshape(19, 1)
  preds = model.predict(x_new)

  fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", labels={'x': 'Age (Years)',
                                                                                'y': 'Height (Inches)'})

  fig.add_trace(go.Scatter(x=x_new.reshape(x_new.shape[0]), y=preds, mode='lines', name='Model'))

  new_preds = model.predict(new_inp_np_arr)

  fig.add_trace(go.Scatter(x=new_input_np_arr.reshape(len(new_input_np_arr)),
                                                     y=new_preds, name='New Outputs', mode='markers', 
                                                     marker=dict(color='purple', size=20, 
                                                     line=dict(color='purple', width=2))))
  
  fig.write_image(output_file, width=800, engine='kaleido')
  fig.show()

def floats_string_to_np_arr(floats_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False
  floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
  return floats.reshape(len(floats), 1)
