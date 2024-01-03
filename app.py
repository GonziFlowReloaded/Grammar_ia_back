from flask import Flask, request, jsonify
from gramformer import Gramformer
import torch

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)

gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector


def correct(influent_sentence):
    corrected_sentences = gf.correct(influent_sentence, max_candidates=1)
    cadena_resultado = next(iter(corrected_sentences))
    return cadena_resultado

app = Flask(__name__)

@app.route('/', methods=['GET'])
def api():
    return 'Hello World!'

@app.route('/correction', methods=['POST'])
def correction():
    sentence = request.json['sentence']
    print(sentence)
    return jsonify({'sentence': correct(sentence)})

if __name__ == '__main__':
    app.run(debug=True)


