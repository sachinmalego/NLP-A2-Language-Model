from flask import Flask, render_template, request
from lstm import LSTMLanguageModel, generate
from torchtext.data.utils import get_tokenizer
import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import pickle

app = Flask(__name__)

#if you are using cuda use this
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#if you are using mac mps use this
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

vocab = torch.load('pickle/vocab')
tokenizer = get_tokenizer('basic_english')

vocab_size = len(vocab)
emb_dim = 1024                # 400 in the paper
hid_dim = 1024                # 1150 in the paper
num_layers = 2                # 3 in the paper
dropout_rate = 0.65              

model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
model.load_state_dict(torch.load('pickle/best-val-lstm_lm.pt',  map_location=device))
model.eval()

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        # Hyperparameters for model inference
        prompt = request.form['prompt']
        max_seq_len = int(request.form['max_seq_len'])
        temperature = float(request.form['temperature'])
        seed = int(request.form['seed'])

        output = generate(prompt, max_seq_len, temperature, model, tokenizer,
                            vocab, device, seed)
        return render_template('home.html', output=' '.join(output), show_text="block")

    else:
        return render_template('home.html', show_text="none")

if __name__ == '__main__':
    app.run(debug=True)