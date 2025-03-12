from flask import Flask, jsonify
import py_eureka_client.eureka_client as eureka_client

app = Flask(__name__)

# Eureka server config
EUREKA_SERVER = "http://localhost:8500/eureka/"
SERVICE_NAME = "transformer-service"
SERVICE_PORT = 6000
INSTANCE_HOST="127.0.0.1"

# Register with Eureka
eureka_client.init(
    eureka_server=EUREKA_SERVER,
    app_name=SERVICE_NAME,
    instance_port=SERVICE_PORT,
    instance_host=INSTANCE_HOST
)
@app.route("/TransformerTranslate", methods=["POST"])
def test():
    translation = "due to mismanagement , the trained model and datasets have been lost"
    return jsonify({"translation": translation})



# @app.route("/TransformerTranslateLater", methods=["POST"])
# def translate():
#     german_sentence = "aufgrund von misswirtschaft sind das trainierte modell und die datensätze verloren gegangen dies ist der einzige satz , den ich im moment schreiben kann ; mehr wird noch kommen ."
#     src_vocab = {
#         'aufgrund': 0, 'von': 1, 'misswirtschaft': 2, 'sind': 3, 'das': 4,
#         'trainierte': 5, 'modell': 6, 'und': 7, 'die': 8, 'datensätze': 9,
#         'verloren': 10, 'gegangen': 11, '.': 12, 'dies': 13, 'ist': 14,
#         'der': 15, 'einzige': 16, 'satz': 17, ',': 18, 'den': 19,
#         'ich': 20, 'im': 21, 'moment': 22, 'schreiben': 23, 'kann': 24,
#         ';': 25, 'mehr': 26, 'wird': 27, 'noch': 28, 'kommen': 29, 'P': 30
#     }
#     tgt_vocab = {
#         'S': 0, 'due': 1, 'to': 2, 'mismanagement': 3, ',': 4,
#         'the': 5, 'trained': 6, 'model': 7, 'and': 8, 'datasets': 9,
#         'have': 10, 'been': 11, 'lost': 12, '.': 13, 'this': 14,
#         'is': 15, 'only': 16, 'sentence': 17, 'i': 18, 'am': 19,
#         'capable': 20, 'of': 21, 'writing': 22, 'for': 23, 'now': 24,
#         ';': 25, 'more': 26, 'will': 27, 'be': 28, 'coming': 29, 'E': 30
#     }
#     idx2word = {i: w for i, w in enumerate(tgt_vocab)}
#     src_vocab_size = len(src_vocab)  # 31
#     tgt_vocab_size = len(tgt_vocab)  # 31
#     d_model = 512  # Embedding Size
#
#     translation = (
#         transformer.usage(german_sentence, "/content/transformer.model",
#                           src_vocab, tgt_vocab, idx2word,
#                           src_vocab_size, tgt_vocab_size, d_model))
#     return jsonify({"translation": translation})

    # data = request.get_json()
    # translated_text=transformerTranslator.result("german_sentence")
    # print(translated_text)
    # return jsonify({"translation": translated_text})

# Internal function
def process_data():
    return {"message": "Hello from Python service!"}

# API endpoint calling the internal function
@app.route("/process", methods=["GET"])
def process():
    return jsonify(process_data())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=SERVICE_PORT)