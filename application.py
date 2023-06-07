from flask import Flask, jsonify, request
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaTokenizer
from transformers import pipeline

app = Flask(__name__)

# 예제 AI 모델을 불러옵니다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hate_generator = pipeline(model="jason9693/SoongsilBERT-base-beep",device=device)
meaningless_generator = pipeline('moontaijin/petition_classification',device=device)

@app.route('/predict', methods=['POST'])
def predict():
    # POST 방식으로 전송된 데이터를 추출합니다.
    data = request.json
    text = data['text']
    print(type(text))
    
    # AI 모델을 사용하여 혐오 텍스트 데이터를 분류합니다.
    hate_result = hate_generator(text,max_length=128)[0]['label']

    if hate_result in ['hate', 'offensive']:
        response = {'content': str(text), 'result': 'hate'}
    else:
        # AI 모델을 사용하여 청원글 텍스트를 분류합니다.
        meaningless_result = meaningless_generator(text,max_length=128)[0]['label']
    
        if meaningless_result == 0:
            response = {'content': str(text), 'result': 'meaningless'}
        else:
            response = {'content': str(text), 'result': 'petition'}
    
    # 분류 결과를 JSON 형식으로 반환합니다.
    return jsonify(response)

if __name__ == '__main__':
    print('Server Run')
    app.run(debug=True)
