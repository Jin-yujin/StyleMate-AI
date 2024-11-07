from flask import Flask, render_template, request, jsonify
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 파일 업로드 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 최대 16MB 파일 크기 제한

load_dotenv()  # .env 파일 로드 

# OpenAI 클라이언트 설정
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # 환경변수에서 API 키 가져오기

# 업로드 폴더 생성 
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# CustomModel 정의
class CustomModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomModel, self).__init__()
        
        # Pretrained ResNet18 모델 로드 및 수정
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # ResNet18의 fully connected layer를 Identity로 변경
        
        # 추가 입력 데이터의 피처를 결합하기 위한 밀집층
        self.fc_input = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU()
        )
        
        # 이미지 특징과 추가 입력 데이터를 결합한 후 최종 예측을 위한 레이어
        self.fc_combined = nn.Sequential(
            nn.Linear(num_ftrs + 32, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)  # 아웃풋 데이터의 개수에 맞춰서 출력층 크기 설정
        )
    
    def forward(self, image, additional_input):
        # 이미지 데이터에서 특징 추출
        image_features = self.resnet(image)
        
        # 추가 입력 데이터 처리
        additional_features = self.fc_input(additional_input)
        
        # 이미지 특징과 추가 입력 데이터를 결합
        combined = torch.cat((image_features, additional_features), dim=1)
        
        # 최종 예측
        output = self.fc_combined(combined)
        return output

# 모델 초기화 및 가중치 로드
input_size = 4       # 추가 인풋 데이터의 크기 (키, 몸무게, 성별_F, 성별_M)
output_size = 10     # 예측하려는 출력의 크기
model = CustomModel(input_size=input_size, output_size=output_size)

# 모델 가중치 파일 경로 설정
MODEL_PATH = './final_saved_model.pth'
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()  # 평가 모드로 설정

# 이미지 전처리 함수 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 파일 확장자 검증 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 예측 결과 해석 함수
def interpret_prediction(prediction):
    # prediction 리스트의 각 특징을 해석해서 문자열로 반환
    features = [
        f"어깨너비: {prediction[0]:.1f}cm",
        f"소매길이: {prediction[1]:.1f}cm",
        f"가슴단면: {prediction[2]:.1f}cm",
        f"상의총장: {prediction[3]:.1f}cm",
        f"허리단면: {prediction[4]:.1f}cm",
        f"엉덩이단면: {prediction[5]:.1f}cm",
        f"허벅지단면: {prediction[6]:.1f}cm",
        f"밑단단면: {prediction[7]:.1f}cm",
        f"하의총장: {prediction[8]:.1f}cm",
        f"밑위: {prediction[9]:.1f}cm"
    ]
    return ", ".join(features)

# 예측 함수 정의
def predict(image_path, additional_input):
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    
    # 추가 입력 데이터 (키, 몸무게, 성별_F, 성별_M)를 텐서로 변환
    additional_input = torch.tensor(additional_input, dtype=torch.float32).unsqueeze(0)
    
    # 예측 수행
    with torch.no_grad():
        output = model(image, additional_input)
    
    return output.squeeze().tolist()  # 예측 결과를 리스트로 반환

# Flask 라우트 설정
@app.route('/')
def index():
    return render_template('index.html')  # index.html 파일 필요

@app.route('/recommend', methods=['POST'])
@app.route('/recommend', methods=['POST'])
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        age = int(request.form.get('age'))
        gender = request.form.get('gender')

        if gender not in ['남성', '여성']:
            return jsonify({'error': '성별을 올바르게 선택해주세요.'}), 400

        # 성별을 성별_F, 성별_M 형식으로 변환
        gender_F, gender_M = (1, 0) if gender == '여성' else (0, 1)
        additional_input = [height, weight, gender_F, gender_M]

        # 이미지 처리
        image_path = None
        prediction = None
        if 'photo' in request.files:
            file = request.files['photo']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_path = filepath

                # PyTorch 모델을 사용한 예측 수행
                prediction = predict(image_path, additional_input)

        # OpenAI API 호출 (이미지 특징과 함께)
        messages = [
            {"role": "system", "content": "당신은 체형 분석과 의상 추천을 해주는 스타일 전문가입니다."},
            {
                "role": "user",
                "content": f"키: {height}cm, 몸무게: {weight}kg, 나이: {age}세, 성별: {gender}인 사람에게 어울리는 옷을 추천해주세요. 연령대에 맞는 스타일도 고려해서 추천해주세요."
            }
        ]

        if prediction:
            # 예측된 특징을 해석하여 메시지에 추가
            prediction_text = interpret_prediction(prediction)
            messages.append({
                "role": "user",
                "content": (
                    f"고객님의 체형 정보는 다음과 같습니다: {prediction_text}. "
                    "이 체형 정보를 답변에 명시적으로 포함하고, 각 수치를 활용하여 맞춤형 추천을 주세요. "
                    "예를 들어, 어깨너비가 {prediction[0]:.1f}cm이므로 이에 어울리는 상의, "
                    "허벅지 단면이 {prediction[6]:.1f}cm이므로 이에 어울리는 하의를 추천하는 방식으로 답변을 작성해주세요."
                )
            })

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        recommendation = response.choices[0].message.content
        return jsonify({'recommendation': recommendation, 'prediction': prediction})
        
    except Exception as e:
        return jsonify({'error': f'서버 오류가 발생했습니다: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
