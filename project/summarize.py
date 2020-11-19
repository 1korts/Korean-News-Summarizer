import torch
from utils.tokenization_kobert import KoBertTokenizer
from utils.utils import load_text
from translate.predictor import build_predictor
from abs_summarizer.model import AbsSummarizer

params = {
    "bert_fine_tune": True,
    "use_bert_emb": True,
    "dec_layers": 6,
    "dec_hidden_size": 768,
    "dec_ff_size": 2048,
    "dec_dropout": 0.1,
    "label_smoothing": 0.1,
    "generator_shard_size": 5,
    "dec_heads": 12
}

class AbstractSummarizer(object):
    def __init__(self, params, model_path='storage/ext_trained_model/model.pt'):
        checkpoint = torch.load(
            model_path, map_location=torch.device("cpu"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AbsSummarizer(params, device, checkpoint)
        self.model.eval()

        tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
        self.predictor = build_predictor(tokenizer, self.model)

    def get_summary(self, texts):
        text_iter = load_text(texts)
        return self.predictor.translate(text_iter)


if __name__ == '__main__':
    summarizer = AbstractSummarizer(params)
    print(summarizer.get_summary([
        """한동식 기자

구, 수흐바토르구 국제교류단 초청 역사도보관광 등 체험 프로 진행
인천시 중구가 몽골 울란바토르시 수흐바토르구와 청소년 국제교류를 적극적으로 추진하고 있다.

구는 지난 19일부터 23일까지 자매우호도시인 몽골 울란바토르시 수흐바토르구 청소년 국제교류단을 초청해 청소년 국제교류 프로그램을 진행했다.

이번 교류는 양 도시가 서로 방문해 청소년 국제교류 프로그램을 진행하는 방식으로 추진됐다.

앞서 구는 지난 8월 13일부터 17일까지 중구지역 내 중·고등학생 청소년 19명이 몽골 울란바토르시 수흐바토르구를 방문해 교류 프로그램을 실시한 바 있다.
이번에는 몽골 수흐바토르구 중·고등학생 청소년 20명이 중구를 방문해 청소년 국제교류 프로그램을 진행했다.

이번 교류는 19일 홍인성 중구청장과의 공식 접견을 시작으로 20일 월미도 월미공원 한복체험 및 한식만들기체험, 차이나타운, 개항장거리 역사도보관광 등으로 중구의 역사와 문화에 대해 알아보는 시간을 가졌다.
또 영종국제도시 레일바이크 체험과 파라다이스시티호텔 수영장과 놀이공원 체험 등 한국과 인천의 문화에 대해 이해하는 시간과 두 도시 청소년 간 선물 전달식 등 우정을 나누는 공식 교류행사 등으로 진행됐다.

홍인성 중구청장은 "이번 교류가 양 도시 청소년들의 공감대 형성과 글로벌 마인드를 기르는 등 청소년 여러분의 성장에 밑거름이 되길 바란다"며 "이를 바탕으로 양 도시의 우호관계 증진에도 크게 도움이 될 것"이라고 강조했다.

한동식 기자 dshan@kihoilbo.co.kr"""
    ]))