
from flask import Flask
from flask import request
from flask import jsonify
import datetime
import hashlib
import numpy as np
import pandas as pd
import aml_v1 as v1

app = Flask(__name__)
####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'my@gmail.com'          #
SALT = 'my_salt'                        #
#########################################


def generate_server_uuid(input_string):
    """ Create your own server_uuid
    @param input_string (str): information to be encoded as server_uuid
    @returns server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string+SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def predict(article):
    """ Predict your model result
    @param article (str): a news article
    @returns prediction (list): a list of name
    """

    # news = "遭控收受政治獻金 未按規定申報〔記者黃捷／台北報導〕中華統一促進黨總裁「白狼」張安樂及其子張瑋，遭控收受政治獻金卻未按規定申報，還涉替張瑋成立的「華夏大地旅行社」逃漏稅，掏空兩千多萬元公司資產，兩人昨遭台北地檢署依違反政治獻金法、幫助逃漏稅及業務侵占等罪起訴。至於華夏有近千萬港幣資金來自境外，張家人卻交代不清，檢調懷疑有中資介入，將是接下來追查重點。除了張氏父子，張瑋妻子王姝茵、統促黨前任黨主席李新一、現任黨主席張馥堂及黨工張陳淑媜，在整個犯罪過程中皆扮演關鍵角色，一併遭起訴。檢調查出，張瑋經營的華夏大地旅行社專接中國旅行團客，不僅涉嫌驗資不實，張家人還將公司資產視為「家族禁臠」，擅自挪為私用，支付妻小貸款、卡費等生活費用。旅行社近千萬港幣資金 來自境外另，張安樂並非華夏員工，張瑋夫妻卻支薪給張安樂並投保勞健保，張家人並以華夏名義租用十輛小發財車供統促黨調度，花費四四七萬餘元，以此手法逃漏稅多達七十七萬餘元，統促黨也未將租車的經濟效益列入政治獻金帳冊。華夏於二○一三至二○一五年間，陸續收到英屬維京群島商「韜略公司Strategic Sports （BVI） LTD」近千萬港幣匯款，名義上是旅行團費，但錢進台灣後卻未如實記載在華夏會計報告，張家人對金錢來源、流向也多推稱「忘記了」。檢調懷疑有中資介入 積極追查中由於張瑋是韜略公司股東，對公司有實質影響力，檢調不排除有匯款來自中國，積極追查中。張安樂等多名統促黨成員昨因另涉違反集遊法而至北檢開庭，庭後竟企圖直闖檢察官辦公室索要起訴書，但遭法警攔阻。張安樂說，檢調辦他就是「雷聲大，雨點小」，抱怨之前遭檢調搜索，好像他犯了什麼滔天大罪，最後罪名都是羅織的。他反問說：「我跟兒子媳婦的錢，我們用來用去，這算挪用公款嗎？」又不像美國人會把家人的錢分得清清楚楚，連租車子給黨部用都違反政治獻金法，直呼「太扯了！」"

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    prediction = v1.query(article)
    # prediction = ['aha','danny','jack']

    ####################################################
    prediction = _check_datatype_to_list(prediction)
    return prediction


def _check_datatype_to_list(prediction):
    """ Check if your prediction is in list type or not. 
        And then convert your prediction to list type or raise error.

    @param prediction (list / numpy array / pandas DataFrame): your prediction
    @returns prediction (list): your prediction in list type
    """
    if isinstance(prediction, np.ndarray):
        _check_datatype_to_list(prediction.tolist())
    elif isinstance(prediction, pd.core.frame.DataFrame):
        _check_datatype_to_list(prediction.values)
    elif isinstance(prediction, list):
        return prediction
    raise ValueError('Prediction is not in list type.')


@app.route('/healthcheck', methods=['POST'])
def healthcheck():
    """ API for health check """
    data = request.get_json(force=True)
    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)
    server_timestamp = t.strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'esun_uuid': data['esun_uuid'], 'server_uuid': server_uuid, 'captain_email': CAPTAIN_EMAIL, 'server_timestamp': server_timestamp})


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API """
    data = request.get_json(force=True)
    esun_timestamp = data['esun_timestamp']  # 自行取用

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL+ts)

    try:
        answer = predict(data['news'])
    except:
        raise ValueError('Model error.')
    server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({'esun_timestamp': data['esun_timestamp'], 'server_uuid': server_uuid, 'answer': answer, 'server_timestamp': server_timestamp, 'esun_uuid': data['esun_uuid']})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
