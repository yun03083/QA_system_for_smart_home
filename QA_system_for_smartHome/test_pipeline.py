
from multilabel_pipeline import MultiLabelPipeline
from transformers import ElectraTokenizer
from model import ElectraForMultiLabelClassification
from pprint import pprint
from glob import glob
import operator

import numpy as np

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-finetuned-goemotions")
# model = ElectraForMultiLabelClassification.from_pretrained("monologg/koelectra-base-finetuned-goemotions")
model = ElectraForMultiLabelClassification.from_pretrained("ckpt/koelectra-base-goemotions-ckpt(TRFAABE_real_last)/checkpoint-10500")

goemotions = MultiLabelPipeline(
  model=model,
  tokenizer=tokenizer,
  threshold=0.3
)

# texts = [
#     "에어컨 트는 시간 오후 4시로",
#     "에어컨 예약시간 4시로",
#     "에어컨 냉방에서 송풍으로 바꿔 줘",
#     "에어컨 제습으로 바꿔 줘",
#     "공기청정기 수면으로 바꿔 줘"
#     "전기는 기간과 날짜로 조회할 수 있다",
#     "등 모드에는 웜톤이 있고 쿨톤이 있고 취침등이 있다",
#     "냉장고 기능에는 온도가 있고 음식이 있고 유통기한이 있다",
#     "공기 청정기 기능에는 전원이 있고 필터가 있고 설정이 있다",
#     "에어컨 모드에는 상하가 있고 약이 있고 무풍이 있다",
#     "로봇 청소기 모드에는 주행모드가 있고 반복청소가 있고 강이 있다",
#     "TV 기능에는 편성표가 있고 예약이 있고 전원이 있다"
# ]
# results, labels, scores = goemotions(texts)
#
#
#
# # pprint(results)
# # print("------------------------------\n")
#
# TDresults = []
# TIresults = []
# for index in range(len(labels)):
#   TD = {'TV': 0, '로봇청소기': 0, '에어컨': 0, '공기청정기': 0, '냉장고': 0, '전구': 0, '전기': 0}
#   TI = {'전원': 0, '예약': 0, '편성표': 0, '주행': 0, '흡입': 0, '충전': 0, '설정': 0,
#         '온도': 0, '바람': 0, '모드': 0, '필터': 0, '공기상태': 0, '세기': 0, '유통기한': 0,
#         '음식': 0, '밝기': 0, '색상': 0, '날짜': 0, '기간': 0, '상태': 0, '전원제어': 0,
#         '예약제어': 0, '모드제어': 0, '세기제어': 0, '온도제어': 0, '청소제어': 0,
#         '바람제어': 0, '밝기제어': 0, '색깔제어': 0, '예외': 0}
#
#   label = labels[index]
#   score = scores[index]
#   for i in range(len(label)):
#     if label[i] in TD:
#       TD[label[i]] = score[i]
#     if label[i] in TI:
#       TI[label[i]] = score[i]
#   TDresults.append(max(TD.items(), key=operator.itemgetter(1))[0])
#   TIresults.append(max(TI.items(), key=operator.itemgetter(1))[0])
#
# # pprint(TDresults)
# # pprint(TIresults)
#
# for k in range(len(texts)):
#   pprint(texts[k] + " -> " + TDresults[k] + ", " + TIresults[k])

lines = []
with open("TRFAABEmultiCheck(C).txt", "r", encoding="utf-8") as f:
    for line in f:
        lines.append(line.strip())

original_labels = []
input_texts = []
for (i, line) in enumerate(lines):
    line = line.strip()
    items = line.split("\t")
    text_a = items[0]
    input_texts.append(text_a)
    label = list(map(int, items[1].split(",")))
    original_labels.append(label)

checkResults, checkLabels, checkScores = goemotions(input_texts)

# with open ("checkResult.txt", 'w')as f:
#     f.write(str(checkResults))
checkTDresults = []
checkTIresults = []
text_length = len(original_labels)
for index in range(text_length):
  TD = {'TV': 0, '로봇청소기': 0, '에어컨': 0, '공기청정기': 0, '냉장고': 0, '전구': 0, '전기': 0}
  TI = {'전원': 0, '예약': 0, '편성표': 0, '주행': 0, '흡입': 0, '충전': 0, '설정': 0,
        '온도': 0, '바람': 0, '모드': 0, '필터': 0, '공기상태': 0, '세기': 0, '유통기한': 0,
        '음식': 0, '밝기': 0, '색상': 0, '날짜': 0, '기간': 0, '상태': 0, '전원제어': 0,
        '예약제어': 0, '모드제어': 0, '세기제어': 0, '온도제어': 0, '청소제어': 0,
        '바람제어': 0, '밝기제어': 0, '색깔제어': 0}
  label = checkLabels[index]
  score = checkScores[index]
  for i in range(len(label)):
    if label[i] in TD:
      TD[label[i]] = score[i]
    if label[i] in TI:
      TI[label[i]] = score[i]
  checkTDresults.append(max(TD.items(), key=operator.itemgetter(1))[0])
  checkTIresults.append(max(TI.items(), key=operator.itemgetter(1))[0])


checkTD = {0: 'TV', 1: '로봇청소기', 2: '에어컨', 3: '공기청정기', 4: '냉장고', 5: '전구', 6: '전기'}
checkTI = {7: '전원', 8: '예약', 9: '편성표', 10: '주행', 11: '흡입', 12: '충전', 13: '설정',
           14: '온도', 15: '바람', 16: '모드', 17: '필터', 18: '공기상태', 19: '세기',
           20: '유통기한', 21: '음식', 22: '밝기', 23: '색상', 24: '날짜', 25: '기간', 26: '상태',
           27 : '전원제어', 28 : '예약제어', 29 : '모드제어', 30 : '세기제어', 31 : '온도제어',
           32 : '청소제어', 33 : '바람제어', 34 : '밝기제어', 35 : '색깔제어'}
count = 0
out_list = []
for k in range(text_length):
  out_list.append("정답 : (" + checkTD[original_labels[k][0]] + ", " + checkTI[original_labels[k][1]] +  ") / 예측 : (" + checkTDresults[k] + ", " + checkTIresults[k]+ ')\n')
  if checkTD[original_labels[k][0]] == checkTDresults[k] and checkTI[original_labels[k][1]] == checkTIresults[k]:
      count = count + 1

with open("TDTIcheck_acc_ogiginal_and_preds(TRFAABE_real_last).txt", "w") as f:
    f.write("acc = " + str(count/text_length * 100) + '\n')
    for i in range(len(out_list)):
        f.write(out_list[i])