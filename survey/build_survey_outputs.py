# -*- coding: utf-8 -*-
"""이미 추출한 서베이 샘플(survey_sample_key.csv)로 배포용 파일 2종을 만든다.
  1) 원어민검수_설문지.xlsx  — 문항이 첫 시트에 바로 보이는 엑셀
  2) create_survey_form.gs   — 구글 Apps Script: 실행하면 모바일용 구글 폼 + 응답 시트 생성

사용: python build_survey_outputs.py survey_sample_key.csv
샘플·순서는 survey_sample_key.csv를 그대로 따른다(문항 번호 = 정답키의 '문항').
"""
import json, sys
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation

key = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "survey_sample_key.csv").sort_values("문항")
items = [{"n": int(r["문항"]), "p": r["premise"], "h": r["hypothesis"]} for _, r in key.iterrows()]
N = len(items)

GUIDE = [
    "각 문항의 [전제]만 읽었을 때 [가설]이 어떤지 고르고, 두 문장이 얼마나 자연스러운지 표시해 주세요.",
    "· 참 — 전제가 사실이면 가설도 반드시 사실이다",
    "· 거짓 — 전제가 사실이면 가설은 사실일 수 없다",
    "· 알 수 없음 — 가설이 사실일 수도 있지만, 전제만으로는 확정할 수 없다",
    "※ '보통 그렇겠지' 같은 상식적 추측은 하지 말고, 문장에 적힌 내용만으로 판단해 주세요.",
    "· 자연스러움: 1 = 매우 어색함, 3 = 보통, 5 = 매우 자연스러움",
    "· 정답이 정해진 시험이 아닙니다. 직관대로 빠르게 답해 주세요. 앞 문항으로 돌아가 답을 바꾸거나 다른 응답자와 상의하지 말아 주세요.",
]

# ---------------- 1) 엑셀: 첫 시트 = 설문 ----------------
wb = Workbook()
ws = wb.active
ws.title = "설문"
thin = Side(style="thin", color="BBBBBB")
ws["A1"] = f"한국어 문장 판단 설문 ({N}문항, 약 40~50분)"
ws["A1"].font = Font(bold=True, size=13)
for i, t in enumerate(GUIDE, start=2):
    ws.cell(row=i, column=1, value=t)
    ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=6)
ws.merge_cells("A1:F1")
name_row = len(GUIDE) + 3
ws.cell(row=name_row, column=1, value="응답자 이름(또는 코드):").font = Font(bold=True)
ws.merge_cells(start_row=name_row, start_column=1, end_row=name_row, end_column=2)
ws.cell(row=name_row, column=3).fill = PatternFill("solid", fgColor="FFF2CC")
ws.merge_cells(start_row=name_row, start_column=3, end_row=name_row, end_column=4)

HEADER_ROW = name_row + 2
headers = ["문항", "전제", "가설", "판단", "자연스러움(1~5)", "의견(선택)"]
for c, h in enumerate(headers, 1):
    cell = ws.cell(row=HEADER_ROW, column=c, value=h)
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="44546A")
    cell.alignment = Alignment(horizontal="center", vertical="center")
for k, it in enumerate(items):
    r = HEADER_ROW + 1 + k
    for c, v in enumerate([it["n"], it["p"], it["h"], None, None, None], 1):
        cell = ws.cell(row=r, column=c, value=v)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = Border(top=thin, bottom=thin, left=thin, right=thin)
    ws.cell(row=r, column=4).fill = PatternFill("solid", fgColor="FFF2CC")
    ws.cell(row=r, column=5).fill = PatternFill("solid", fgColor="FFF2CC")
last = HEADER_ROW + N
dv1 = DataValidation(type="list", formula1='"참,거짓,알 수 없음"', allow_blank=True,
                     showErrorMessage=True, errorTitle="입력 오류", error="참 / 거짓 / 알 수 없음 중에서 골라 주세요.")
dv2 = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True,
                     showErrorMessage=True, errorTitle="입력 오류", error="1~5 사이의 정수를 입력해 주세요.")
ws.add_data_validation(dv1); ws.add_data_validation(dv2)
dv1.add(f"D{HEADER_ROW + 1}:D{last}"); dv2.add(f"E{HEADER_ROW + 1}:E{last}")
for col, w in zip("ABCDEF", [6, 48, 38, 14, 14, 30]):
    ws.column_dimensions[col].width = w
for i in range(2, len(GUIDE) + 2):
    ws.row_dimensions[i].height = 18
ws.freeze_panes = ws.cell(row=HEADER_ROW + 1, column=2)
wb.save("원어민검수_설문지.xlsx")

# ---------------- 2) 구글 폼 생성용 Apps Script ----------------
PER_PAGE = 10
gs = r"""/**
 * 원어민 검수 설문 — 구글 폼 자동 생성 스크립트
 *
 * 사용법
 *  1. https://script.google.com 접속 → [새 프로젝트]
 *  2. 기본 코드(function myFunction...)를 지우고 이 파일 전체를 붙여넣기 → 저장(Ctrl+S)
 *  3. 위쪽 함수 선택이 createSurveyForm 인지 확인 → [실행]
 *  4. 처음 한 번 권한 요청 → 본인 계정 선택 → [고급] → [(프로젝트 이름)(으)로 이동] → [허용]
 *  5. 1~3분 뒤 아래 [실행 로그]에 '응답자용 링크'가 찍히면 그 링크를 3명에게 보내기
 *     (응답은 함께 만들어지는 구글 시트에 자동으로 쌓입니다)
 */
const FORM_TITLE = '한국어 문장 판단 설문';
const PER_PAGE = __PER_PAGE__;
const ITEMS = __ITEMS__;
const GUIDE = __GUIDE__;
const REMINDER = '참 = 반드시 사실 · 거짓 = 사실일 수 없음 · 알 수 없음 = 전제만으로 확정 불가\n상식적 추측 없이, 문장에 적힌 내용만으로 판단해 주세요.';

function createSurveyForm() {
  const form = FormApp.create(FORM_TITLE);
  form.setDescription(GUIDE.join('\n') + '\n\n총 ' + ITEMS.length + '문항, 약 40~50분 · ' + PER_PAGE + '문항씩 나뉘어 있습니다.');
  form.setProgressBar(true);
  form.setShowLinkToRespondAgain(false);
  form.setAllowResponseEdits(false);
  form.setConfirmationMessage('응답해 주셔서 감사합니다!');

  form.addTextItem().setTitle('응답자 이름(또는 코드)').setRequired(true);

  const pages = Math.ceil(ITEMS.length / PER_PAGE);
  for (let p = 0; p < pages; p++) {
    const chunk = ITEMS.slice(p * PER_PAGE, (p + 1) * PER_PAGE);
    form.addPageBreakItem()
        .setTitle((p + 1) + ' / ' + pages + ' 페이지 (문항 ' + chunk[0].n + '–' + chunk[chunk.length - 1].n + ')')
        .setHelpText(REMINDER);
    chunk.forEach(function (it) {
      form.addMultipleChoiceItem()
          .setTitle('[' + it.n + '] 가설은 전제에 비추어…')
          .setHelpText('전제: ' + it.p + '\n가설: ' + it.h)
          .setChoiceValues(['참', '거짓', '알 수 없음'])
          .setRequired(true);
      form.addScaleItem()
          .setTitle('[' + it.n + '] 두 문장의 자연스러움')
          .setBounds(1, 5)
          .setLabels('매우 어색함', '매우 자연스러움')
          .setRequired(true);
    });
    form.addParagraphTextItem()
        .setTitle('[의견 ' + (p + 1) + '페이지] 어색하거나 판단이 애매했던 문항이 있으면 번호와 함께 적어 주세요 (선택)');
  }

  const sheet = SpreadsheetApp.create(FORM_TITLE + ' (응답)');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, sheet.getId());
  try { form.setPublished(true); } catch (e) { /* 이 기능이 없는 계정이면 무시 */ }

  Logger.log('응답자용 링크 (이걸 보내세요): ' + form.getPublishedUrl());
  Logger.log('폼 편집 링크 (본인만): ' + form.getEditUrl());
  Logger.log('응답 시트 링크: ' + sheet.getUrl());
}
"""
gs = (gs.replace("__PER_PAGE__", str(PER_PAGE))
        .replace("__ITEMS__", json.dumps(items, ensure_ascii=False, indent=1))
        .replace("__GUIDE__", json.dumps(GUIDE, ensure_ascii=False, indent=1)))
open("create_survey_form.gs", "w", encoding="utf-8").write(gs)
print(f"ok: {N} items, header row {HEADER_ROW}, {-(-N // PER_PAGE)} form pages")
