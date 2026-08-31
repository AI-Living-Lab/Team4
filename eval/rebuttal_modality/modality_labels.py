# -*- coding: utf-8 -*-
"""
modality_labels.py — UnAV-100 100개 카테고리의 모달리티 라벨 (3-run).

라벨링 조건 (중요):
  * 카테고리 '이름'만 보고 판정했다. 예측 파일·성능 수치는 이 단계에서 일절 참조하지 않았다.
  * categories_stats.csv 의 통계(mean_N_gt 등)는 판정 근거로 쓰지 않았다.

rubric
  Q1(visual): 음소거 영상만으로 이벤트 시작/끝을 ±1s 로 찍을 수 있나?
  Q2(audio) : 소리만으로 시작/끝을 ±1s 로 찍을 수 있나?
  A  = Q1 no , Q2 yes   (sound-delimited)
  V  = Q1 yes, Q2 no    (vision-delimited)
  AV = Q1 yes, Q2 yes
  H  = Q1 no , Q2 no    (hard/ambiguous)

핵심 판정 원칙: "소리가 나는가"가 아니라 "경계를 무엇이 결정하는가".
  UnAV-100 은 전 카테고리가 소리를 동반하므로 소리 유무는 변별력이 없다.
  발성/기계음처럼 주체가 화면에 계속 보이지만 on/off 가 소리로만 드러나면 A.
  타격·연주·구기처럼 동작 자체가 프레임에서 보이면 AV.

3-run 은 같은 rubric 을 세 번 독립 적용한 결과다. 확실한 항목은 세 번 다 같고,
경계가 애매한 항목(대개 '동작이 보이지만 on/off 는 소리'인 부류)에서 갈렸다.

각 튜플: (run1, run2, run3, offscreen_possible, intermittent_sound,
          continuous_visual, rationale)
"""

LABELS = {
 # ---- 발성/울음소리: 주체는 계속 보이나 발성 on/off 는 소리로만 드러남 -> A
 "baby babbling":      ("A","A","A", False, True,  True,  "아기는 내내 보이고 옹알이의 시작·끝은 소리로만 구분된다."),
 "baby crying":        ("A","A","A", False, True,  True,  "울음 구간 경계는 소리로 정해지고 화면상 아기는 계속 보인다."),
 "baby laughter":      ("A","A","A", False, True,  True,  "웃음의 on/off 가 소리로만 끊긴다."),
 "kid speaking":       ("A","A","A", False, True,  True,  "발화 구간은 소리로만 경계가 잡힌다."),
 "man speaking":       ("A","A","A", False, True,  True,  "입 움직임으로 ±1s 를 찍기 어렵고 발화 경계는 음성이 정한다."),
 "woman speaking":     ("A","A","A", False, True,  True,  "발화 경계는 음성이 정한다."),
 "people whispering":  ("A","A","A", False, True,  True,  "속삭임은 시각적으로 거의 드러나지 않는다."),
 "people shouting":    ("A","A","A", True,  True,  True,  "외침의 시작·끝은 소리로만 명확하다."),
 "people battle cry":  ("A","A","A", False, True,  True,  "함성 구간 경계가 소리로 정해진다."),
 "people sobbing":     ("A","A","A", False, True,  True,  "흐느낌의 경계는 소리 쪽이 뚜렷하다."),
 "people coughing":    ("A","A","A", False, True,  True,  "기침은 짧고 시각 신호가 약해 소리로 끊긴다."),
 "people burping":     ("A","A","A", False, True,  True,  "트림은 화면상 거의 보이지 않는다."),
 "people slurping":    ("A","A","A", False, True,  True,  "후루룩 소리 외에 시각 단서가 약하다."),
 "people whistling":   ("A","A","A", True,  True,  True,  "휘파람은 입모양만으로 경계를 못 잡는다."),
 "people cheering":    ("A","A","AV",True,  True,  True,  "환호는 소리가 주 신호지만 군중 동작이 같이 보이기도 한다."),
 "people crowd":       ("A","A","H", True,  False, True,  "군중은 내내 화면에 있고 '웅성거림' 경계 자체가 모호하다."),
 "people laughing":    ("AV","A","A",False, True,  True,  "웃는 표정이 보이긴 하나 ±1s 경계는 소리가 정한다."),
 "child singing":      ("A","A","A", False, False, True,  "노래 구간의 시작·끝은 소리로만 결정된다."),
 "female singing":     ("A","A","A", False, False, True,  "가창 경계는 소리가 정한다."),
 "male singing":       ("A","A","A", False, False, True,  "가창 경계는 소리가 정한다."),
 "beat boxing":        ("A","A","A", False, True,  True,  "연행자는 계속 보이고 비트의 on/off 는 소리뿐이다."),

 # ---- 동물: 주체가 화면 밖일 수 있고 울음 경계는 소리 -> A
 "bird chirping":      ("A","A","A", True,  True,  False, "새는 작거나 화면 밖이고 지저귐 경계는 소리다."),
 "cat meowing":        ("A","A","A", True,  True,  True,  "울음의 시작·끝이 소리로만 구분된다."),
 "dog barking":        ("A","A","A", True,  True,  True,  "짖는 구간은 소리로 끊기고 개는 화면 밖일 수 있다."),
 "dog howling":        ("A","A","A", True,  True,  True,  "하울링 경계는 소리가 정한다."),
 "bull bellowing":     ("A","A","A", True,  True,  True,  "울음소리로만 경계가 잡힌다."),
 "lions roaring":      ("A","A","A", True,  True,  True,  "포효 구간은 소리로 구분된다."),
 "frog croaking":      ("A","A","A", True,  True,  False, "개구리는 대개 화면에 없고 울음만 들린다."),
 "sheep bleating":     ("A","A","A", True,  True,  True,  "울음 경계는 소리 쪽이다."),
 "horse clip-clop":    ("AV","AV","A",True, True,  True,  "말의 보행이 보이지만 발굽 소리가 더 정확한 경계를 준다."),

 # ---- 차량/사이렌/경적: 음원이 화면 밖인 경우가 많음 -> A
 "airplane flyby":     ("A","A","AV",True, False, False, "비행기는 작거나 화면 밖이고 통과음의 고조·감쇠가 경계다."),
 "helicopter":         ("A","A","AV",True, False, False, "로터음이 주 신호이고 기체는 화면 밖일 수 있다."),
 "ambulance siren":    ("A","A","A", True,  False, False, "사이렌 자체가 이벤트이고 차량은 대개 화면 밖이다."),
 "fire truck siren":   ("A","A","A", True,  False, False, "사이렌 소리로만 구간이 정해진다."),
 "police car siren":   ("A","A","A", True,  False, False, "사이렌 소리로만 구간이 정해진다."),
 "vehicle honking":    ("A","A","A", True,  True,  False, "경적은 순간음이고 차량이 안 보여도 성립한다."),
 "train horning":      ("A","A","A", True,  True,  False, "기적 소리가 이벤트 경계를 정한다."),
 "train wheels squealing": ("A","A","A", True, True, False, "쇳소리로만 구간이 잡힌다."),
 "car passing by":     ("A","AV","A", True, False, False, "통과음의 도플러가 경계를 주지만 차가 프레임을 가로지르기도 한다."),
 "skidding":           ("A","A","AV", True, False, False, "타이어 마찰음이 주 신호이고 차량은 안 보일 수 있다."),
 "engine knocking":    ("A","A","A", True,  True,  False, "엔진 노킹은 보닛/실내 화면으로는 경계를 못 잡는다."),

 # ---- 기계/환경음: 시각적으로 on/off 가 안 보임 -> A
 "hair dryer drying":  ("A","AV","A", False, False, True, "드라이어는 계속 보이지만 켜짐/꺼짐은 소리로만 안다."),
 "telephone bell ringing": ("A","A","A", True, True, False, "전화기가 화면 밖이어도 벨소리로 구간이 정해진다."),
 "church bell ringing":("A","A","A", True,  True,  False, "종탑은 대개 화면 밖이고 타종음이 경계다."),
 "thunder":            ("A","A","AV", True, True,  False, "천둥소리가 이벤트이고 번개 섬광이 같이 보이기도 한다."),
 "wind noise":         ("A","A","A", True,  False, False, "바람소리는 시각적으로 드러나지 않는다."),
 "water burbling":     ("A","A","A", True,  False, False, "물소리로만 구간이 잡힌다."),
 "raining":            ("AV","H","A", True, False, True,  "빗줄기가 보이기도 하나 구간 경계가 영상 전체와 뒤섞여 애매하다."),
 "sea waves":          ("AV","H","H", False, False, True, "파도는 내내 보이고 들려서 구간 경계 자체가 정의되기 어렵다."),

 # ---- 동작이 프레임에서 보이는 부류 -> AV
 "people clapping":    ("AV","AV","AV", False, True, True, "손뼉 동작과 박수 소리가 같은 시점에 뚜렷하다."),
 "people slapping":    ("AV","AV","AV", False, True, True, "때리는 동작과 타격음이 동시에 잡힌다."),
 "people sneezing":    ("AV","A","AV", False, True, True,  "재채기 동작이 크게 보이지만 순간음이 더 정확하다."),
 "people nose blowing":("AV","AV","A", False, True, True,  "코를 푸는 동작이 보이나 경계는 소리가 더 뚜렷하다."),
 "people eating":      ("AV","AV","AV", False, True, True, "먹는 동작이 지속적으로 보이고 저작음도 들린다."),
 "people running":     ("AV","V","AV", False, True, True,  "달리는 동작이 명확하고 발소리도 동반된다."),
 "rope skipping":      ("AV","V","AV", False, True, True,  "줄넘기 동작이 시각적으로 매우 뚜렷하다."),
 "skateboarding":      ("AV","AV","AV", False, True, True, "보드 주행이 보이고 바퀴/착지음이 동반된다."),
 "tap dancing":        ("AV","AV","AV", False, True, True, "발동작과 탭음이 동기화된다."),
 "hammering nails":    ("AV","AV","AV", False, True, True, "망치질 동작과 타격음이 동시에 보인다."),
 "chainsawing trees":  ("AV","AV","A", False, True, True,  "톱질 동작이 보이나 엔진음 on/off 가 더 정확하다."),
 "lawn mowing":        ("AV","A","AV", True, False, True,  "잔디깎이 이동이 보이지만 엔진 가동은 소리로 안다."),
 "vacuum cleaner cleaning floors": ("AV","A","AV", False, False, True, "청소기 이동이 보이나 가동 여부는 소리다."),
 "typing on computer keyboard": ("AV","AV","AV", False, True, True, "손가락 타이핑과 키음이 함께 잡힌다."),
 "striking bowling":   ("AV","AV","AV", False, True, True, "투구 동작과 핀 타격음이 모두 뚜렷하다."),
 "basketball bounce":  ("AV","AV","AV", False, True, True, "드리블 동작과 바운스음이 동기화된다."),
 "machine gun shooting": ("AV","A","AV", True, True, True, "총구 화염이 보이기도 하나 연사음이 더 확실하다."),
 "fireworks banging":  ("AV","AV","A", True, True, False,  "섬광이 보이지만 폭음이 경계를 더 잘 준다."),
 "auto racing":        ("AV","AV","AV", True, True, True,  "차량 주행이 보이고 엔진음도 계속 들린다."),
 "driving buses":      ("AV","AV","V", True, False, True,  "버스 주행이 시각적으로 뚜렷하다."),
 "driving motorcycle": ("AV","AV","AV", True, False, True, "주행 장면과 엔진음이 함께 잡힌다."),
 "sailing":            ("V","V","AV", False, False, True,  "배의 항행은 보이지만 소리는 바람·물결이라 경계를 못 준다."),
 "people whistling ":  ("A","A","A", True, True, True,     "(중복 방지용 미사용 키)"),

 # ---- 구기/스포츠: 랠리 동작이 보임 -> AV
 "playing badminton":  ("AV","AV","AV", False, True, True, "랠리 동작과 셔틀 타격음이 함께 보인다."),
 "playing table tennis":("AV","AV","AV", False, True, True,"랠리 동작과 공 타격음이 동기화된다."),
 "playing tennis":     ("AV","AV","AV", False, True, True, "스윙 동작과 타구음이 함께 잡힌다."),
 "playing volleyball": ("AV","V","AV", False, True, True,  "경기 동작이 뚜렷하고 타구음도 들린다."),

 # ---- 악기 연주: 연주 동작이 프레임에 보임 -> AV
 "playing accordion":  ("AV","AV","AV", False, False, True, "주름상자 동작과 연주음이 함께 잡힌다."),
 "playing acoustic guitar": ("AV","AV","AV", False, False, True, "스트로크 동작과 소리가 동기화된다."),
 "playing banjo":      ("AV","AV","AV", False, False, True, "연주 동작이 보이고 소리도 뚜렷하다."),
 "playing base guitar":("AV","AV","AV", False, False, True, "연주 동작과 소리가 함께 잡힌다."),
 "playing cello":      ("AV","AV","AV", False, False, True, "보잉 동작이 소리와 동기화된다."),
 "playing clarinet":   ("AV","AV","A", False, False, True,  "운지가 보이나 발음 시점은 소리가 더 정확하다."),
 "playing cornet":     ("AV","AV","A", False, False, True,  "취주 동작만으로는 발음 시점을 못 찍는다."),
 "playing drum kit":   ("AV","AV","AV", False, True, True,  "타격 동작과 드럼음이 동시에 잡힌다."),
 "playing electronic organ": ("AV","AV","A", False, False, True, "건반 동작이 보이나 음 출력은 소리로만 안다."),
 "playing erhu":       ("AV","AV","AV", False, False, True, "보잉 동작과 소리가 동기화된다."),
 "playing flute":      ("AV","AV","A", False, False, True,  "취주 자세만으로는 발음 경계를 못 찍는다."),
 "playing harmonica":  ("AV","AV","A", False, False, True,  "입에 대는 동작과 실제 발음 시점이 어긋난다."),
 "playing harp":       ("AV","AV","AV", False, False, True, "현을 뜯는 동작이 소리와 함께 보인다."),
 "playing piano":      ("AV","AV","AV", False, False, True, "건반 타건 동작과 소리가 동기화된다."),
 "playing saxophone":  ("AV","AV","A", False, False, True,  "취주 자세보다 소리가 정확한 경계를 준다."),
 "playing snare drum": ("AV","AV","AV", False, True, True,  "타격 동작과 타격음이 동시에 잡힌다."),
 "playing synthesizer":("AV","AV","A", False, False, True,  "건반 동작이 보이나 출력음은 소리로만 안다."),
 "playing tabla":      ("AV","AV","AV", False, True, True,  "손 타격 동작과 소리가 동기화된다."),
 "playing trombone":   ("AV","AV","AV", False, False, True, "슬라이드 동작이 소리와 함께 보인다."),
 "playing trumpet":    ("AV","AV","A", False, False, True,  "취주 자세만으로 발음 시점을 못 찍는다."),
 "playing ukulele":    ("AV","AV","AV", False, False, True, "스트로크 동작과 소리가 동기화된다."),
 "playing violin":     ("AV","AV","AV", False, False, True, "보잉 동작이 소리와 함께 보인다."),
 "playing zither":     ("AV","AV","AV", False, False, True, "현을 뜯는 동작이 보인다."),
 "orchestra":          ("AV","AV","A", False, False, True,  "연주 동작이 보이나 합주 시작·끝은 소리가 정한다."),
 "singing choir":      ("AV","A","AV", False, False, True,  "합창 동작이 보이나 발성 경계는 소리다."),
}
LABELS.pop("people whistling ", None)   # 자리표시용 키 제거
