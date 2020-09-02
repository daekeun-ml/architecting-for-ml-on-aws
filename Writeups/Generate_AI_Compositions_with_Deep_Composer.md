# Architecting for ML - DeepComposer Module

## Introduction

여러분은 우주 탐사 및 식민지 개척에 관한 신규 영화의 사운드 트랙을 구성하라는 Amazon 스튜디오(가상의 회사)의 콘텐츠 요청에 대응해야 합니다. Amazon 스튜디오는 바로크 음악 문화 테마와 우주 여행에 관련된 첨단 과학 테마를 모두 반영하는 새로운 사운드를 원합니다. 그들은 최근 머신 러닝 Generative AI를 사용하여 새로운 음악을 생성하는 AWS DeepComposer에 대해 들었습니다. 그들은 바흐와 같은 음악 코드를 포함하면서 우아롭고 인상적인 1분 길이의 테스트 트랙을 구성하고 싶어합니다. 그들이 당신이 DeepComposer로 작곡한 트랙을 좋아한다면, 당신은 영화 음악 작곡가로 고용될 것입니다.

## 모든 개발자와 데이터 과학자를 위한 머신 러닝

산업 전반의 조직들은 머신 러닝을 사용하여 예측 분석, 예측(Forecasting), 비즈니스 최적화, 로봇 공학, 산업 프로세스 자동화, 설계 효율성 등 다양한 사용 사례들을 해결하고 있습니다. AWS는 모든 개발자와 데이터 과학자를 머신 러닝을 지향하고 있습니다. 이를 위해 Object Detection 사용 사례를 위한 **AWS DeepLens**, 강화 학습을 재미있게 배우기 위한  **AWS DeepRacer**를 출시했으며 이제 **AWS DeepComposer**를 통해 신규 고객에게 몰입감 있고 재미있는 플랫폼을 제공하고 교육하고자 합니9다. 이 서비스는 **일련의 입력 데이터를 기반으로 창의적인 새로운 데이터를 생성**할 수 있는 Generative Adversarial Networks(**GAN**)이라고 하는 머신 러닝 분야의 어플리케이션입니다. GAN은 학술 분야 뿐만 아니라 다양한 산업에서 이미 인기가 있으며 그 중 한 예시로는 Autodesk에서 GAN을 사용하여 고효율 비행기 날개를 설계하고 있습니다.

## Mission

### Objective
팀의 목표는 AWS Amazon DeepComposer를 활용하여 음악을 작곡하는 것입니다.

### 팀원 및 추천 서비스
- 머신 러닝이나 음악에 대한 배경 지식이 필요 없습니다. 만약 데이터 과학자나 머신 러닝 엔지니어가 팀에 소속되어 있다면, 커스텀 모델을 훈련해 보세요.
- 추천 서비스
    - Amazon DeepComposer

## Generative AI with AWS DeepComposer

최근 10년 동안 가장 흥미로운 머신 러닝 아이디어로 간주되는 Generative AI는 컴퓨터가 주어진 문제의 기본 패턴을 학습하고 이 지식을 사용하여 입력 (이미지, 음악, 텍스트 등)에서 새로운 콘텐츠를 생성할 수 있도록 합니다. 예를 들어 고양이와 개의 이미지를 구별하는 특성을 식별하여 구별하는 방법을 배우는 일반적으로 사용되는 머신 러닝 모델과 달리, 고양이 이미지를 기반으로 하는 Generative AI 모델은 고양이에서 공통적인 특징을 학습합니다. 그 지식을 사용하여 고양이라고 생각할 수 있는 완전히 새로운 이미지를 생성합니다. Generative AI 알고리즘의 발전으로 기계는 데이터의 패턴을 자동으로 발견하고 훈련된 데이터를 기반으로 새로운 데이터를 생성할 수 있게 되었습니다.

여러분은 머신 러닝(ML) 또는 음악에 대한 경험 없이도 AWS DeepComposer를 사용하여 Generative AI를 개발할 수 있습니다. AWS DeepComposer에는 Generative AI 모델을 이해하고 사용하는 데 도움이 되는 학습 캡슐, 샘플 코드 및 훈련 데이터가 포함되어 있습니다.

## The Challenge

챌린지를 완료하려면 다음 두 단계를 완료해야 합니다.

1. 모델이 단일 트랙 입력 멜로디에 반주를 추가하는 방법을 훈련할 수 있도록 GAN(Generative Adversarial Network)을 사용하여 바흐 작곡 데이터셋을 사용하여 ML 모델을 훈련시킵니다. 즉, 사용자가 "반짝 반짝 작은 별"과 같은 노래의 단일 피아노 트랙을 제공하는 경우 GAN 모델은 음악 사운드가 바흐에서 영감을 더 많이 받도록 3개의 다른 피아노 트랙을 추가합니다.

2. AWS DeepComposer를 사용하여 바로크 또는 바흐 스타일 구성과 유사한 음악을 생성합니다. DeepComposer에서 이미 제공한 여러 샘플들 중 하나를 선택할 수 있습니다. 그런 다음 Autoregressive CNN 및 GAN 기술의 조합을 사용하여 창의성, 반주 및 예기치 않은 추가 패턴을 입력 데이터에 주입하여 다양한 음악적 변형을 줄 수 있습니다. (물론 바흐 또는 바로크풍에 가깝게 유지해야 합니다.) 작곡을 완료하면 **#aug-arch-ml-workshop** 태그 (chart buster challenge 체크박스 선택 취소)를 사용하여 SoundCloud에 업로드할 수 있습니다. (계정 생성 필요) 작곡 시 고려해야 할 몇 가지 팁과 기준은 아래과 같습니다.

 * 독창성 – 독특한 구성
 * 리듬 – 멜로디/하모니 개선
 * 반주/코드 진행 – 지원
 * 소리의 청결 – 세련된 구성
 * 감수성 - 음악을 들을 때 기분이 어떤지?


## Get Started
* AWS DeepComposer Workshop (한국어 번역): https://github.com/daekeun-ml/aws-deepcomposer-samples/blob/master/README-ko.md
* 샘플 미디 파일 - https://github.com/aws-samples/aws-deepcomposer-samples/tree/master/Lab%202/original_midi

**Hint:** 아래 레퍼런스에 제공된 링크를 사용하여 GAN 및 DeepComposer 사용 방법에 대해 먼저 알아보세요.

## References

* DeepComposer Demo Videos from Youtube - https://www.youtube.com/watch?v=XH2EbK9dQlg&t=9s
* DeepComposer Learning Capsule 1 Generative Adversarial Networks - https://console.aws.amazon.com/deepcomposer/home?region=us-east-1#learningCapsules/introToGANs
* DeepComposer Learning Capsule 2 Autoregressive CNN - https://console.aws.amazon.com/deepcomposer/home?region=us-east-1#learningCapsules/autoregressive
* Creating a new music genre model - https://aws.amazon.com/blogs/machine-learning/creating-a-music-genre-model-with-your-own-data-in-aws-deepcomposer/
* SoundCloud - https://soundcloud.com/tags/awsdeepcomposer
