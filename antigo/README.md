<div align="center">

<img src="https://img.shields.io/badge/TCC-Ciência_da_Computação-1F3864?style=for-the-badge&logoColor=white" />
<img src="https://img.shields.io/badge/Centro_Universitário-FEI-C0392B?style=for-the-badge&logoColor=white" />
<img src="https://img.shields.io/badge/Semestre-1º_2026-2E75B6?style=for-the-badge&logoColor=white" />

<br/><br/>

# 🧠 MindFlow AI

### *Ecossistema Multimodal de Análise Cognitiva para Ambientes Virtuais, Corporativos e Educacionais*

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev)
[![LSTM](https://img.shields.io/badge/Model-LSTM-6C3483?style=flat-square&logoColor=white)](https://pytorch.org)
[![DAiSEE](https://img.shields.io/badge/Dataset-DAiSEE-148F77?style=flat-square&logoColor=white)](https://iith.ac.in/~daisee-dataset)
[![LGPD](https://img.shields.io/badge/Conformidade-LGPD-C0392B?style=flat-square&logoColor=white)](https://www.gov.br/lgpd)
[![Status](https://img.shields.io/badge/Status-Em_Desenvolvimento-F39C12?style=flat-square&logoColor=white)](.)

<br/>

> **O comunicador online opera em vácuo de feedback.**
> Câmera ligada não significa mente presente.
> O MindFlow AI resolve isso.

<br/>

</div>

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [O Problema](#-o-problema)
- [Contextos de Uso](#-contextos-de-uso)
- [Arquitetura Geral](#-arquitetura-geral)
- [Pipeline Técnico](#-pipeline-técnico)
  - [① Treinamento Offline](#-treinamento-offline)
  - [② Cliente Local em Sessão](#-cliente-local-em-sessão)
  - [③ Camada de Aplicação](#-camada-de-aplicação)
- [Semáforo Cognitivo](#-semáforo-cognitivo)
- [Dashboard e Chatbot RAG](#-dashboard-e-chatbot-rag)
- [Privacidade e LGPD](#-privacidade-e-lgpd)
- [Dataset — DAiSEE](#-dataset--daisee)
- [Estado da Arte](#-estado-da-arte)
- [Delimitações do Escopo](#-delimitações-do-escopo)
- [Equipe](#-equipe)
- [Referências](#-referências)

---

## 🔍 Sobre o Projeto

O **MindFlow AI** é um ecossistema de análise cognitiva e apoio à decisão desenvolvido como Trabalho de Conclusão de Curso em Ciência da Computação no **Centro Universitário FEI**. Seu objetivo é detectar e sinalizar estados cognitivos — engajamento, tédio e confusão — em qualquer contexto de comunicação online mediado por vídeo, em tempo real e de forma ética.

O sistema integra **Visão Computacional** e **Computação Afetiva** em uma arquitetura projetada para operar sem armazenar imagens, sem reconhecimento facial e sem comprometer a privacidade dos participantes — garantindo plena conformidade com a **LGPD**.

```
latência de inferência: < 200ms por janela de 30 frames
saída:  { state: "confusion", confidence: 0.78, timestamp: "14:23:41" }
```

---

## 🎯 O Problema

Em ambientes presenciais, professores, palestrantes e facilitadores leem sinais não-verbais continuamente para calibrar sua comunicação. Em ambientes online, essa camada desaparece.

```
Presencial                          Online
──────────────────────────────────────────────────────────
✅ Expressões faciais visíveis      ❌ Câmera desligada ou comprimida
✅ Postura corporal legível         ❌ Participante recostado, invisível
✅ Direção do olhar detectável      ❌ Olhar para o celular, indetectável
✅ Feedback contínuo e em tempo real ❌ Vácuo de feedback
```

> O resultado: o comunicador só descobre que a audiência desengajou **depois**, pelo fracasso do objetivo da sessão.

O MindFlow AI resolve isso em **três momentos de valor**:

| Momento | Valor entregue |
|---|---|
| 🔴 **Durante a sessão** | Intervenção antes que o desengajamento se torne irreversível |
| 🟡 **No pós-sessão** | Evidência precisa de onde e quando o público perdeu a atenção |
| 🟢 **Ao longo do tempo** | Histórico comparativo de formatos, ritmos e abordagens eficazes |

---

## 🏢 Contextos de Uso

<details>
<summary><b>🎓 Educação Online</b> — aulas ao vivo, cursos gravados, tutoriais</summary>

<br/>

Um professor de curso ao vivo integra o MindFlow à sua plataforma de videoconferência. Ao final de uma aula de 90 minutos, o relatório mostra que entre os minutos 35 e 52 a confusão disparou em 60% dos alunos — exatamente enquanto ele explicava backpropagation.

O sistema gera automaticamente:

> *"Alto índice de confusão detectado entre **35:10 e 51:40** — considere revisar o conceito de gradiente com um exemplo visual na próxima aula."*

</details>

<details>
<summary><b>🎤 Palestras e Eventos Corporativos</b> — conferências, keynotes, webinars</summary>

<br/>

Um palestrante em uma conferência online com 500 participantes usa o Semáforo Cognitivo. Quando o engajamento coletivo cai abaixo do limiar configurado, ele recebe uma **notificação discreta** sugerindo pausa, enquete ao vivo ou mudança de ritmo — antes que a audiência abandone mentalmente a sessão.

</details>

<details>
<summary><b>💼 Vendas e Apresentações de Produto</b> — pitches, demos, lives de venda</summary>

<br/>

Uma equipe comercial realiza uma live de demonstração de produto. O MindFlow registra quais segmentos geraram maior engajamento e em quais momentos o tédio ou a confusão aumentaram. Com esses dados, o time reformula o roteiro da próxima live.

</details>

<details>
<summary><b>🏠 Home Office e Reuniões Corporativas</b> — calls remotas, apresentações internas</summary>

<br/>

Um facilitador de reuniões remotas monitora o engajamento coletivo durante calls longas. O dashboard pós-reunião indica os momentos em que os participantes desligaram cognitivamente, auxiliando no redesenho de estrutura, duração e frequência de interação das reuniões.

</details>

---

## 🏛️ Arquitetura Geral

O sistema é organizado em **3 níveis arquiteturais** com um princípio central de privacidade:

```
┌─────────────────────────────────────────────────────────┐
│  ① TREINAMENTO OFFLINE  (executado uma única vez)       │
│  DAiSEE → MediaPipe → Fusion → Norm → SMOTE → PCA      │
│         → Divisão 60/20/20 → LSTM → Modelo Treinado    │
└───────────────────────┬─────────────────────────────────┘
                        │ deploy (pesos congelados)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  ② CLIENTE LOCAL EM SESSÃO  (dispositivo do participante)│
│                                                         │
│  Stream da Call (vídeo + áudio + tela compartilhada)    │
│       │              │              │                   │
│  MediaPipe       Whisper        OCR + Frames           │
│  (landmarks)  (transcrição)   (tela compart.)          │
│       │              │              │                   │
│  Vetor ~120d    Texto + ts     Texto OCR + ts          │
│       │                                                 │
│  ⚠️  STREAM ORIGINAL DESCARTADO — nada bruto sai       │
│       │                                                 │
│  LSTM Inferência → Engajado / Entediado / Confuso      │
└───────────────────────┬─────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
  ③ TEMPO REAL   PERSISTÊNCIA   PÓS-SESSÃO
  Semáforo       DB Séries +    Dashboard +
  Cognitivo      DB Vetorial    Chatbot RAG
```

> **Princípio central:** nas três modalidades, o dado bruto é processado localmente, a representação reduzida é transmitida e o dado bruto é descartado. Nenhum frame de vídeo, trecho de áudio ou screenshot sai do dispositivo.

---

## ⚙️ Pipeline Técnico

### ① Treinamento Offline

| # | Etapa | Entrada | Saída | Justificativa-chave |
|---|---|---|---|---|
| 1 | **DAiSEE** | — | Vídeos anotados (engajamento, tédio, confusão) | Único dataset público de escala para estados cognitivos em webcam real |
| 2 | **MediaPipe** | Frames anotados | 468 pts faciais + 33 pts pose + vetor iris | Sem GPU, execução local, descarte imediato do frame |
| 3 | **Early Fusion** | 3 vetores separados | Vetor unificado ~120d | Modalidades sincronizadas no mesmo frame justificam concatenação direta |
| 4 | **Normalização Z-score** | Vetor ~120d | Vetor padronizado (μ=0, σ=1) | Normalizar *antes* do SMOTE garante distâncias uniformes na síntese |
| 5 | **SMOTE** | Dataset desbalanceado | Dataset balanceado | Aplicado sobre vetor unificado para coerência interna das amostras sintéticas |
| 6 | **PCA** *(opcional)* | Vetor ~120d | Vetor reduzido (95% variância) | Validado empiricamente — treinamento com e sem PCA comparado na validação |
| 7 | **Divisão 60/20/20** | Dataset completo | Treino / Validação / Teste | Teste usado **uma única vez** ao final para não contaminar os resultados |
| 8 | **LSTM** | Sequências de 30 frames | Modelo treinado (pesos congelados) | Engajamento é estado temporal — LSTM captura dependências que CNN ignora |

<details>
<summary>📐 Detalhes da arquitetura LSTM</summary>

<br/>

```
Input: sequência de 30 frames × vetor ~120d
  │
  ├── LSTM 128 unidades
  │      └── Dropout 0.3
  ├── LSTM 64 unidades
  │
  └── Dense 3 → Softmax
         │
         ├── Engajado
         ├── Entediado
         └── Confuso
```

- Janela temporal: **30 frames ≈ 1 segundo**
- Latência de inferência em produção: **< 200ms**
- Por que não CNN: arquitetura otimizada para estrutura espacial, não captura dependências temporais dos dados de sequência de features

</details>

---

### ② Cliente Local em Sessão

O cliente local é uma **extensão de browser ou aplicativo leve** que se conecta à call como participante e captura o stream completo.

```
                    Stream da Call
                   (vídeo + áudio + tela)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     MediaPipe         Whisper       OCR + Frames
    (landmarks)     (transcrição)   (tela compart.)
    visão → vetor   áudio → texto   tela → texto
          │               │               │
     descarta         descarta        descarta
      frame            áudio         screenshot
```

| Extrator | Entrada | Saída | Vai para |
|---|---|---|---|
| **MediaPipe** | Frames de vídeo | Vetor ~120d + timestamp | LSTM Inferência |
| **Whisper** (local) | Áudio da sessão | Texto transcrito + timestamps | Alinhamento Temporal |
| **OCR + Frames-chave** | Tela compartilhada | Texto OCR + frames + timestamp | Alinhamento Temporal (RAG) |

> **Por que Whisper local?** Mesmo princípio do MediaPipe: o áudio bruto nunca sai do dispositivo. Apenas a representação reduzida (texto) é transmitida — garantindo simetria de privacidade entre as três modalidades.

> **Papel da tela compartilhada:** alimenta **exclusivamente o RAG do Chatbot** (contexto de conteúdo). Não influencia a classificação de estados cognitivos pelo LSTM.

---

### ③ Camada de Aplicação

#### 🔴 Trilho Tempo Real — Semáforo Cognitivo

```
LSTM Inferência
      │  stream em tempo real
      ▼
Agregação Janela 30s → Semáforo Cognitivo → 📢 Notificação ao Comunicador
(engajamento coletivo)   (limiar configurável)   (não persiste em disco)
```

#### 🟡 Trilho Persistência

```
Inferências cognitivas + ts  ─┐
Texto transcrito + ts        ─┼─→ Alinhamento Temporal → Chunking + Embeddings → DB Vetorial
Texto OCR + ts               ─┘          │
                                          └──────────────────────────────→ DB Séries Temporais
```

| Componente | Função |
|---|---|
| **Alinhamento Temporal** | Sincroniza por timestamp as 3 saídas do cliente local |
| **Chunking + Embeddings** | Divide texto em trechos semânticos e gera vetores |
| **DB Séries Temporais** | Inferências + texto agregados por janela — alimenta o Dashboard |
| **DB Vetorial** | Embeddings dos chunks — alimenta o Chatbot (busca semântica) |

> Cada sessão é isolada por chave própria. Dados de clientes diferentes nunca se cruzam.

#### 🟢 Trilho Pós-Sessão

```
DB Séries Temporais ─────────────────────→ Dashboard Analítico ─┐
DB Séries Temporais ──→ Chatbot (LLM+RAG) ─────────────────────┼─→ Comunicador
DB Vetorial ─────────→ Chatbot (LLM+RAG) ─────────────────────┘
```

---

## 🚦 Semáforo Cognitivo

O Semáforo Cognitivo é o módulo de **intervenção em tempo real** do ecossistema.

Emite notificações visuais discretas ao comunicador quando identifica:

```
🔴  Queda crítica no engajamento geral  →  "Considere uma pausa ou enquete"
🟡  Pico de confusão coletiva           →  "Reexplique o conceito por outra perspectiva"
🟠  Aumento de tédio sustentado         →  "Mude o formato visual ou conte uma história"
```

**Ações sugeridas:** pausa · enquete interativa · mudança de formato visual · storytelling · reexplicação sob nova perspectiva

---

## 📊 Dashboard e Chatbot RAG

### Dashboard Analítico

Lê do **DB Séries Temporais** e entrega:

- 📈 Timeline de engajamento por sessão
- 🗺️ Distribuição percentual por estado cognitivo
- ⚠️ Momentos críticos identificados com timestamp
- 📋 Comparativo entre sessões ao longo do tempo

### Chatbot (LLM externo + RAG)

O chatbot permite ao comunicador **conversar com os dados da sua sessão** em linguagem natural.

```
Usuário:  "Em que ponto do conteúdo a confusão aumentou?"
          "O que estava na tela quando o público desengajou?"
          "Qual parte da apresentação gerou mais tédio?"

Sistema:  consulta semântica no DB Vetorial
       +  cross-reference temporal no DB Séries
       +  contexto enviado ao LLM via API
          │
          └──→ Resposta fundamentada em conteúdo + reações reais
```

<details>
<summary>📐 Por que RAG e não fine-tuning?</summary>

<br/>

| | RAG ✅ | Fine-tuning ❌ |
|---|---|---|
| Dados absorvidos pelo modelo | Não — contexto efêmero por consulta | Sim — irreversivelmente nos pesos |
| Direito ao esquecimento (LGPD) | ✅ Basta excluir registros do banco | ❌ Impossível remover dados dos pesos |
| Custo computacional | Baixo | Alto |
| Isolamento por sessão | ✅ Garantido por arquitetura | ❌ Dados se misturam no modelo |

</details>

---

## 🔒 Privacidade e LGPD

> Privacidade é **restrição de design**, não camada adicional. Conformidade com a LGPD, Art. 11 (dados sensíveis e biométricos).

### Pilares do Cliente Local (aplicáveis às 3 modalidades)

| Pilar | Garantia |
|---|---|
| 🎥 **Nenhuma mídia bruta armazenada** | Frames, áudio e screenshots descartados imediatamente após extração local |
| 👤 **Nenhum rosto reconhecido** | Sistema processa vetores matemáticos — identidade é matematicamente inacessível |
| ✋ **Participação opt-in** | Apenas quem escolhe explicitamente tem o stream processado |

### Pilares da Camada de Aplicação

| Pilar | Garantia |
|---|---|
| 🪟 **Agregação por janela** | Impede reconstrução de comportamentos individuais finos |
| 🔁 **RAG sem fine-tuning** | LLM não absorve dados das sessões; exclusão do banco = esquecimento total |
| 🔐 **Isolamento por sessão e cliente** | Dados de sessões diferentes nunca se cruzam |
| 📤 **Transmissão apenas de representações reduzidas** | Nada bruto sai do dispositivo para serviços externos |

---

## 📦 Dataset — DAiSEE

| Atributo | Valor |
|---|---|
| **Nome completo** | Dataset for Affective States in E-Environments |
| **Desenvolvido por** | IIT Delhi |
| **Vídeos** | 9.068 clipes |
| **Participantes** | 112 estudantes |
| **Captura** | Webcam em condições reais (não laboratório) |
| **Anotações** | Engajamento · Tédio · Confusão · Frustração (escala 0–3) |
| **Avaliadores** | Múltiplos avaliadores por clipe |
| **Principal desafio** | Desbalanceamento severo de classes (engajado predomina) → motiva o SMOTE |

**Por que DAiSEE?** É o único dataset público de escala relevante construído para captura real por webcam, cujos estados rotulados são diretamente transferíveis para todos os contextos de uso do MindFlow AI além do educacional.

---

## 📚 Estado da Arte

<details>
<summary><b>Eixo 1 — Detecção de Engajamento</b> (benchmark no DAiSEE)</summary>

<br/>

| Modelo | Acurácia | Limitação |
|---|---|---|
| CNN end-to-end (IEEE 2019) | **98,82%** binária | Não distingue tédio de confusão |
| CNN + SVD + OpenFace (IJACSA 2023) | **77,97%** 4 classes | Depende de armazenamento de imagens faciais |
| ResNet + TCN (arXiv 2024) | **63,9%** 4 classes | Sobreposição entre classes adjacentes |
| DenseAttNet (Tech Science Press 2024) | **63,59%** 4 classes | Sensível a variações de iluminação |
| SVM + fusão áudio-visual (EmotiW 2018) | **65%** val · **61%** teste | Baseline fraco (50,05%) |

</details>

<details>
<summary><b>Eixo 2 — Reconhecimento de Emoções Faciais</b></summary>

<br/>

| Modelo | Acurácia | Dataset |
|---|---|---|
| ResNet-50 + CBAM + CNN 3D (Aly & Alotaibi 2025) | 97,3% tempo real | FER2013, CK+, KDEF |
| VGG-19 + Dlib + Faster R-CNN (Gupta et al. 2023) | 92,58% binária | FER2013, CK+ |
| DenseNet-161 / ViT (IAPRESS Survey 2024) | até 99,52% | JAFFE, AffectNet |
| CNNs e Transformers (BT-FER 2023) | 79% RAF-DB · 63% AffectNet | RAF-DB, FER2013 |

</details>

<details>
<summary><b>Eixo 3 — Sistemas Multimodais</b></summary>

<br/>

| Trabalho | Técnica | Contribuição |
|---|---|---|
| JISEM 2025 | BERT + acústico + CNN | Superior a qualquer modalidade isolada |
| IEEE 2025 | LLM + CNN + acústico | Estado da arte em monitoramento multimodal em tempo real |
| Nature Scientific Reports 2026 | Aprendizado colaborativo multimodal | Melhora substancial em expressões ambíguas, foco multicultural |

</details>

### 📍 Posição do MindFlow AI no Estado da Arte

```
Alta acurácia, sem privacidade     (92–98%)   ←── modelos que armazenam imagens
                                    lacuna
Privacidade rigorosa, sem acurácia (≈50%)     ←── monitoramento de abas/teclado

                        ↑
              MindFlow AI busca preencher essa lacuna:
         acurácia competitiva dentro de envelope ético rigoroso
```

**Diferenciais sobre o estado da arte:**
- 3 vetores multimodais (visão + áudio + tela) — vs. áudio-visual da maioria
- LSTM temporal — vs. classificação frame-a-frame
- Operação sobre geometrias, não texturas — reduz viés demográfico
- Conformidade LGPD por arquitetura, não por declaração

---

## 🚫 Delimitações do Escopo

O sistema **não faz** e **não é projetado para fazer**:

- ❌ Identificar *quem* está desengajado — processa estados coletivos e sessões anônimas
- ❌ Armazenar vídeo, imagem, áudio ou screenshot em qualquer etapa
- ❌ Reconhecimento facial ou biometria de qualquer natureza
- ❌ Operar sem consentimento explícito — participação sempre reversível
- ❌ Usar CNN como classificador — modelo adotado é LSTM (temporal)
- ❌ Fine-tuning do LLM com dados das sessões
- ❌ Cruzar dados entre sessões ou entre clientes diferentes

---

## 👥 Equipe

<div align="center">

| | Nome | E-mail |
|---|---|---|
| 👨‍💻 | Gustavo Bertoluzzi Cardoso | unifgcardoso@fei.edu.br |
| 👨‍💻 | Henrique Hodel Babler | uniehbabler@fei.edu.br |
| 👩‍💻 | Isabella Vieira Silva Rosseto | unifirosseto@fei.edu.br |
| 👨‍💻 | Matheus Ferreira de Freitas | unifmfreitas@fei.edu.br |

**Orientadora:** Profa. Leila Cristina Carneiro Bergamasco — `leila.cristina@fei.edu.br`
Departamento de Ciência da Computação — Centro Universitário FEI

</div>

---

## 📖 Referências

<details>
<summary>Ver todas as 15 referências</summary>

<br/>

```
[1]  ACM ICMI. "EmotiW 2018: Audio-Video, Student Engagement and Group-Level
     Affect Prediction." ACM International Conference on Multimodal Interaction, 2018.

[2]  M. Aly e N. S. Alotaibi. "A comprehensive deep learning framework for
     real-time emotion detection in online learning." PMC/NIH, 2025.

[3]  arXiv. "A General Model for Detecting Learner Engagement." 2024.

[4]  arXiv/CVPR. "Benchmarking Deep Facial Expression Recognition." 2023.

[5]  S. Gupta et al. "A multimodal facial cues based engagement detection
     system in e-learning." PMC/NIH, 2023.

[6]  IAPRESS. "Facial Expression Recognition: A Survey of Techniques,
     Datasets and Challenges." Science of Information and Communications, 2024.

[7]  IEEE. "A Multimodal Approach for Real-Time Engagement Monitoring
     in E-Learning." IEEE, 2025.

[8]  IEEE. "An Novel End-to-end Network for Automatic Student Engagement
     Recognition." IEEE Conference on Automatic Face and Gesture Recognition, 2019.

[9]  IFSP. "Método de Avaliação do Engajamento em Ambientes Virtuais
     com Deep Learning." CONICT, 2025.

[10] IJACSA. "CNN Model based Students' Engagement Detection."
     International Journal of Advanced Computer Science and Applications, 2023.

[11] JISEM Journal. "Multimodal Emotion Recognition: A Tri-modal Approach
     Using Text, Audio and Visual Data." JISEM, 2025.

[12] Nature Scientific Reports. "Multi-modal collaborative learning for
     facial expression recognition in e-learning." 2026.

[13] RAIS Education. "Reducing Racial and Ethnic Bias in AI Models." 2024.

[14] SciELO Brasil. "Vigilância e privacidade no ambiente digital." RDBCI, 2023.

[15] Tech Science Press. "Detection of Student Engagement in E-Learning
     Environments Using DenseAttNet." Journal of Artificial Intelligence, 2024.
```

</details>

---

<div align="center">

**MindFlow AI** · TCC Ciência da Computação · Centro Universitário FEI · 2026

*Desenvolvido com foco em análise cognitiva ética, precisa e em tempo real.*

</div>
