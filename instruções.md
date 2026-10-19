# MindFlow AI — tutorial e documentação completa do protótipo

> **Versão documentada:** extensão 1.6.2
> 
> **Site:** [Mind Flow Site](mindflowiaa.netlify.app)


## Sumário

- [Como instalar e rodar](#como-instalar-e-rodar)
- [1. O que compõe o projeto](#1-o-que-compõe-o-projeto)
- [2. Fluxo completo, do Meet ao site](#2-fluxo-completo-do-meet-ao-site)
- [3. Como a extensão funciona dentro do Google Meet](#3-como-a-extensão-funciona-dentro-do-google-meet)
- [4. DOM: como a extensão encontra e acompanha os vídeos](#4-dom-como-a-extensão-encontra-e-acompanha-os-vídeos)
- [5. Login, título e ciclo da reunião](#5-login-título-e-ciclo-da-reunião)
- [6. Como os frames chegam ao Python](#6-como-os-frames-chegam-ao-python)
- [7. Como o Python transforma imagem em 120 características](#7-como-o-python-transforma-imagem-em-120-características)
- [8. Como a LSTM produz os quatro indicadores](#8-como-a-lstm-produz-os-quatro-indicadores)
- [9. O que todas as porcentagens significam](#9-o-que-todas-as-porcentagens-significam)
- [10. Como funcionam farol, suavização e gráfico ao vivo](#10-como-funcionam-farol-suavização-e-gráfico-ao-vivo)
- [11. O que o Supabase salva](#11-o-que-o-supabase-salva)
- [12. Como o site usa os dados](#12-como-o-site-usa-os-dados)
- [13. Usando somente o Python](#13-usando-somente-o-python)
- [14. Treinamento, DAiSEE e precisão atual](#14-treinamento-daisee-e-precisão-atual)
- [15. O que mudou desde o mindflow_sem_data_set](#15-o-que-mudou-desde-o-mindflow_sem_data_set)
- [16. Privacidade e capturas locais de teste](#16-privacidade-e-capturas-locais-de-teste)
- [17. Problemas comuns](#17-problemas-comuns)
- [18. Críticas e melhorias recomendadas](#18-críticas-e-melhorias-recomendadas)
- [19. Resumo para apresentação](#19-resumo-para-apresentação)
- [Glossário](#glossário)

---

## Como instalar e rodar

Esta é a parte prática. A explicação interna começa na seção seguinte.

### O que precisa estar no pacote

Depois de extrair `mindflow-integration.zip`, a pasta deve conter pelo menos:

```text
mindflow-integration\
├── extension\
│   └── manifest.json
├── mindflow_extractor\
├── models\
│   └── apresentacao_TCC\
│       ├── lstm_mindflow.pt
│       ├── scaler.joblib
│       ├── pca.joblib
│       └── metrics.json
├── site\
├── supabase\
├── training\
├── tools\
├── requirements.txt
└── iniciar_mindflow.bat
```

A pasta `.venv` não precisa e não deve ser enviada pelo Git. Cada integrante cria seu próprio ambiente Python.

### Requisitos

- Windows;
- Google Chrome ou navegador Chromium compatível;
- Python 3.11 instalado;
- internet para instalar as dependências e usar o Supabase;
- o pacote completo, incluindo `models/apresentacao_TCC`.

### Primeira execução do Python

Abra o PowerShell dentro da pasta `mindflow-integration` e execute:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

O `requirements.txt` instala as bibliotecas usadas pelo projeto, como MediaPipe, OpenCV, NumPy, PyTorch, scikit-learn e joblib. O arquivo não cria a `.venv`; o primeiro comando é quem cria o ambiente virtual.

O usuário comum **não precisa baixar o DAiSEE nem treinar o modelo**. A extensão usa os artefatos que já estão em `models/apresentacao_TCC`.

### Instalar a extensão no Chrome

1. Abra `chrome://extensions`.
2. Ative **Modo do desenvolvedor**.
3. Clique em **Carregar sem compactação**.
4. Selecione a pasta `mindflow-integration\extension`.
5. Confirme que o Chrome mostra **MindFlow AI — Controle Local**, versão **1.6.2**.
6. Se o Meet já estava aberto, atualize a página.

O Chrome deve receber a pasta que contém `manifest.json`, não o ZIP fechado e não a raiz inteira do projeto.

### Sempre que for usar em uma reunião

1. Dê dois cliques em `iniciar_mindflow.bat`.
2. Aguarde a mensagem de modelo carregado e servidor em `http://127.0.0.1:8765`.
3. Mantenha essa janela aberta.
4. Abra o Google Meet e entre efetivamente na chamada.
5. Faça login no painel MindFlow com a mesma conta usada no site.
6. Aguarde a criação da reunião no Supabase.
7. Confira as etiquetas sobre os vídeos.
8. Participantes permitidos começam a ser analisados automaticamente; use **Ignorar análise** quando uma fonte não deve entrar no cálculo.
9. Aguarde a coleta inicial de 30 frames processados.
10. Confira o farol, os quatro indicadores, o FPS real e o status do histórico.

Um estado saudável se parece com:

```text
Python conectado · modelo carregado · alvo 30 fps
Histórico Supabase: ativo · CÓDIGO · N registros
```

Para conferir o servidor diretamente, abra no mesmo computador:

```text
http://127.0.0.1:8765/health
```

`"ok": true` confirma que o servidor respondeu. `"modelLoaded": true` confirma que os três artefatos necessários foram carregados. `history.enabled` só fica verdadeiro depois que a extensão informa uma reunião válida.

---

## 1. O que compõe o projeto

O MindFlow não é apenas uma extensão. O protótipo é uma integração entre quatro partes:

| Parte | Responsabilidade principal |
|---|---|
| **Extensão do Chrome** | Entra no Google Meet, observa os vídeos, bloqueia tela/apresentador, captura frames autorizados, mostra resultados e controla a conta e a reunião. |
| **Python local** | Recebe imagens, executa MediaPipe, produz as características numéricas, roda o modelo e envia histórico numérico. |
| **Supabase** | Autentica usuários e guarda reuniões, resultados, preferências, dashboards e anotações com regras de acesso. |
| **Site** | Consulta o histórico, mostra dashboards, permite comparar reuniões, selecionar intervalos, anotar momentos e personalizar a extensão. |

### Pastas e arquivos principais

| Caminho | Para que serve |
|---|---|
| `extension/manifest.json` | Declara a extensão, permissões, páginas atendidas e arquivos carregados. |
| `extension/content.js` | Atua dentro do Meet: DOM, fontes, overlays, captura, painel, suavização e gráfico ao vivo. |
| `extension/background.js` | Faz a ponte com Python e Supabase, login, reunião, heartbeat e sincronização. |
| `extension/styles.css` | Aparência do painel e das etiquetas. |
| `mindflow_extractor/pipeline.py` | Coordena Face Mesh, Pose e os sete blocos de features. |
| `mindflow_extractor/inference.py` | Carrega scaler, PCA e LSTM e calcula níveis, intensidade e confiança. |
| `mindflow_extractor/runners/extension_server.py` | Servidor HTTP local usado pela extensão. |
| `mindflow_extractor/supabase_history.py` | Fila assíncrona que grava o histórico numérico. |
| `models/apresentacao_TCC/` | Modelo atualmente utilizado pela extensão. |
| `training/` | Código de preparação, treinamento, avaliação e gráficos. |
| `tools/run_daisee_pilot.py` | Piloto de extração e treino com vídeos DAiSEE a 30 FPS. |
| `site/index.html` | Estrutura das telas do site. |
| `site/styles.css` | Aparência do site. |
| `site/app.js` | Comportamentos, gráficos, intervalos, dashboards e navegação. |
| `site/data-service.js` | Camada que traduz as telas para consultas ao Supabase. |
| `site/supabase-setup.sql` | Ajustes de perfil, segurança, demonstrações, preferências, layouts e anotações. |
| `supabase/README.md` | Contrato esperado das funções de histórico. |

Não existe `offscreen.js` nesta versão. O MediaPipe também não está dentro da extensão: ele roda no Python.

---

## 2. Fluxo completo, do Meet ao site

```text
Google Meet
   ↓
manifest.json carrega content.js e styles.css
   ↓
content.js observa o DOM e encontra elementos <video>
   ↓
classifica fonte: participante / tela / apresentador local
   ↓
participante permitido entra automaticamente em análise
   ↓
canvas cria um JPEG temporário 640 × 480
   ↓
content.js envia uma mensagem ao background.js
   ↓
background.js faz POST em http://127.0.0.1:8765/frame
   ↓
Python executa MediaPipe Face Mesh e Pose
   ↓
478 landmarks faciais + 33 landmarks corporais
   ↓
120 features nos blocos A–G
   ↓
buffer separado por fonte anônima
   ↓
30 frames → scaler → PCA → LSTM
   ↓
Engajamento + Tédio + Confusão + Frustração
   ├──→ resposta ao painel da extensão
   └──→ histórico numérico no Supabase
                         ↓
                    site e dashboards
```

Há dois caminhos diferentes:

- **caminho ao vivo:** o resultado volta imediatamente do Python para a extensão;
- **caminho histórico:** o Python coloca uma cópia numérica em uma fila e a envia ao Supabase sem atrasar o painel.

O site não inicia a reunião. A reunião começa no Google Meet, é criada automaticamente no banco pela extensão e aparece no site para acompanhamento ou consulta posterior.

---

## 3. Como a extensão funciona dentro do Google Meet

O `manifest.json` manda o Chrome carregar `content.js` e `styles.css` em endereços que começam com:

```text
https://meet.google.com/*
```

O `content.js` roda depois da estrutura inicial da página e faz o seguinte:

1. impede que outra cópia do próprio script seja carregada;
2. lê dispositivos de câmera para ajudar a reconhecer a câmera local;
3. procura vídeos visíveis;
4. cria pseudônimos como `participant_01`;
5. classifica tela, apresentador local ou participante;
6. insere etiquetas sobre os tiles;
7. cria o painel flutuante;
8. verifica login, Python e reunião;
9. inicia observação do DOM, captura, heartbeat e atualização visual.

### Arquivos da extensão e ambientes diferentes

| Arquivo | Onde roda | Função |
|---|---|---|
| `content.js` | Aba do Meet | Enxerga o DOM, os vídeos e o painel. |
| `background.js` | Service worker da extensão | Pode chamar `127.0.0.1` e o Supabase com as permissões declaradas. |
| `popup.*` | Janela do ícone da extensão | Conta e informações auxiliares. |
| `login-panel.*` | Iframe dentro do painel | Login compacto sem redirecionar o usuário para outro site. |
| `styles.css` | Aba do Meet | Visual dos componentes inseridos. |

O `content.js` não chama o Python diretamente. Ele manda uma mensagem interna ao `background.js`, que faz a requisição HTTP:

```text
content.js
   ↓ chrome.runtime.sendMessage
background.js
   ├──→ Python local
   └──→ Supabase
```

### Estados de uma fonte

| Estado | Significado |
|---|---|
| **TELA DETECTADA** | Compartilhamento bloqueado automaticamente. |
| **APRESENTADOR DETECTADO** | Câmera local bloqueada automaticamente. |
| **LOGIN NECESSÁRIO** | Nenhum frame é enviado até a autenticação. |
| **IA ATIVA** | Fonte participante sendo enviada ao Python. |
| **IGNORADO** | Usuário interrompeu manualmente a análise dessa fonte. |

Depois do login e da criação da reunião, participantes elegíveis entram automaticamente em **IA ATIVA**, até o limite de cinco fontes. O botão permite **Ignorar análise** e, depois, **Retomar análise**. Tela compartilhada e apresentador local não oferecem essa escolha porque são bloqueados automaticamente.

---

## 4. DOM: como a extensão encontra e acompanha os vídeos

DOM significa **Document Object Model**. Na prática, é a árvore de elementos que o navegador montou para exibir a página.

```text
Página do Meet
├── cabeçalho e controles
├── grade da chamada
│   ├── tile de participante
│   │   └── <video>
│   ├── tile de participante
│   │   └── <video>
│   └── tile de apresentação
│       └── <video>
└── chat, pessoas e atividades
```

O MindFlow usa essa árvore para:

- encontrar todos os `<video>`;
- medir onde cada tile está;
- ler atributos e textos próximos;
- detectar sinais como “você”, “you”, “apresentação” e “screen”;
- tentar obter o título da reunião;
- perceber controles de chamada e telas de encerramento;
- inserir e reposicionar painel e etiquetas.

### Filtragem de vídeos

Um `<video>` só vira fonte se:

- ainda estiver ligado à página;
- não estiver oculto ou transparente;
- tiver pelo menos 48 × 36 pixels;
- estiver dentro da área visível da janela.

A busca completa ocorre a cada 900 ms.

### Por que existe um MutationObserver

O Meet muda a grade sem recarregar a página. Um `MutationObserver` acompanha elementos adicionados, removidos ou reorganizados e mudanças de `src`, classe e estilo. Quando algo muda, a extensão espera 120 ms e verifica novamente.

Alterações feitas dentro dos próprios componentes MindFlow são ignoradas para que o observador não reaja à própria renderização em um ciclo infinito.

Também há nova leitura quando a janela é redimensionada, rolada ou quando os dispositivos de câmera mudam.

### Identidade anônima e continuidade

Cada fonte recebe um ID local, por exemplo:

```text
participant_01
participant_02
participant_03
```

O código tenta manter esse ID mesmo quando a grade muda usando sinais técnicos do stream, da faixa de vídeo e de atributos do DOM. Isso não é reconhecimento facial e não usa o nome exibido no Meet.

A ligação serve para manter o mesmo buffer no Python:

```text
<video> ↔ participant_02 ↔ buffer da fonte ↔ histórico pseudônimo
```

Quando uma fonte some, existe uma tolerância de 12 segundos. Depois disso, a etiqueta é removida e o Python recebe `/reset/<id>` para descartar o estado antigo.

### Como tela e apresentador local são bloqueados

A fonte pode ser classificada como tela quando o navegador ou o tile indicam monitor, janela, navegador, display, compartilhamento ou apresentação.

A fonte pode ser classificada como apresentador local quando:

- o `deviceId` coincide com uma câmera do computador;
- o nome técnico da câmera coincide;
- o texto do tile indica “você” ou “you”.

Essa detecção é heurística. O Google pode alterar o DOM e os textos do Meet. Portanto, ela deve ser testada periodicamente com grade, destaque, compartilhamento e diferentes idiomas.

### Etiquetas e painel

As etiquetas são elementos `<section class="mindflow-source-overlay">` adicionados ao `document.body`. A extensão usa `getBoundingClientRect()` para posicioná-las sobre o vídeo sem editar o código interno do Meet.

O painel é um `<aside id="mindflow-panel">`, também adicionado ao `document.body`. Ele é:

- flutuante;
- arrastável;
- redimensionável;
- restaurado na posição e tamanho salvos no `chrome.storage.local`.

As preferências sincronizadas controlam tema, escala, tamanho do texto e se tendência, confiança, indicadores, gráfico, fontes e detalhes técnicos ficam fixos, aparecem no hover ou ficam ocultos.

---

## 5. Login, título e ciclo da reunião

### Login obrigatório

Sem login, nenhuma fonte é analisada. O login usa o Supabase Auth:

- e-mail e senha vão do painel para o `background.js` e depois para o Supabase;
- a senha não é enviada ao Python;
- a extensão não salva a senha;
- tokens de acesso e renovação ficam no armazenamento local do Chrome;
- a mesma conta identifica o proprietário da reunião e das preferências.

### Quando a reunião é criada

A extensão procura controles de chamada no DOM. Só depois de entender que o usuário entrou de fato no Meet ela cria a sessão no Supabase.

São registrados o proprietário, título, código interno, plataforma, horário, status e atividade recente da extensão.

### Como o título é encontrado

O `content.js` procura atributos de título, rótulos acessíveis, `document.title` e cabeçalhos visíveis. Ele rejeita textos genéricos, códigos de sala e mensagens como “pronto para participar”.

Se não encontrar algo confiável, usa:

```text
Google Meet · abc-defg-hij
```

O título aparece ao lado de “MindFlow AI” e pode ser corrigido clicando nele. A correção atualiza o Supabase e a sessão informada ao Python.

### Presença e encerramento

A cada 15 segundos, a extensão envia um heartbeat e repassa novamente a sessão ao Python. Isso permite que o histórico se recupere se o servidor for aberto ou reiniciado depois que o usuário já entrou no Meet.

A reunião é encerrada quando a extensão já viu uma chamada ativa e depois reconhece uma tela de chamada encerrada ou de retorno. O site também considera uma reunião abandonada como antiga quando o heartbeat deixa de chegar e pode fechá-la depois de aproximadamente 60 segundos.

Há uma limitação: fechar abruptamente o navegador, desligar o computador ou uma mudança inesperada no DOM pode impedir o encerramento imediato. Por isso existe o tratamento de sessões antigas no site.

Se o mesmo link do Meet for reaberto em até oito horas, o armazenamento local pode reaproveitar a reunião ativa da mesma conta. Depois desse período, ou se a sessão anterior foi encerrada, uma nova reunião é criada.

---

## 6. Como os frames chegam ao Python

Para cada fonte em **IA ATIVA**:

1. o `content.js` possui um canvas de 640 × 480;
2. desenha o frame atual do vídeo nesse canvas;
3. converte o canvas em JPEG com qualidade 0,72;
4. adiciona ID anônimo, horário e tempo desde o início da página;
5. envia o pacote ao `background.js`;
6. o `background.js` faz `POST /frame` em `http://127.0.0.1:8765`.

### Por que 127.0.0.1 funciona em qualquer rede

`127.0.0.1` sempre significa “este mesmo computador”. Não é o IP do Wi-Fi. Se Chrome e Python estão na mesma máquina, trocar de rede não muda esse endereço.

### 30 FPS é um alvo, não uma garantia

A captura tenta iniciar a cada `1000 / 30` ms, mas existe no máximo uma requisição em andamento por fonte. Enquanto o Python processa o frame anterior, novos frames daquela fonte são pulados.

```text
30 FPS configurados ≠ 30 FPS realmente processados
```

O FPS real depende de CPU, quantidade de fontes, MediaPipe, tamanho dos frames e latência. Cinco fontes podem pesar bastante mesmo em um protótipo.

### Resolução atual e proporção

A resolução enviada atualmente é 640 × 480. Esse tamanho foi escolhido porque corresponde à resolução comum dos vídeos DAiSEE e reduz o custo de até cinco fontes.

Hoje o canvas desenha qualquer vídeo diretamente em 640 × 480. Se a fonte do Meet for 16:9, isso pode esticar a imagem para 4:3. Preservar a proporção com margens ou recorte controlado é uma melhoria importante antes de apenas aumentar a resolução.

Aumentar para 960 × 720 pode ajudar a detectar rostos pequenos, mas precisa ser um experimento medido. Resolução maior não garante modelo melhor: o MediaPipe pode detectar mais pontos, mas o custo cresce e treino e inferência precisam continuar compatíveis.

---

## 7. Como o Python transforma imagem em 120 características

O servidor é iniciado por `mindflow_extractor.runners.extension_server` e expõe:

| Endpoint | Função |
|---|---|
| `GET /health` | Estado do servidor, modelo, fontes, snapshots e histórico. |
| `POST /frame` | Decodifica e processa um frame. |
| `POST /reset/<id>` | Remove pipeline e buffer de uma fonte. |
| `POST /history/session` | Informa código e título da reunião ao escritor do Supabase. |

Cada fonte possui seu próprio `MindFlowPipeline`, buffer, qualidade, previsão e trava de processamento.

### MediaPipe

O JPEG é decodificado pelo OpenCV, convertido de BGR para RGB e enviado a dois extratores:

| Extrator | Resultado |
|---|---|
| **Face Mesh** | 478 pontos: 468 da face e 10 das íris, com no máximo um rosto por fonte. |
| **Pose** | 33 pontos corporais com posição e visibilidade. |

Os landmarks são coordenadas geométricas. Eles não são enviados diretamente à LSTM.

### Blocos A–G

| Bloco | Dimensão | Exemplos |
|---|---:|---|
| **A — sinais faciais** | 32 | aproximações de sobrancelhas, olhos, bochechas, lábios e mandíbula. |
| **B — cabeça** | 12 | yaw, pitch, roll e mudanças temporais. |
| **C — olhar** | 24 | direção, deslocamento, estabilidade e sacadas aproximadas. |
| **D — olhos** | 16 | abertura ocular, EAR, piscadas e sonolência aproximada. |
| **E — boca** | 8 | abertura, movimento e sinal aproximado de bocejo. |
| **F — postura** | 24 | tronco, ombros, braços, inclinação e simetria. |
| **G — qualidade** | 4 | qualidade da face, pose e condições do frame. |
| **Total** | **120** | Descrição numérica de um frame. |

### Lifecycle e qualidade

O estado de uma fonte fica em memória. Se ela deixa de aparecer por cinco minutos no servidor, é considerada expirada. Quando a extensão ignora ou remove a fonte, chama o reset imediatamente.

O Python acompanha:

- presença de face por frame;
- presença de pose por frame;
- FPS realmente processado;
- latência do frame;
- quantidade de números finitos e de `NaN`.

Esses dados ajudam a distinguir “o modelo estimou algo” de “a imagem tinha qualidade suficiente”. Atualmente a previsão ainda não é bloqueada automaticamente quando a qualidade é baixa; isso aparece nas melhorias recomendadas.

### NaN: quando uma medida não existe

Se não há rosto, os blocos A–E recebem `NaN`. Partes invisíveis da pose também podem gerar `NaN`. Isso significa “não foi possível observar”, não “o valor medido é zero”.

No caminho de inferência atual, o Python:

```text
vetor com NaN/±infinito
   ↓
substitui valores não finitos por zero
   ↓
aplica scaler
   ↓
aplica PCA
```

Esse é o comportamento real do código. Existe um comentário antigo em `config.py` dizendo que a substituição ocorreria depois do Z-score, mas a implementação de treino e de inferência a executa antes do scaler. A documentação deve seguir a implementação, e o comentário merece ser corrigido no código em uma revisão futura.

---

## 8. Como a LSTM produz os quatro indicadores

### Buffer por fonte

Cada participante possui um buffer independente:

```text
participant_01 → últimas features e previsão
participant_02 → últimas features e previsão
participant_03 → últimas features e previsão
```

O modelo oficial atual recebe 30 frames × 120 características.

```text
30 × 120
   ↓
Scaler salvo no treinamento
   ↓
PCA salvo no mesmo treinamento
   ↓
LSTM temporal
   ↓
4 cabeças independentes
```

As quatro cabeças são:

- Engajamento;
- Tédio;
- Confusão;
- Frustração.

Cada cabeça produz quatro probabilidades:

```text
0 = muito baixo
1 = baixo
2 = alto
3 = muito alto
```

A primeira previsão ocorre após 30 frames processados. Depois, uma nova inferência é feita a cada 15 frames. O Python mantém a última previsão entre essas atualizações.

Se o processamento real fosse 30 FPS, a primeira previsão levaria perto de um segundo e seria atualizada a cada meio segundo. Se a fonte estiver a 8 FPS reais, por exemplo, os mesmos 30 frames cobrem aproximadamente 3,75 segundos.

### Os arquivos do modelo são um conjunto inseparável

```text
models/apresentacao_TCC/
├── scaler.joblib
├── pca.joblib
└── lstm_mindflow.pt
```

O scaler, o PCA e a LSTM precisam vir do mesmo treinamento. Copiar apenas o `.pt` pode fazer o modelo receber dados em outra escala ou dimensão e produzir resultados inválidos.

---

## 9. O que todas as porcentagens significam

Esta é a seção mais importante para interpretar o sistema. O projeto usa pelo menos quatro tipos diferentes de porcentagem.

### 9.1 Intensidade da reação

É o número principal exibido para Engajamento, Tédio, Confusão e Frustração.

O modelo primeiro produz uma distribuição entre quatro níveis. O Python calcula:

```text
intensidade = (0×P(muito baixo) + 1×P(baixo) + 2×P(alto) + 3×P(muito alto)) ÷ 3 × 100
```

Exemplo:

```text
Engajamento
muito baixo: 10%
baixo:       20%
alto:        50%
muito alto:  20%

intensidade = (0×0,10 + 1×0,20 + 2×0,50 + 3×0,20) ÷ 3 × 100
intensidade = 60%
```

**Engajamento 60% significa uma posição de intensidade na escala entre muito baixo e muito alto.**

Não significa:

- 60% de chance de a pessoa estar engajada;
- 60% das pessoas engajadas;
- 60% de precisão do modelo.

As quatro intensidades são independentes. Elas não precisam somar 100%. É possível haver engajamento e confusão altos ao mesmo tempo.

### 9.2 Confiança no nível escolhido

O nível escolhido é aquele com a maior probabilidade. No exemplo anterior, o nível é **alto** e a confiança é 50%, porque `P(alto) = 50%`.

```text
Engajamento alto
Confiança no nível: 50%
```

Essa confiança mostra o quanto a saída interna favoreceu aquele nível naquela previsão. Ela **não é a precisão comprovada do modelo**. Redes neurais podem ser excessivamente confiantes; seria necessário calibrar as probabilidades para tratá-las como confiança estatística confiável.

### 9.3 Fração de fontes do grupo

O Supabase agrega resultados em janelas. `60% das fontes em engajamento alto ou muito alto` significa que, naquela janela, 60 de cada 100 fontes válidas seriam classificadas nos dois níveis superiores de engajamento.

Com cinco fontes:

```text
3 fontes em alto/muito alto ÷ 5 fontes válidas = 60%
```

Essa porcentagem representa pessoas/fontes agregadas e é diferente da intensidade média exibida no gráfico.

### 9.4 Acurácia e F1 do teste

Essas são métricas globais calculadas com exemplos de teste rotulados, não com uma pessoa ao vivo.

- **Acurácia:** proporção total de classificações corretas.
- **F1 macro:** calcula a qualidade de cada nível e dá o mesmo peso a níveis raros e comuns.
- **Matriz de confusão:** mostra quais níveis o modelo troca entre si.

Em um dataset desbalanceado, acurácia alta pode esconder que o modelo ignora níveis raros. Por isso F1 macro e matriz de confusão são fundamentais.

### Tabela de leitura rápida

| Texto visto | O que realmente significa |
|---|---|
| **Engajamento 60%** | Intensidade média estimada na escala ordinal. |
| **Confiança 50%** | Probabilidade interna do nível vencedor nessa saída. |
| **60% das fontes engajadas** | Fração de fontes classificadas em alto/muito alto na janela. |
| **Acurácia 49,5%** | Resultado global de uma avaliação registrada, não chance de acerto deste frame. |

Uma linguagem segura para o produto é:

```text
Engajamento alto
Intensidade estimada: 60%
Confiança no nível: 50%
```

Evite escrever apenas “60% engajado”, porque essa frase pode parecer porcentagem de pessoas ou certeza psicológica.

---

## 10. Como funcionam farol, suavização e gráfico ao vivo

### Suavização individual

A extensão guarda as últimas cinco previsões de cada fonte e calcula a média de intensidade e confiança. Isso reduz mudanças bruscas entre dois resultados consecutivos.

Uma previsão deixa de valer se a última atualização tiver mais de seis segundos.

### Resultado do grupo

Depois da suavização individual, a extensão faz uma média simples entre fontes ainda em **IA ATIVA**:

```text
intensidade do grupo = soma das intensidades válidas ÷ fontes válidas
```

Nenhuma fonte recebe peso extra por estar maior ou em outra posição da grade.

### Farol da extensão

A extensão escolhe a dimensão com maior intensidade média. Se a intensidade vencedora for pelo menos 50%:

| Dimensão dominante | Cor |
|---|---|
| Engajamento | Verde |
| Confusão | Amarelo |
| Tédio | Vermelho |
| Frustração | Vermelho |

Se nenhuma intensidade vencedora atingir 50%, o farol fica cinza.

### Tendência

A seta observa o engajamento atual e um ponto recente:

- aumento maior que 3 pontos: `↗`;
- queda maior que 3 pontos: `↘`;
- mudança pequena: `→`.

### Gráfico de cinco minutos

O painel registra uma amostra do grupo a cada dois segundos e conserva somente os últimos cinco minutos em memória. Ao fechar ou atualizar a aba, esse gráfico ao vivo é perdido.

O gráfico do site não é uma cópia desse array. Ele é reconstruído a partir das janelas agregadas no Supabase. Pequenas diferenças entre os dois são esperadas, mas diferenças grandes precisam ser investigadas.

### Farol do site e alerta do banco

A regra usada nas janelas históricas é diferente da regra visual da extensão:

1. calcula a fração de fontes com engajamento alto ou muito alto;
2. 60% ou mais gera verde;
3. de 40% a 59% gera amarelo;
4. abaixo de 40% gera vermelho;
5. confusão de 30% ou mais rebaixa um verde para amarelo;
6. um alerta é salvo apenas quando a cor muda para amarelo ou vermelho.

Portanto, hoje há **duas regras de semáforo**. Essa é uma das causas possíveis para o painel ao vivo e o resumo não parecerem idênticos. A recomendação é escolher uma regra canônica e compartilhá-la entre extensão, banco e documentação.

---

## 11. O que o Supabase salva

### Autenticação e propriedade

O Supabase Auth identifica a conta. A sessão criada pela extensão recebe `comunicador_id`, ligando a reunião ao proprietário.

As políticas RLS determinam que:

- uma reunião pessoal só pode ser lida, alterada ou excluída pelo dono;
- layouts, preferências e anotações também pertencem à conta;
- sessões marcadas explicitamente como `is_demo = true` podem ser lidas por usuários autenticados;
- a chave `service_role` nunca deve ficar na extensão ou no site público.

### Dados básicos enviados pela extensão

- proprietário;
- título;
- código da reunião;
- plataforma Google Meet;
- horário de início;
- status `em_andamento` ou `encerrada`;
- último heartbeat;
- horário de encerramento;
- preferências visuais da extensão.

### Histórico enviado pelo Python

A cada 30 frames processados por fonte, desde que exista uma previsão e o histórico esteja ativo, o Python coloca na fila:

- pseudônimo local da fonte;
- timestamp;
- quatro probabilidades de nível para Engajamento;
- quatro probabilidades para Tédio;
- quatro probabilidades para Confusão;
- quatro probabilidades para Frustração;
- proporção recente de frames com rosto;
- proporção recente de frames com pose;
- latência do último processamento;
- versão do modelo.

O envio é assíncrono: uma fila de até 256 itens evita que o Supabase bloqueie a resposta ao vivo. Se a fila encher, o contador `dropped` aumenta.

### RPCs e tabelas

O escritor Python usa três funções do banco:

| Função | Papel |
|---|---|
| `entrar_sessao` | Cria/recupera o participante pseudônimo e fornece token. |
| `registrar_inferencia` | Grava distribuições, qualidade, latência e modelo. |
| `sair_sessao` | Encerra a credencial da fonte removida. |

O schema completo utiliza ou espera estruturas como:

| Estrutura | Conteúdo |
|---|---|
| `comunicador` | Perfil da conta. |
| `sessao` | Reunião e propriedade. |
| `participante` | Fonte pseudônima dentro da sessão. |
| `inferencia` | Registro temporal da inferência. |
| `inferencia_dimensao` | Distribuições por indicador. |
| `janela` | Agregação de um intervalo. |
| `alerta_semaforo` | Mudanças relevantes do semáforo. |
| `dashboard_visualizacao` | Layouts personalizados. |
| `extensao_preferencia` | Aparência e visibilidade de itens do painel. |
| `anotacao_sessao` | Texto e intervalo anotado pelo usuário. |
| `v_resumo_sessao` | Resumo por reunião. |
| `v_timeline_sessao` | Linha do tempo usada nos gráficos. |

### O que não vai ao Supabase

- vídeo completo;
- JPEGs de diagnóstico;
- áudio;
- landmarks brutos;
- vetor bruto de 120 features;
- nome exibido de participantes;
- reconhecimento facial;
- conteúdo da tela compartilhada.

### Estados do histórico

| Mensagem | Significado |
|---|---|
| **Pronto · aguardando reunião** | URL, chave pública e consentimento existem, mas falta o código da sessão. |
| **Conectando à reunião** | Extensão está associando sessão e Python. |
| **Ativo · código · N registros** | Python confirmou a sessão; N é o total salvo por esta execução. |
| **Ativo, mas o último envio falhou** | A sessão existe, mas a última RPC falhou. |
| **Desativado no Python** | Faltam URL, chave pública ou versão do consentimento ao iniciar o servidor. |

Mesmo com o histórico Python desativado, a extensão pode criar a linha básica da reunião. Ela então aparece no site sem curvas úteis.

### Lacuna de reprodução do banco

O repositório possui ajustes, políticas e verificações, mas não contém claramente, em um único arquivo versionado, toda a criação inicial das tabelas, tipos, triggers e das três RPCs principais. Isso significa que um integrante pode clonar o Git e ainda não conseguir criar um Supabase vazio idêntico ao atual.

Antes da entrega final, exporte e versione um schema completo e limpo, sem dados pessoais e sem chaves secretas. O ideal é ter uma ordem única de migrações e um comando de verificação.

---

## 12. Como o site usa os dados

O site é a camada de consulta e personalização. Ele não precisa executar MediaPipe nem receber frames.

### Login e perfil

Usa o mesmo Supabase Auth da extensão. O perfil pode conter nome, e-mail, instituição, atuação, gênero e caminho de uma foto privada no Storage. As políticas limitam o perfil e a foto ao próprio usuário.

### Minhas reuniões

Exibe resumos vindos de `v_resumo_sessao`. Reuniões pessoais ficam separadas de demonstrações. O usuário pode:

- abrir o resumo;
- editar o título pelo lápis;
- excluir uma reunião própria;
- identificar sessões em andamento ou concluídas.

Excluir uma sessão depende das chaves estrangeiras configuradas com cascata para remover histórico e anotações relacionados.

### Dashboard geral

Resume várias reuniões, mostra tendências, ranking e comparação. Ao falar em ranking de engajamento, é necessário indicar se o card está usando intensidade média ou fração de fontes; essas duas métricas não podem receber o mesmo rótulo.

O site oferece modelos de dashboard e personalização de ordem, visibilidade e tamanho dos cards. Layouts personalizados são salvos em `dashboard_visualizacao` e sincronizados pela conta.

Para que o usuário veja imediatamente o que adicionou, a interface destaca o card recém-ativado e pode navegar até ele. Essa confirmação visual é preferível a obrigar o usuário a procurar a alteração.

### Resumo da reunião

O resumo usa `v_resumo_sessao`, `v_timeline_sessao`, alertas e anotações. A evolução mostra intensidades agregadas, não porcentagem de pessoas.

O usuário pode selecionar um trecho diretamente no gráfico:

- arrastando a faixa;
- ou escolhendo limites no próprio gráfico.

O mesmo intervalo:

- recalcula os cards do resumo;
- filtra alertas;
- restringe leituras e interpretações;
- define o intervalo da próxima anotação.

Anotações aparecem como faixas discretas sobre a evolução e pertencem somente ao dono da reunião. Demonstrações são somente leitura.

### Reunião ativa

A extensão atualiza `extensao_ativa_em` a cada 15 segundos. O site procura uma reunião da mesma conta com atividade nos últimos 45 segundos e pode mostrar “Você está em uma reunião”.

Se a atividade fica antiga, o site pode marcar a sessão como encerrada. Isso é comunicação por banco e não uma conexão direta entre as duas interfaces.

### Preferências da extensão

Na aba Extensão, o usuário escolhe tema, escala, tamanho do texto e a visibilidade de cada informação. Para cada item, os estados podem ser:

- oculto;
- somente ao passar o mouse;
- sempre visível.

O site salva a configuração em `extensao_preferencia`. O `background.js` lê essa linha, grava a cópia local e o `content.js` aplica o resultado no Meet.

### Suporte

A aba de suporte deve ser a fonte de verdade da linguagem do produto. Ela explica intensidade, confiança, fração de fontes, alerta, seleção de intervalo, anotações e limitações. Textos devem evitar afirmações acusatórias como “a turma estava desinteressada” e preferir “foram estimados sinais associados a tédio neste trecho”.

---

## 13. Usando somente o Python

### Webcam: validar MediaPipe e as features

```powershell
.\.venv\Scripts\python.exe -m mindflow_extractor.runners.webcam_runner
```

Esse modo abre a webcam, desenha face e pose, mostra FPS e permite verificar iluminação e enquadramento. Ele não reproduz sozinho o painel completo da extensão nem salva uma reunião no Supabase.

| Tecla | Ação |
|---|---|
| `Q` ou `Esc` | Encerrar. |
| `S` | Salvar captura manual de diagnóstico. |
| `R` | Limpar buffer temporal. |
| `L` | Mostrar/esconder malha facial. |
| `P` | Mostrar/esconder pose. |

### Processar um vídeo

```powershell
.\.venv\Scripts\python.exe -m mindflow_extractor.runners.video_runner single `
  --video "C:\caminho\video.mp4" `
  --output "C:\caminho\saida" `
  --skip 0
```

O runner gera:

```text
video.npy              → frames × 120 features
video_metadata.json    → FPS, resolução, qualidade, rótulos e tempos
```

Ele prepara dados para treino ou avaliação. Não produz o dashboard do Meet.

### Servidor sem extensão

```powershell
.\.venv\Scripts\python.exe -m mindflow_extractor.runners.extension_server
```

O servidor ficará aguardando `/frame`. Sem um cliente enviando imagens, não há previsão.

### Testar um vídeo no mesmo caminho da extensão

Para uma validação realmente útil, o ideal é criar ou manter um cliente de replay que envie frames de vídeo para `/frame` com o mesmo tamanho e frequência usados pelo Meet. Isso permite comparar resultados repetíveis sem depender de uma chamada ao vivo.

---

## 14. Treinamento, DAiSEE e precisão atual

### O que é treinamento de verdade

Treinar não é somente extrair features. O fluxo completo é:

```text
vídeos DAiSEE + rótulos
   ↓
MediaPipe e 120 features
   ↓
montagem de sequências
   ↓
separação Train / Validation / Test
   ↓
scaler e PCA ajustados somente com treino
   ↓
LSTM aprende com Train
   ↓
Validation escolhe a melhor época
   ↓
Test mede generalização
   ↓
scaler + PCA + LSTM + métricas
```

O DAiSEE fornece quatro rótulos ordinais por clipe: muito baixo, baixo, alto e muito alto para engajamento, tédio, confusão e frustração.

### Comando do piloto

Exemplo de piloto limitado:

```powershell
.\.venv\Scripts\python.exe tools\run_daisee_pilot.py `
  --daisee-root "C:\Users\SEU_USUARIO\Downloads\DAiSEE\DAiSEE" `
  --train-count 200 `
  --validation-count 64 `
  --test-count 64 `
  --workers 2
```

`--workers 2` processa até dois vídeos ao mesmo tempo. Aumentar demais pode deixar o computador mais lento ou causar falta de memória.

Esse piloto usa 30 FPS, sequências de 150 frames para representar cinco segundos e passo de 75 frames. Ele é identificado no próprio código como teste de pipeline, não validação de produção.

### Situação real do modelo distribuído

Os arquivos em `models/apresentacao_TCC` são os mesmos artefatos que vieram do `mindflow_sem_data_set`. Os pilotos posteriores a 30 FPS ficaram em `outputs/training/runs` e **não foram promovidos** para a pasta usada pela extensão.

O modelo oficial atual:

- recebe 30 frames;
- recebe 120 features;
- foi avaliado sobre features cujo metadata indica FPS efetivo próximo de 6;
- portanto aprendeu uma janela de aproximadamente cinco segundos.

O servidor ao vivo tenta receber 30 FPS e também envia 30 frames, que podem representar apenas um segundo. Essa diferença temporal é hoje a principal incompatibilidade técnica do modelo.

Além disso, um novo piloto a 30 FPS usa 150 frames, mas `inference.py` ainda exige 30. Um modelo de 150 frames não deve ser copiado para a produção sem adaptar o servidor e testar todo o contrato.

### Métricas e divergência de avaliação

O `metrics.json` distribuído junto ao modelo registra:

| Indicador | Acurácia registrada | F1 macro registrada |
|---|---:|---:|
| Engajamento | 49,5% | 26,1% |
| Tédio | 29,3% | 19,6% |
| Confusão | 40,1% | 20,7% |
| Frustração | 59,4% | 24,6% |

Entretanto, a reavaliação separada guardada em `reports/daisee_full_test/metrics.json`, executada sobre 1.200 janelas de 400 clipes, encontrou acurácias por janela de aproximadamente:

| Indicador | Reavaliação por janela | Baseline da classe majoritária |
|---|---:|---:|
| Engajamento | 22,0% | 59,5% |
| Tédio | 33,2% | 41,5% |
| Confusão | 18,3% | 75,5% |
| Frustração | 39,3% | 83,3% |

Os números não concordam. Isso pode indicar diferença de código de avaliação, checkpoint, pré-processamento ou forma de calcular a métrica. Antes da apresentação de “precisão do modelo”, essa procedência precisa ser reconciliada e documentada em um único relatório reproduzível.

Mesmo o conjunto de métricas mais favorável possui F1 macro baixo e forte desequilíbrio entre classes. Alguns níveis altos ou muito altos aparecem poucas vezes e quase nunca são reconhecidos. O modelo atual é adequado para demonstrar a integração, mas ainda não sustenta alegações de precisão confiável.

### Como validar corretamente o próximo modelo

1. congelar uma configuração completa de FPS, resolução, sequência e pré-processamento;
2. separar pessoas/clipes de teste que nunca participaram do ajuste;
3. comparar com baseline majoritária;
4. reportar acurácia balanceada, F1 macro, F1 por nível e matriz de confusão;
5. medir qualidade por dimensão, não só uma média geral;
6. avaliar probabilidades com curva de calibração;
7. testar vídeos reais do cenário do TCC com rótulos humanos independentes;
8. promover scaler, PCA e LSTM juntos;
9. registrar versão do modelo e configuração no banco;
10. repetir um teste ponta a ponta extensão → Python → Supabase → site.

---

## 15. O que mudou desde o mindflow_sem_data_set

O projeto anterior era principalmente um extrator, runners e ambiente de treinamento. A integração acrescentou a experiência completa do protótipo.

| Antes | Agora |
|---|---|
| Webcam e arquivos processados diretamente | Google Meet envia fontes ao servidor local. |
| Sem extensão integrada | Manifest V3, painel, etiquetas e login. |
| Sem leitura do Meet | DOM, MutationObserver e classificação de fontes. |
| Uma entrada por runner | Até cinco pipelines e buffers separados. |
| Sem servidor da extensão | `/health`, `/frame`, `/reset` e `/history/session`. |
| Features e treino isolados | Inferência ao vivo com scaler, PCA e LSTM. |
| Sem reunião vinculada | Título, proprietário, código, heartbeat e encerramento. |
| Sem histórico da extensão | Fila numérica para RPCs do Supabase. |
| Sem dashboard integrado | Site com reuniões, resumo, intervalo, alertas e anotações. |
| Sem personalização sincronizada | Layouts e preferências por conta. |
| Sem snapshots automáticos | Log JPEG local de teste a cada cinco segundos. |
| Piloto limitado | Ferramenta a 30 FPS com `--workers` e cache. |

O núcleo MediaPipe e os blocos A–G foram preservados. O modelo oficial também não foi substituído pelos pilotos posteriores.

---

## 16. Privacidade e capturas locais de teste

### Capturas atualmente habilitadas

O arquivo `iniciar_mindflow.bat` inicia o servidor com:

```text
--snapshot-dir "capturas_local" --snapshot-interval-seconds 5
```

Por isso, além do JPEG transitório usado na requisição, o Python salva uma imagem de diagnóstico por fonte aproximadamente a cada cinco segundos:

```text
capturas_local\
└── CODIGO_DA_SESSAO\
    └── participant_01\
        └── AAAAMMDD_HHMMSS_mmm.jpg
```

Essas imagens:

- ficam somente no computador local;
- não são enviadas ao Supabase;
- servem como log manual de teste;
- não participam do treinamento nem da inferência depois de salvas;
- não possuem exclusão automática atualmente.

Para uma versão além do TCC, snapshots devem ficar desativados por padrão. Para desativar, inicie o servidor sem `--snapshot-dir`.

### Cuidados obrigatórios

- obter consentimento claro das pessoas filmadas;
- explicar que o sistema estima sinais visuais e pode errar;
- não usar o resultado para punição ou decisão individual automática;
- proteger e apagar as capturas locais após o teste;
- adicionar `capturas_local/` ao `.gitignore` antes de qualquer commit;
- nunca enviar `.env`, tokens privados ou `service_role` ao Git;
- definir retenção e exclusão dos dados no Supabase;
- documentar responsável, finalidade e acesso aos dados.

Remover nomes não torna o sistema automaticamente anônimo ou compatível com toda exigência de privacidade. Imagens faciais, geometria e padrões de comportamento continuam sensíveis e exigem análise ética e jurídica apropriada.

---

## 17. Problemas comuns

### “Python indisponível” com `/health` funcionando

1. confirme que o Chrome e o Python estão no mesmo computador;
2. confirme `http://127.0.0.1:8765/health` na mesma instalação do navegador;
3. recarregue a extensão em `chrome://extensions`;
4. atualize a aba do Meet;
5. confirme que a extensão carregada é a pasta correta e versão 1.6.2;
6. procure erros do service worker em `chrome://extensions`, não somente no console do Meet;
7. confirme a permissão `http://127.0.0.1:8765/*` no manifest;
8. encerre cópias antigas do servidor que podem estar usando a porta.

O aviso `Permissions policy violation: unload is not allowed` vindo de arquivos do Google geralmente pertence ao próprio Meet e não prova falha do MindFlow.

### Conecta, mas não aparecem reações

- confirme `modelLoaded: true`;
- confirme que `models/apresentacao_TCC` contém os três artefatos;
- faça login;
- aguarde a reunião ficar pronta no Supabase;
- confirme que a etiqueta diz **IA ATIVA**;
- confirme que rosto e pose estão sendo detectados;
- observe o FPS real;
- aguarde 30 frames processados;
- confira erros de `/frame` no service worker;
- verifique se outra fonte ocupou o limite de cinco.

### A reunião aparece, mas o gráfico do site fica vazio

A linha básica da sessão é criada pela extensão. As curvas dependem do Python e das RPCs. Confira em `/health`:

```text
history.available = true
history.enabled = true
history.configured = true
history.saved aumentando
history.lastError = null
```

Também confirme as funções `entrar_sessao`, `registrar_inferencia`, `sair_sessao`, os triggers de janela e as permissões do banco.

### O gráfico do site não é igual ao painel

As causas possíveis incluem:

- painel suaviza as últimas cinco previsões;
- painel amostra a cada dois segundos;
- Python persiste a cada 30 frames;
- banco agrupa em janelas;
- regras de semáforo diferentes;
- dados atrasados na fila;
- fontes ou qualidade diferentes no período.

### “Histórico Supabase: desativado no Python”

O servidor foi iniciado sem alguma destas configurações:

- URL do Supabase;
- chave pública `anon`;
- versão do consentimento.

Use o `iniciar_mindflow.bat` do pacote atual. Não use uma cópia antiga.

### O título ficou genérico

O Meet não expôs um título confiável. Clique no título ao lado de “MindFlow AI” e corrija manualmente.

---

## 18. Críticas e melhorias recomendadas

### Prioridade 1 — alinhar treino e uso real

Hoje o modelo oficial aprendeu 30 frames próximos de 6 FPS, enquanto o servidor tenta usar 30 frames a até 30 FPS. Antes de treinar mais dados, escolha uma duração temporal e aplique-a em todos os lugares.

Uma opção coerente seria:

```text
30 FPS × 5 segundos = 150 frames
```

Nesse caso, treinamento, servidor, testes, persistência e interface precisam ser adaptados juntos. Outra opção é amostrar a entrada ao vivo para o FPS efetivo do modelo atual. A escolha deve ser medida, não apenas presumida.

### Prioridade 2 — resolver a procedência das métricas

O `metrics.json` do modelo e a reavaliação independente apresentam resultados muito diferentes. Defina um comando único, dados congelados, hash dos artefatos e relatório reproduzível antes de divulgar precisão.

### Prioridade 3 — não confundir as porcentagens

O site e a extensão devem sempre nomear:

- **intensidade estimada**;
- **confiança no nível**;
- **fração de fontes**;
- **métrica de teste**.

Nunca use “52% engajado” sem dizer qual desses conceitos é o número.

### Prioridade 4 — unificar o farol

A extensão usa a dimensão dominante; o banco usa fração de engajamento e regra de confusão. Escolha uma regra central, coloque-a em uma especificação e implemente a mesma semântica nos dois lados.

### Prioridade 5 — usar qualidade para decidir quando não responder

O sistema mede face, pose e NaN, mas ainda pode exibir uma previsão mesmo com qualidade ruim. Recomenda-se um estado explícito:

```text
Dados insuficientes — ajuste iluminação ou enquadramento
```

É melhor não responder do que apresentar um número preciso a partir de uma imagem fraca.

### Prioridade 6 — preservar a proporção da imagem

Antes de aumentar 640 × 480, evite esticar fontes 16:9. Depois faça um teste A/B entre 640 × 480 e 960 × 720 medindo:

- FPS real;
- latência;
- detecção de face e pose;
- quantidade de NaN;
- estabilidade das previsões;
- consumo de CPU e memória para cinco fontes.

### Prioridade 7 — tornar o banco reproduzível

Versione o schema completo, RPCs, triggers, RLS, views e dados de demonstração separados. Um integrante deve conseguir criar um projeto Supabase novo seguindo somente o repositório.

### Prioridade 8 — proteger snapshots

Adicione `capturas_local/` ao `.gitignore`, crie limpeza automática e deixe a captura de teste desativada por padrão fora de sessões controladas.

### Prioridade 9 — teste ponta a ponta reproduzível

Use um vídeo conhecido para repetir exatamente:

```text
vídeo → formato da extensão → Python → RPC → views → site
```

O teste deve comparar timestamps, fonte, intensidades e curvas. Isso ajuda a localizar se uma diferença nasceu no modelo, na agregação ou na interface.

### Prioridade 10 — melhorar a experiência sem prometer demais

- manter explicações perto dos números;
- usar texto e ícone além de vermelho/verde;
- mostrar quando o dado é intensidade ou grupo;
- destacar componentes recém-adicionados ao dashboard;
- manter suporte acessível em todas as telas;
- usar frases neutras e orientadas a perguntas;
- mostrar versão do modelo e qualidade no modo técnico;
- permitir exportar um resumo com limitações metodológicas.

---

## 19. Resumo para apresentação

```text
A extensão entra no Google Meet e lê a estrutura visual da página.
Ela encontra os vídeos e tenta bloquear tela compartilhada e câmera local.
Os participantes permitidos recebem IDs anônimos e são analisados automaticamente.
Cada frame é convertido em JPEG 640 × 480 e enviado somente ao Python local.
O Python usa MediaPipe para extrair face e postura e cria 120 características.
Uma LSTM analisa uma sequência e produz quatro indicadores independentes.
A extensão suaviza as previsões e mostra média, farol, tendência e cinco minutos.
O Python envia apenas resultados numéricos e qualidade ao Supabase.
O site usa o banco para mostrar reuniões, dashboards, intervalos e anotações.
As porcentagens principais são intensidades estimadas, não chance de verdade.
O protótipo funciona como integração, mas o modelo ainda precisa de validação.
```

Em uma frase:

> O MindFlow é um protótipo que transforma sinais visuais de fontes autorizadas do Meet em indicadores coletivos, guarda somente o histórico numérico no Supabase e permite interpretar os períodos no site, sempre reconhecendo que o modelo pode errar.

---

## Glossário

| Termo | Explicação simples |
|---|---|
| **DOM** | Árvore de elementos que representa a página do Meet. |
| **Tile** | Bloco visual onde aparece uma câmera ou apresentação. |
| **Frame** | Uma imagem individual do vídeo. |
| **FPS** | Quantidade de frames por segundo. |
| **Canvas** | Área temporária do navegador usada para copiar um frame. |
| **Landmark** | Ponto geométrico detectado no rosto ou corpo. |
| **Feature** | Número calculado a partir dos landmarks. |
| **NaN** | Marca de medida ausente ou impossível de calcular. |
| **Scaler** | Padroniza as features na mesma escala usada no treino. |
| **PCA** | Reduz a quantidade de informações mantendo variações importantes. |
| **LSTM** | Rede neural que considera uma sequência temporal. |
| **Softmax** | Converte saídas do modelo em probabilidades que somam 100% dentro de uma dimensão. |
| **RPC** | Função do banco chamada remotamente. |
| **RLS** | Regras do Supabase que limitam quais linhas cada conta pode acessar. |
| **Heartbeat** | Sinal periódico de que a extensão continua ativa. |
| **Pseudônimo** | Identificador temporário sem usar o nome da pessoa. |
| **Intensidade** | Posição calculada entre muito baixo e muito alto. |
| **Confiança** | Probabilidade interna do nível vencedor, não precisão comprovada. |
| **Fração de fontes** | Proporção de fontes classificadas em uma condição. |
| **Acurácia** | Parcela total de acertos em um conjunto de teste. |
| **F1 macro** | Métrica que avalia cada classe com o mesmo peso. |

