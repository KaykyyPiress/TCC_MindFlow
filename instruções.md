# MindFlow AI — documentação da extensão e do Python

> **Versão descrita:** extensão 1.6.1  
> **Objetivo:** explicar o funcionamento atual do protótipo, sem descrever funcionalidades que ainda não existem.
> **Site:** https://mindflowiaa.netlify.app

## Sumário

- [Como rodar o MindFlow](#como-rodar-o-mindflow)
- [1. Visão geral](#1-visão-geral)
- [2. Como a extensão entra e atua no Google Meet](#2-como-a-extensão-entra-e-atua-no-google-meet)
  - [Onde cada arquivo atua](#21-onde-cada-arquivo-atua)
  - [O que acontece quando o Meet é aberto](#22-o-que-acontece-quando-o-meet-é-aberto)
  - [Como o DOM é usado](#23-como-o-dom-é-usado)
  - [Como os vídeos são encontrados](#24-como-os-vídeos-são-encontrados)
  - [Como a extensão acompanha mudanças](#25-como-a-extensão-acompanha-mudanças-no-meet)
  - [Identificadores anônimos](#26-como-uma-fonte-recebe-um-identificador-anônimo)
  - [Bloqueio de tela e apresentador](#27-como-tela-e-apresentador-são-bloqueados)
  - [Etiquetas sobre os vídeos](#28-como-as-etiquetas-são-inseridas-sobre-os-vídeos)
  - [Painel principal](#29-como-o-painel-principal-é-inserido)
- [3. Login, título e ciclo da reunião](#3-login-título-e-ciclo-da-reunião)
- [4. Como os frames chegam ao Python](#4-como-os-frames-chegam-ao-python)
- [5. O que acontece no Python](#5-o-que-acontece-no-python)
  - [MediaPipe](#51-mediapipe)
  - [Vetor de 120 features](#52-vetor-de-120-features)
  - [NaN e dados ausentes](#53-tratamento-de-valores-ausentes-e-nan)
  - [Buffers e LSTM](#54-buffers-e-lstm)
  - [Significado da porcentagem](#55-o-significado-da-porcentagem)
- [6. Como o painel cria o resultado do grupo](#6-como-o-painel-cria-o-resultado-do-grupo)
- [7. O que é salvo no Supabase](#7-o-que-é-salvo-no-supabase)
- [8. Instalação detalhada para os integrantes](#8-instalação-para-os-integrantes)
- [9. Usando somente o Python](#9-usando-somente-o-python)
- [10. O que mudou desde o mindflow_sem_data_set](#10-o-que-foi-acrescentado-ao-mindflow_sem_data_set)
- [11. Limitações atuais](#11-limitações-atuais)
- [12. Resumo final](#12-resumo-final)

---

## Como rodar o MindFlow

Esta é a sequência recomendada para usar a extensão com o Python e o histórico do Supabase.

### Primeira execução: preparar o Python

Depois de extrair `mindflow-integration.zip`, abra o PowerShell dentro da pasta extraída e execute:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Esse processo cria um ambiente Python separado para o MindFlow e instala MediaPipe, OpenCV, PyTorch e as outras dependências.

### Primeira execução: instalar a extensão

1. Abra `chrome://extensions` no Chrome.
2. Ative **Modo do desenvolvedor**.
3. Clique em **Carregar sem compactação**.
4. Selecione a pasta `mindflow-integration\extension`.
5. Confirme que aparece **MindFlow AI — Controle Local**, versão **1.6.1**.

> Se você recebeu somente o ZIP da extensão, extraia-o antes. O Chrome precisa receber a pasta que contém o arquivo `manifest.json`.

### Sempre que for utilizar

1. Dê dois cliques em `iniciar_mindflow.bat`.
2. Aguarde o Python informar que o modelo foi carregado e que o servidor está em `http://127.0.0.1:8765`.
3. Mantenha o Python aberto durante a reunião.
4. Abra ou atualize a página do Google Meet.
5. Faça login no painel MindFlow.
6. Entre efetivamente na chamada.
7. Confira se a tela compartilhada e a câmera local foram bloqueadas automaticamente.
8. Na fonte permitida, clique em **Analisar**.
9. Aguarde a coleta inicial de 30 frames.
10. Confirme que aparecem as quatro reações e que o histórico do Supabase está ativo.

O status esperado é semelhante a:

```text
Python conectado · modelo carregado · alvo 30 fps
Histórico Supabase: ativo · CÓDIGO-DA-REUNIÃO · N registros
```

Se aparecer **Pronto · aguardando reunião**, o Python está configurado, mas a extensão ainda não informou uma chamada ativa. Entre na chamada, confirme o login e aguarde até 15 segundos.

Se aparecer **Desativado no Python**, feche as cópias antigas do servidor e abra novamente `iniciar_mindflow.bat`.

---

## 1. Visão geral

O MindFlow atual é formado por três partes:

| Parte | Responsabilidade |
|---|---|
| **Extensão do Chrome** | Encontra os vídeos no Google Meet, identifica quais fontes não devem ser analisadas, mostra o painel, controla o login e envia temporariamente os frames autorizados. |
| **Python local** | Executa MediaPipe, transforma cada frame em 120 características, mantém as sequências temporais e executa a LSTM. |
| **Supabase e site** | Guardam a reunião, as configurações do usuário e o histórico numérico utilizado posteriormente pelo dashboard. |

O fluxo completo é:

```text
Google Meet
   ↓
content.js observa o DOM e encontra os elementos <video>
   ↓
classifica automaticamente tela compartilhada / apresentador / participante
   ↓
somente uma fonte autorizada como "Analisar" pode ser capturada
   ↓
canvas temporário cria um JPEG 640 × 480
   ↓
background.js encaminha o frame para http://127.0.0.1:8765/frame
   ↓
Python: MediaPipe Face Mesh + Pose
   ↓
478 pontos faciais + 33 pontos corporais
   ↓
120 features divididas nos blocos A–G
   ↓
buffer separado de 30 frames por fonte anônima
   ↓
Scaler → PCA → LSTM com quatro saídas
   ↓
Engajamento + Tédio + Confusão + Frustração
   ↓
extensão suaviza e calcula a média do grupo
   ↓
farol, seta de tendência e gráfico de cinco minutos
   ↓
Python envia o histórico numérico ao Supabase
```

Não existe `offscreen.js` nesta versão. A captura é feita pelo `content.js`, o encaminhamento HTTP pelo `background.js` e toda a visão computacional fica no Python.

---

## 2. Como a extensão entra e atua no Google Meet

A extensão não é um programa separado que tenta localizar a janela do Meet por fora. O próprio Chrome a carrega automaticamente dentro das páginas cujo endereço começa com `https://meet.google.com/`.

Essa regra fica no `manifest.json`:

```text
Google Meet aberto no Chrome
   ↓
Chrome verifica as regras do manifest.json
   ↓
o endereço corresponde a https://meet.google.com/*
   ↓
Chrome injeta content.js e styles.css na página
   ↓
content.js passa a enxergar o DOM e os elementos <video>
```

O carregamento acontece com `document_idle`. Isso significa que o Chrome espera a estrutura inicial da página ficar disponível antes de executar o MindFlow. A extensão não precisa que o usuário aperte o ícone para começar a observar a página; abrir ou atualizar o Meet já ativa o `content.js`.

### 2.1 Onde cada arquivo atua

Os arquivos da extensão não executam todos no mesmo lugar:

| Arquivo | Onde atua | O que faz |
|---|---|---|
| `manifest.json` | Configuração da extensão | Informa ao Chrome em quais páginas carregar o MindFlow e quais endereços podem ser acessados. |
| `content.js` | Dentro da aba do Google Meet | Lê o DOM, encontra os vídeos, cria etiquetas, monta o painel, captura frames autorizados e mostra os resultados. |
| `styles.css` | Dentro da aba do Google Meet | Define o visual do painel e das etiquetas inseridas pelo `content.js`. |
| `background.js` | Área interna da extensão | Faz a ponte com o Python e com o Supabase, realiza login, renova a sessão e guarda preferências. |
| `popup.html` e `popup.js` | Janela aberta pelo ícone da extensão | Oferecem acesso alternativo às informações e ações da conta. |
| `login-panel.html` e arquivos relacionados | Interface da extensão | Dão suporte ao login sem transformar o painel principal em uma página externa. |

O `background.js` não aparece no DOM. Ele roda como um service worker da extensão. Essa separação é importante: o `content.js` pode interagir com a página, enquanto o `background.js` possui as permissões necessárias para acessar `127.0.0.1:8765` e o Supabase.

O caminho de comunicação é:

```text
content.js na página
   ↓ chrome.runtime.sendMessage
background.js na extensão
   ├──→ Python local
   └──→ Supabase
```

### 2.2 O que acontece quando o Meet é aberto

Ao ser carregado, o `content.js` executa uma sequência de inicialização:

1. Verifica se outra cópia do MindFlow já foi carregada na mesma página.
2. Consulta as câmeras do computador para reconhecer a fonte local.
3. Faz a primeira busca pelos elementos `<video>`.
4. Carrega posição, tamanho e preferências do painel.
5. Solicita ao `background.js` a sincronização das preferências do site.
6. Verifica se existe uma conta autenticada.
7. Consulta `GET /health` para saber se o Python e o modelo estão disponíveis.
8. Instala o observador que acompanhará mudanças no Meet.
9. Inicia os ciclos de atualização da interface, lifecycle da reunião, heartbeat e captura.

Existe uma trava chamada `window.__mindflowFinalLoaded`. Ela impede que o painel seja inserido duas vezes caso o script seja reinjetado na mesma página.

O `content.js` executa no ambiente isolado de uma extensão. Ele compartilha o DOM visual com o Meet, mas não depende das variáveis JavaScript internas e privadas do Google. Em outras palavras, ele lê os elementos que o navegador mostra, sem alterar a lógica interna da videoconferência.

### 2.3 Como o DOM é usado

DOM significa **Document Object Model**. No MindFlow, ele não é apenas uma definição teórica: é a principal forma de a extensão compreender o que está visível na reunião.

O navegador representa a página como uma árvore:

```text
Página do Google Meet
├── barra de controles
├── área da reunião
│   ├── tile com um participante
│   │   └── elemento <video>
│   ├── tile com outro participante
│   │   └── elemento <video>
│   └── tile da apresentação
│       └── elemento <video>
└── chat e lista de participantes
```

A extensão utiliza essa árvore para:

- localizar os elementos de vídeo;
- medir a posição e o tamanho de cada tile;
- ler sinais próximos que indiquem “você” ou “apresentação”;
- encontrar o título da reunião;
- perceber se o botão de sair da chamada está presente;
- reconhecer telas de reunião encerrada;
- inserir o painel e as etiquetas do MindFlow.

Ela não substitui os elementos do Meet. Os componentes MindFlow são acrescentados ao `document.body` e ficam visualmente sobre a interface existente.

### 2.4 Como os vídeos são encontrados

A extensão procura todos os elementos `<video>` usando o equivalente a:

```javascript
document.querySelectorAll("video")
```

Encontrar uma tag `<video>` não significa que ela será analisada. Primeiro, o código verifica se ela realmente está visível. O vídeo precisa:

- continuar ligado ao DOM;
- não possuir `display: none` ou `visibility: hidden`;
- não estar transparente;
- possuir pelo menos 48 × 36 pixels;
- estar dentro da área visível da janela.

Depois dessa filtragem, cada vídeo visível vira uma possível “fonte”. A busca completa acontece a cada 900 milissegundos e também pode ser antecipada quando o Meet muda a página.

### 2.5 Como a extensão acompanha mudanças no Meet

O Meet é uma aplicação dinâmica: seus elementos mudam sem que a página seja recarregada. Uma pessoa pode entrar, sair, compartilhar a tela ou mudar de posição na grade enquanto o endereço continua igual.

Por isso, a extensão registra um `MutationObserver` no elemento raiz da página. Ele observa:

- elementos adicionados ou removidos;
- alterações de `src`;
- alterações de classe e estilo;
- reorganização dos tiles.

Quando ocorre uma mudança externa, o observador agenda uma nova leitura após 120 milissegundos. Esse pequeno intervalo permite que o Meet termine a reorganização antes de o MindFlow medir novamente os vídeos.

Mudanças dentro de `#mindflow-panel` ou `.mindflow-source-overlay` são ignoradas. Sem essa proteção, o MindFlow detectaria a própria atualização, atualizaria de novo e poderia criar um ciclo contínuo.

Também são monitorados redimensionamento da janela, rolagem e mudanças nos dispositivos de câmera. Assim, uma etiqueta acompanha o tile mesmo depois de ele mudar de posição.

### 2.6 Como uma fonte recebe um identificador anônimo

Cada vídeo filtrado recebe um identificador local:

```text
participant_01
participant_02
participant_03
```

O identificador não representa o nome da pessoa. Ele serve para conectar três partes temporárias do sistema:

```text
elemento <video> no Meet
   ↕ mesmo pseudônimo
buffer de 30 frames no Python
   ↕ mesmo pseudônimo
registros numéricos daquela fonte no histórico
```

Para tentar preservar o pseudônimo quando a grade é reorganizada, a extensão compara ID da faixa, ID do stream, atributos técnicos do tile e, quando existe, ID do dispositivo. Esses valores ficam na memória da sessão e não realizam reconhecimento facial.

Se a fonte desaparecer, a extensão espera 12 segundos antes de apagá-la definitivamente. Se ela voltar rapidamente, o mesmo estado pode continuar. Depois desse prazo, o overlay é removido e o `background.js` solicita ao Python que limpe o buffer correspondente.

### 2.7 Como tela e apresentador são bloqueados

Depois de criar a fonte anônima, o `content.js` tenta descobrir o papel daquele vídeo. Essa etapa acontece antes da captura.

Uma fonte é considerada **tela compartilhada** quando encontra, por exemplo:

- `displaySurface` indicando monitor, janela ou navegador;
- nome técnico da faixa contendo “screen”, “display”, “window” ou “tela”;
- textos próximos ao tile indicando apresentação ou compartilhamento.

Uma fonte é considerada **apresentador/local** quando encontra:

- o mesmo `deviceId` de uma câmera instalada no computador;
- o mesmo nome técnico da câmera local;
- texto do Meet indicando “você” ou “you”.

Quando um desses papéis é reconhecido, qualquer autorização anterior é cancelada, os resultados visuais são limpos e o Python recebe uma solicitação para descartar o buffer daquela fonte.

Os demais vídeos permanecem como possíveis participantes. Eles não são enviados automaticamente: aparecem primeiro no estado **REVISAR** e só entram no ciclo de captura depois de **Analisar**.

> A classificação depende dos sinais fornecidos pelo navegador e pelo DOM do Meet. Como o Google pode alterar essa estrutura, os seletores e palavras de detecção devem ser revalidados periodicamente.

### 2.8 Como as etiquetas são inseridas sobre os vídeos

Para cada fonte visível, o `content.js` cria um elemento `<section>` com a classe:

```text
mindflow-source-overlay
```

Esse elemento é acrescentado ao `document.body`. Em seguida, a extensão consulta `getBoundingClientRect()` no vídeo para descobrir suas coordenadas e posiciona a etiqueta alguns pixels para dentro do tile.

A etiqueta não fica “presa” dentro do código do Meet. Ela é uma camada independente posicionada sobre o vídeo. Sempre que a grade muda, sua posição é calculada novamente.

Os eventos de clique e toque da etiqueta são interrompidos antes de chegar ao Meet. Isso permite usar os botões **Analisar** e **Ignorar** sem acionar acidentalmente os controles que estão atrás da etiqueta.

Os estados possíveis são:

| Estado | Significado |
|---|---|
| **TELA DETECTADA** | Compartilhamento bloqueado automaticamente. |
| **APRESENTADOR DETECTADO** | Câmera local bloqueada automaticamente. |
| **LOGIN NECESSÁRIO** | O usuário precisa entrar na conta antes de analisar. |
| **REVISAR** | A fonte ainda não é enviada. |
| **IA ATIVA** | A fonte foi autorizada e os frames podem ser enviados. |
| **IGNORADO** | A fonte foi bloqueada manualmente. |

O protótipo aceita no máximo cinco fontes analisadas simultaneamente.

### 2.9 Como o painel principal é inserido

O painel principal é criado pelo `content.js` como um elemento `<aside>` com o identificador `mindflow-panel`. Assim como as etiquetas, ele é acrescentado ao `document.body`, mas não pertence ao código original do Meet.

O painel possui três responsabilidades:

1. mostrar login, título e estado da reunião;
2. apresentar o resultado agregado do grupo;
3. informar situação do Python, FPS e histórico do Supabase.

O cabeçalho funciona como alça de movimentação. Durante o arraste, a extensão atualiza as coordenadas do painel; a alça no canto altera largura e altura. Esses valores são gravados no `chrome.storage.local`, por isso o painel pode reaparecer na posição e no tamanho escolhidos.

O conteúdo interno é renderizado novamente conforme previsões, preferências ou estados mudam. O `styles.css` limita seus estilos ao identificador e às classes do MindFlow para reduzir conflitos com o CSS do Google Meet.

As preferências criadas no site são lidas da tabela `extensao_preferencia` pelo `background.js`, salvas localmente e aplicadas pelo `content.js`. Elas controlam tema, escala, tamanho do texto e se tendência, confiança, índices, gráfico, fontes e informações técnicas aparecem sempre, no hover ou ficam ocultas.

---

## 3. Login, título e ciclo da reunião

O login é mostrado dentro do painel da extensão. O e-mail e a senha são encaminhados diretamente pelo `background.js` para a autenticação do Supabase.

- A senha não é enviada ao Python.
- A senha não é gravada pela extensão.
- Os tokens de acesso e renovação ficam no armazenamento local do Chrome.

Depois do login, a extensão só registra a reunião quando detecta que o usuário realmente entrou em uma chamada. Essa verificação procura os controles de sair ou encerrar a chamada no DOM.

O título é procurado em elementos e atributos acessíveis do Meet. Se não for encontrado, é usado um título de segurança baseado no código da sala, por exemplo:

```text
Google Meet · abc-defg-hij
```

O título aparece ao lado de “MindFlow AI” e pode ser corrigido clicando nele.

No Supabase, a extensão mantém:

- usuário responsável;
- título;
- código interno da sessão;
- plataforma “Google Meet”;
- horário de início;
- status `em_andamento` ou `encerrada`;
- horário da última atividade da extensão;
- horário de encerramento.

Um heartbeat é enviado a cada 15 segundos. Na versão 1.6.1, esse heartbeat também tenta reconectar o histórico ao Python. Assim, se o Python for iniciado depois da entrada no Meet, a gravação pode se recuperar automaticamente.

---

## 4. Como os frames chegam ao Python

Somente uma fonte com estado **IA ATIVA** participa da captura.

Para cada fonte autorizada:

1. O `content.js` cria um `canvas` temporário de 640 × 480.
2. O frame atual do `<video>` é desenhado nesse canvas.
3. O canvas é convertido temporariamente em JPEG com qualidade 0,72.
4. O pacote recebe pseudônimo, horário da captura e tempo de sessão.
5. O pacote é enviado ao `background.js` com `chrome.runtime.sendMessage`.
6. O `background.js` faz `POST /frame` para `http://127.0.0.1:8765`.

O alvo é 30 FPS, mas existe no máximo uma requisição em andamento por fonte. Se o Python ainda estiver processando o frame anterior, a extensão pula novos frames em vez de criar uma fila infinita.

Portanto:

```text
30 FPS configurados ≠ garantia de 30 FPS processados
```

O valor real depende da CPU, do número de fontes, da resolução recebida do Meet e do tempo de inferência.

O JPEG fica apenas na memória durante a requisição local. Ele não é gravado no Supabase.

---

## 5. O que acontece no Python

O servidor local é executado em:

```text
http://127.0.0.1:8765
```

`127.0.0.1` significa que o serviço está disponível somente no próprio computador.

Os principais endpoints são:

| Endpoint | Função |
|---|---|
| `GET /health` | Informa se o Python, o modelo e o histórico estão disponíveis. |
| `POST /frame` | Recebe e processa um frame autorizado. |
| `POST /reset/<id>` | Apaga o buffer de uma fonte que foi ignorada ou removida. |
| `POST /history/session` | Informa ao Python qual reunião deve receber o histórico. |

### 5.1 MediaPipe

O Python decodifica o JPEG, converte BGR para RGB e executa dois extratores:

| Extrator | Saída |
|---|---|
| **Face Mesh** | 478 landmarks: 468 da face e 10 adicionais das íris. |
| **Pose** | 33 landmarks corporais com posição e visibilidade. |

O Face Mesh está limitado a um rosto por fonte. Como cada tile normalmente representa uma pessoa, cada fonte possui seu próprio pipeline e seu próprio buffer temporal.

### 5.2 Vetor de 120 features

Os landmarks não entram diretamente na LSTM. Eles são convertidos em um vetor menor e padronizado:

| Bloco | Quantidade | Exemplos do que representa |
|---|---:|---|
| **A — sinais faciais** | 32 | movimentos aproximados de sobrancelhas, olhos, bochechas e mandíbula. |
| **B — posição da cabeça** | 12 | yaw, pitch, roll e variações temporais. |
| **C — olhar** | 24 | direção, deslocamento e estabilidade do olhar. |
| **D — olhos** | 16 | abertura ocular, piscadas e sinais de sonolência. |
| **E — boca** | 8 | abertura da boca, movimento e aproximação de bocejo. |
| **F — postura** | 24 | inclinação, simetria e posição do tronco, ombros e braços. |
| **G — qualidade** | 4 | confiança da face, confiança da pose e condições gerais do frame. |
| **Total** | **120** | Uma descrição numérica do frame. |

### 5.3 Tratamento de valores ausentes e NaN

Nem toda característica pode ser calculada em todos os frames. Um braço pode estar fora da câmera, o rosto pode estar coberto ou a iluminação pode impedir uma detecção.

Durante a extração, o sistema usa `NaN` para representar “valor não observável”. Isso é diferente de zero: zero poderia significar uma característica realmente medida com valor zero.

Antes de entrar no modelo, os valores `NaN`, `+infinito` e `-infinito` são substituídos por zero. Em seguida são aplicados o scaler e o PCA usados no treinamento. O servidor também devolve a quantidade de valores finitos e ausentes para ajudar no diagnóstico.

### 5.4 Buffers e LSTM

Cada fonte anônima possui estado separado:

```text
participant_01 → pipeline + buffer + qualidade + previsões
participant_02 → pipeline + buffer + qualidade + previsões
participant_03 → pipeline + buffer + qualidade + previsões
```

A primeira previsão só pode ser produzida depois de 30 frames processados. Depois disso, a LSTM é atualizada a cada 15 frames processados.

No alvo ideal de 30 FPS:

- primeira previsão: aproximadamente após um segundo;
- atualização da previsão: aproximadamente duas vezes por segundo;
- gravação no Supabase: aproximadamente uma vez por segundo.

Se o FPS real for menor, esses intervalos também ficam maiores.

O pipeline de inferência é:

```text
30 frames × 120 features
   ↓
limpeza de NaN e infinito
   ↓
Scaler
   ↓
PCA
   ↓
LSTM temporal
   ↓
quatro cabeças independentes
```

As quatro cabeças são:

- Engajamento;
- Tédio;
- Confusão;
- Frustração.

Cada cabeça produz probabilidades para quatro níveis:

```text
muito baixo, baixo, alto, muito alto
```

### 5.5 O significado da porcentagem

A porcentagem exibida para uma reação não significa simplesmente “chance de a pessoa estar naquele estado”.

O modelo fornece quatro probabilidades de nível. O Python calcula uma intensidade ponderada na escala de 0 a 100:

```text
níveis: 0, 1, 2 e 3
score = média ponderada do nível ÷ 3 × 100
```

Exemplo:

```text
Engajamento: 72%
```

Significa que a intensidade estimada de engajamento ficou em 72% da escala entre “muito baixo” e “muito alto”. Não significa, isoladamente, “72% de certeza de que está engajado”.

A **confiança** é outro número: corresponde à probabilidade atribuída pelo modelo ao nível mais provável daquela reação.

---

## 6. Como o painel cria o resultado do grupo

Para reduzir oscilações, a extensão mantém as últimas cinco previsões de cada fonte e calcula uma média.

Depois, calcula a média simples entre todas as fontes que continuam com **IA ATIVA**:

```text
média do grupo = soma dos resultados das fontes ÷ quantidade de fontes válidas
```

Nenhuma fonte recebe peso maior apenas por estar em posição diferente na tela.

O farol escolhe a reação com maior intensidade média:

| Reação dominante | Cor |
|---|---|
| Engajamento | Verde |
| Confusão | Amarelo |
| Tédio | Vermelho |
| Frustração | Vermelho |

Se a maior intensidade ainda estiver abaixo de 50%, o farol fica cinza. A seta compara o engajamento atual com pontos recentes:

- `↗` aumento superior a 3 pontos;
- `↘` queda superior a 3 pontos;
- `→` variação pequena.

O gráfico da extensão guarda em memória os últimos cinco minutos, com uma amostra do grupo a cada dois segundos. Esse gráfico desaparece quando a página é fechada; o histórico durável é o que foi enviado ao Supabase.

---

## 7. O que é salvo no Supabase

Existem duas gravações diferentes.

### 7.1 Dados enviados diretamente pela extensão

Quando o usuário está logado, a extensão pode salvar:

- ID do usuário responsável;
- título e código da reunião;
- plataforma;
- início e encerramento;
- status;
- último heartbeat da extensão;
- preferências de visualização.

### 7.2 Histórico enviado pelo Python

Quando aparece **Histórico Supabase: ativo**, o Python grava aproximadamente a cada 30 frames processados por fonte:

- pseudônimo temporário da fonte;
- timestamp;
- distribuição de quatro níveis para Engajamento;
- distribuição de quatro níveis para Tédio;
- distribuição de quatro níveis para Confusão;
- distribuição de quatro níveis para Frustração;
- proporção recente de frames com rosto;
- proporção recente de frames com pose;
- latência;
- versão do modelo.

O envio utiliza as funções protegidas `entrar_sessao`, `registrar_inferencia` e `sair_sessao`. O Python usa somente a chave pública `anon`, nunca a chave `service_role`.

### 7.3 Significado dos estados do histórico

| Mensagem | Significado |
|---|---|
| **Pronto · aguardando reunião** | Python recebeu as configurações do Supabase, mas ainda não recebeu o código da reunião. |
| **Conectando à reunião** | A extensão está tentando associar a reunião criada ao Python. |
| **Ativo · código · N registros** | A reunião foi associada; `N` mostra quantos registros o Python confirmou nesta execução. |
| **Ativo, mas o último envio falhou** | A sessão está configurada, porém o último acesso ao Supabase retornou erro. |
| **Desativado no Python** | O servidor foi iniciado sem URL, chave pública ou versão de consentimento. |

Mesmo com o histórico numérico desativado, a extensão ainda pode ter criado o registro básico da reunião. Nesse caso, o dashboard encontra a reunião, mas não possui as curvas das quatro reações.

### 7.4 Dados que não são salvos

- vídeo;
- frames JPEG;
- áudio;
- landmarks brutos;
- vetor de 120 features;
- nome dos participantes;
- reconhecimento facial;
- conteúdo da tela compartilhada.

---

## 8. Instalação para os integrantes

### 8.1 Preparar o Python pela primeira vez

Extraia `mindflow-integration.zip`, abra o PowerShell na pasta extraída e execute:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Os arquivos do modelo devem existir em:

```text
models\apresentacao_TCC\
├── scaler.joblib
├── pca.joblib
└── lstm_mindflow.pt
```

### 8.2 Iniciar o servidor da extensão

Dê dois cliques em:

```text
iniciar_mindflow.bat
```

O arquivo inicia o servidor com o Supabase configurado. A janela deve permanecer aberta durante a reunião.

Também é possível iniciar manualmente, desde que as variáveis do Supabase tenham sido configuradas:

```powershell
.\.venv\Scripts\python.exe -m mindflow_extractor.runners.extension_server
```

### 8.3 Adicionar a extensão no Chrome

1. Abra `chrome://extensions`.
2. Ative **Modo do desenvolvedor**.
3. Clique em **Carregar sem compactação**.
4. Selecione a pasta `mindflow-integration\extension`.
5. Confirme que a versão mostrada é **1.6.1**.
6. Atualize a página do Google Meet.

Um ZIP da extensão precisa ser extraído antes de ser carregado sem compactação.

### 8.4 Teste completo recomendado

1. Inicie `iniciar_mindflow.bat`.
2. Abra ou atualize o Google Meet.
3. Faça login no painel MindFlow.
4. Entre efetivamente em uma chamada.
5. Confira se o título apareceu no painel.
6. Confirme que a própria câmera e a tela compartilhada foram bloqueadas.
7. Marque uma fonte permitida como **Analisar**.
8. Aguarde pelo menos 30 frames processados.
9. Confira as quatro reações, o farol e o FPS real.
10. Confira se o status do histórico ficou ativo e se o contador aumenta.
11. Saia da chamada e verifique posteriormente a reunião no dashboard.

---

## 9. Usando somente o Python

“Usar somente o Python” pode significar três coisas diferentes.

### 9.1 Webcam: validar MediaPipe e as 120 features

Execute:

```powershell
.\.venv\Scripts\python.exe -m mindflow_extractor.runners.webcam_runner
```

Esse modo:

- abre a webcam padrão;
- desenha a malha facial e a pose;
- mostra FPS e algumas features;
- informa se face e pose foram detectadas;
- permite verificar iluminação, enquadramento e funcionamento do MediaPipe.

Teclas:

| Tecla | Ação |
|---|---|
| `Q` ou `Esc` | Encerrar. |
| `S` | Salvar manualmente uma captura de diagnóstico. |
| `R` | Limpar o buffer temporal. |
| `L` | Mostrar ou esconder Face Mesh. |
| `P` | Mostrar ou esconder Pose. |

Esse runner não usa o Supabase e, na implementação atual, não mostra as quatro previsões da LSTM. Ele é um modo de diagnóstico do extrator.

### 9.2 Arquivo de vídeo: gerar features para dataset

Execute:

```powershell
.\.venv\Scripts\python.exe -m mindflow_extractor.runners.video_runner single `
  --video "C:\caminho\video.mp4" `
  --output "C:\caminho\saida" `
  --skip 0
```

Esse modo processa o arquivo e gera:

```text
video.npy
video_metadata.json
```

O `.npy` possui formato:

```text
quantidade de frames × 120 features
```

O JSON registra FPS e resolução da fonte, frames processados, detecção de rosto e pose, quantidade de NaN e tempo de processamento.

Esse runner prepara dados para treinamento e avaliação. Ele não grava uma reunião no Supabase e não produz, sozinho, o painel do Meet.

### 9.3 Processar um lote do DAiSEE

Exemplo:

```powershell
.\.venv\Scripts\python.exe -m mindflow_extractor.runners.video_runner daisee `
  --root "C:\Users\SEU_USUARIO\Downloads\DAiSEE\DAiSEE" `
  --output "C:\caminho\features_daisee" `
  --split Train `
  --max 20 `
  --skip 0
```

Esse processo extrai as features dos vídeos do dataset e lê os rótulos do DAiSEE. Treinamento e avaliação são etapas posteriores, executadas pelos scripts da pasta `training` ou pela ferramenta piloto.

### 9.4 Servidor Python sem a extensão

É possível iniciar:

```powershell
.\.venv\Scripts\python.exe -m mindflow_extractor.runners.extension_server
```

Porém, sozinho ele apenas fica aguardando requisições. Sem a extensão ou outro cliente enviando frames para `/frame`, nenhum participante será processado e nenhuma previsão será criada.

---

## 10. O que foi acrescentado ao `mindflow_sem_data_set`

O projeto-base já possuía o extrator MediaPipe, os blocos A–G, o vetor de 120 features e os runners de webcam e vídeo. A integração atual acrescentou principalmente:

- extensão Manifest V3 para o Google Meet;
- leitura e acompanhamento do DOM;
- classificação automática de tela e câmera local;
- pseudônimos e buffers separados para até cinco fontes;
- captura temporária em 640 × 480 com alvo de 30 FPS;
- servidor HTTP local em `127.0.0.1:8765`;
- carregamento do scaler, PCA e LSTM treinada;
- quatro saídas independentes;
- suavização, média do grupo, farol, tendência e gráfico;
- login do Supabase dentro do painel;
- criação automática de reunião, título, heartbeat e encerramento;
- histórico numérico no Supabase;
- reconexão automática do histórico na versão 1.6.1;
- testes de compatibilidade do vetor, modelo e payload do Supabase.

O MediaPipe permaneceu no Python. A extensão não executa extração de landmarks nem a LSTM.

---

## 11. Limitações atuais

- O protótipo depende da estrutura mutável do DOM do Google Meet.
- Cinco fontes a 30 FPS representam uma carga alta; o FPS real pode ficar abaixo do alvo.
- O painel indica o que o modelo estimou, não uma verdade psicológica ou diagnóstico.
- Precisão científica depende da avaliação com dados não usados no treinamento.
- O farol utiliza regras visuais operacionais e ainda precisa ser validado com usuários.
- A reunião só é vinculada ao histórico quando existe login, chamada ativa, Python disponível e funções RPC configuradas.
- O modo webcam do Python valida features, mas ainda não substitui a experiência completa da extensão.

---

## 12. Resumo final

Em termos simples:

```text
A extensão decide de qual vídeo pode tirar frames.
O Python transforma esses frames em números e executa a IA.
O Supabase guarda apenas o resultado numérico e os dados da reunião.
O site lê esse histórico depois.
```

O DOM é a ponte entre a extensão e o Google Meet. Ele permite encontrar tiles, acompanhar mudanças, bloquear tela/apresentador e posicionar o painel. O Python é o núcleo de visão computacional: MediaPipe, 120 features, sequência temporal e LSTM. O Supabase é a memória durável do sistema, sem receber vídeo, landmarks ou nomes de participantes.
