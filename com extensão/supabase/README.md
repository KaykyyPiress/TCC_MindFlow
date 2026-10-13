# Histórico da extensão no Supabase

O servidor local usa somente a chave pública `anon` e as RPCs já definidas no
schema do MindFlow:

1. `entrar_sessao`: cria o pseudônimo e devolve um token temporário;
2. `registrar_inferencia`: salva as quatro distribuições de probabilidade;
3. os triggers recalculam `janela` e `alerta_semaforo`;
4. `v_timeline_sessao` alimenta o dashboard;
5. `sair_sessao` invalida o token quando o participante deixa de ser observado.

O banco não recebe frames nem landmarks. Somente probabilidades, qualidade,
latência e timestamps são persistidos.

## Inicialização

Crie e inicie a sessão no dashboard. Depois execute o servidor com as variáveis:

```powershell
$env:MINDFLOW_SUPABASE_URL = "https://SEU-PROJETO.supabase.co"
$env:MINDFLOW_SUPABASE_ANON_KEY = "SUA-CHAVE-ANON"
$env:MINDFLOW_CONSENT_VERSION = "VERSAO-DO-TERMO-ACEITO"
python -m mindflow_extractor.runners.extension_server
```

Ao entrar no Google Meet, a extensão autenticada cria a reunião com um título
legível e envia o código interno ao servidor local. Não é mais necessário copiar
`MINDFLOW_SESSION_CODE` manualmente. Se a página não fornecer um título útil, a
extensão usa “Google Meet · data e hora”; o identificador confiável continua sendo
o `id`/`codigo` gerado pelo Supabase.

A versão do termo não possui valor padrão de propósito: o histórico só é
ativado depois que essa variável é informada explicitamente. Ela deve
corresponder ao consentimento que os participantes realmente aceitaram; não a
use para simular consentimento.

Por padrão, a inferência ao vivo pode atualizar a cada 15 frames, enquanto a
persistência ocorre a cada 30 frames por participante (aproximadamente 1 Hz no
alvo de 30 FPS). A fila do Supabase roda em segundo plano e não bloqueia o farol.

Não use `service_role` na extensão ou no código distribuído ao navegador.

