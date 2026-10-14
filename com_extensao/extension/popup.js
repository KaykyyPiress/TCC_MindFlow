const signedOutView = document.querySelector('#signedOutView');
const signedInView = document.querySelector('#signedInView');
const statusElement = document.querySelector('#status');

function sendMessage(message) {
  return new Promise(resolve => {
    const timeout = setTimeout(() => resolve({ success: false, error: 'A extensão não respondeu em 12 segundos. Recarregue-a no Chrome.' }), 12_000);
    chrome.runtime.sendMessage(message, response => {
      clearTimeout(timeout);
      if (chrome.runtime.lastError) {
        resolve({ success: false, error: chrome.runtime.lastError.message });
        return;
      }
      resolve(response || { success: false, error: 'Sem resposta da extensão.' });
    });
  });
}

function setStatus(message, isError = false) {
  statusElement.textContent = message;
  statusElement.classList.toggle('error', isError);
}

function preferenceSummary(preferences) {
  const theme = preferences?.theme === 'dark' ? 'escuro' : 'claro';
  const density = preferences?.density === 'comfortable' ? 'completo' : 'compacto';
  return `Tema ${theme} · painel ${density}`;
}

function showAccount(state) {
  signedOutView.hidden = state.signedIn;
  signedInView.hidden = !state.signedIn;
  document.querySelector('#accountEmail').textContent = state.user?.email || 'Conta conectada';
}

async function loadState() {
  const response = await sendMessage({ type: 'MINDFLOW_ACCOUNT_STATE' });
  if (!response.success) return setStatus('Não foi possível ler a conta.', true);
  showAccount(response);
  if (response.signedIn) {
    setStatus('Sincronizando com o site...');
    const sync = await sendMessage({ type: 'MINDFLOW_SYNC_PREFERENCES' });
    setStatus(sync.success ? `Visual atualizado: ${preferenceSummary(sync.preferences)}.` : `Conta conectada, mas não sincronizou: ${sync.error || 'erro desconhecido'}`, !sync.success);
  }
}

document.querySelector('#loginForm').addEventListener('submit', async event => {
  event.preventDefault();
  const button = document.querySelector('#loginButton');
  button.disabled = true;
  button.textContent = 'Entrando...';
  const response = await sendMessage({
    type: 'MINDFLOW_LOGIN',
    email: document.querySelector('#email').value,
    password: document.querySelector('#password').value
  });
  button.disabled = false;
  button.textContent = 'Entrar e sincronizar';
  if (!response.success) return setStatus(`Não foi possível entrar: ${response.error || 'verifique e-mail e senha'}`, true);
  showAccount({ signedIn: true, user: response.user });
  setStatus(response.sync
    ? `Conta conectada. ${preferenceSummary(response.sync.preferences)}.`
    : `Conta conectada, mas não sincronizou: ${response.syncError || 'tente novamente'}`,
  !response.sync);
});

document.querySelector('#syncButton').addEventListener('click', async event => {
  event.currentTarget.disabled = true;
  setStatus('Sincronizando...');
  const response = await sendMessage({ type: 'MINDFLOW_SYNC_PREFERENCES' });
  event.currentTarget.disabled = false;
  setStatus(response.success ? `Visual atualizado: ${preferenceSummary(response.preferences)}.` : `Não foi possível sincronizar: ${response.error || 'erro desconhecido'}`, !response.success);
});

document.querySelector('#logoutButton').addEventListener('click', async () => {
  await sendMessage({ type: 'MINDFLOW_LOGOUT' });
  showAccount({ signedIn: false });
  setStatus('Conta desconectada.');
});

void loadState();
