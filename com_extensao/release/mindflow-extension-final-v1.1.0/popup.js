const signedOutView = document.querySelector('#signedOutView');
const signedInView = document.querySelector('#signedInView');
const statusElement = document.querySelector('#status');

function sendMessage(message) {
  return new Promise(resolve => chrome.runtime.sendMessage(message, response => resolve(response || { success: false, error: 'Sem resposta' })));
}

function setStatus(message, isError = false) {
  statusElement.textContent = message;
  statusElement.classList.toggle('error', isError);
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
    setStatus(sync.success ? 'Visual atualizado com o site.' : 'Conta conectada; sincronização indisponível.', !sync.success);
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
  if (!response.success) return setStatus('E-mail, senha ou sincronização inválidos.', true);
  showAccount({ signedIn: true, user: response.user });
  setStatus('Conta conectada e visual sincronizado.');
});

document.querySelector('#syncButton').addEventListener('click', async event => {
  event.currentTarget.disabled = true;
  setStatus('Sincronizando...');
  const response = await sendMessage({ type: 'MINDFLOW_SYNC_PREFERENCES' });
  event.currentTarget.disabled = false;
  setStatus(response.success ? 'Visual atualizado no Meet.' : 'Não foi possível sincronizar.', !response.success);
});

document.querySelector('#logoutButton').addEventListener('click', async () => {
  await sendMessage({ type: 'MINDFLOW_LOGOUT' });
  showAccount({ signedIn: false });
  setStatus('Conta desconectada.');
});

void loadState();
