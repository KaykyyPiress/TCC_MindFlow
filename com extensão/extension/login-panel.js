const statusElement = document.querySelector('#status');

function sendMessage(message) {
  return new Promise(resolve => {
    const timeout = setTimeout(() => resolve({ success: false, error: 'A extensão não respondeu em 12 segundos.' }), 12_000);
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
  if (!response.success) {
    setStatus(response.error || 'Não foi possível entrar.', true);
    return;
  }
  setStatus(response.sync ? 'Conta conectada e visual sincronizado.' : 'Conta conectada; sincronize novamente no painel.', !response.sync);
});
