// Electron main process. Spawns the Python backend, then loads the UI from
// it once it's listening. Killing the window kills the backend.

const { app, BrowserWindow, Menu } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const net = require('net');

const PROJECT_ROOT = __dirname;
const VENV_PYTHON  = path.join(PROJECT_ROOT, '.venv', 'bin', 'python');
const SERVER_PY    = path.join(PROJECT_ROOT, 'server.py');
const PORT         = 8765;
const READY_PROBE_INTERVAL_MS = 120;
const READY_PROBE_TIMEOUT_MS  = 12000;

let backend = null;
let mainWindow = null;

function spawnBackend() {
  backend = spawn(VENV_PYTHON, [SERVER_PY], {
    cwd: PROJECT_ROOT,
    stdio: ['ignore', 'inherit', 'inherit'],
  });
  backend.on('exit', (code, signal) => {
    console.log(`[backend] exited code=${code} signal=${signal}`);
    backend = null;
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.close();
  });
  backend.on('error', (err) => {
    console.error('[backend] spawn error:', err);
  });
}

// Don't try to load the URL until the Python HTTP server is accepting
// connections — otherwise Electron shows an error page that doesn't reload.
function waitForPort(port, timeoutMs) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tryOnce = () => {
      const sock = net.createConnection({ port, host: '127.0.0.1' });
      sock.once('connect', () => { sock.destroy(); resolve(); });
      sock.once('error', () => {
        sock.destroy();
        if (Date.now() - start > timeoutMs) return reject(new Error('backend timeout'));
        setTimeout(tryOnce, READY_PROBE_INTERVAL_MS);
      });
    };
    tryOnce();
  });
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 760,
    minWidth: 900,
    minHeight: 620,
    backgroundColor: '#0a0a0a',
    title: 'Turret Tracking',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  Menu.setApplicationMenu(null);
  try {
    await waitForPort(PORT, READY_PROBE_TIMEOUT_MS);
    mainWindow.loadURL(`http://localhost:${PORT}/`);
  } catch (e) {
    console.error('Backend never came up:', e.message);
    mainWindow.loadURL(
      `data:text/html,<pre style="color:#f88;background:#111;padding:24px">Backend failed to start.\n${e.message}</pre>`
    );
  }
}

app.whenReady().then(() => {
  spawnBackend();
  createWindow();
});

function killBackend() {
  if (backend) {
    try { backend.kill('SIGTERM'); } catch (_) {}
    backend = null;
  }
}

app.on('window-all-closed', () => { killBackend(); app.quit(); });
app.on('before-quit', killBackend);
process.on('exit',     killBackend);
process.on('SIGINT',   () => { killBackend(); process.exit(0); });
process.on('SIGTERM',  () => { killBackend(); process.exit(0); });
