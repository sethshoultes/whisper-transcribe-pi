const { app, BrowserWindow, ipcMain, systemPreferences } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let pythonProcess;

// Enable logging
const logPath = path.join(app.getPath('userData'), 'whisper-app.log');
const logStream = fs.createWriteStream(logPath, { flags: 'a' });

function log(message) {
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${message}\n`;
  console.log(logMessage);
  logStream.write(logMessage);
}

// Start Python backend
function startPythonBackend() {
  // Check multiple Python locations
  const pythonPaths = app.isPackaged ? [
    '/usr/local/bin/python3.11',  // Prioritize system Python 3.11
    '/usr/local/bin/python3',
    '/usr/bin/python3',
    path.join(process.resourcesPath, 'venv', 'bin', 'python3'),
    path.join(process.resourcesPath, 'venv', 'bin', 'python')
  ] : [
    path.join(__dirname, '..', 'venv', 'bin', 'python3')
  ];
  
  // Find the first working Python
  let pythonPath = null;
  for (const pyPath of pythonPaths) {
    if (fs.existsSync(pyPath)) {
      pythonPath = pyPath;
      break;
    }
  }
  
  if (!pythonPath) {
    log('ERROR: Could not find Python executable!');
    return;
  }
    
  const scriptPath = app.isPackaged
      ? path.join(process.resourcesPath, 'backend', 'server.py')
      : path.join(__dirname, 'backend', 'server.py');

    log(`Starting Python backend: ${pythonPath} ${scriptPath}`);

    try {
      // Add common binary paths to environment
      const env = Object.assign({}, process.env);
      const extraPaths = ['/usr/local/bin', '/opt/homebrew/bin', '/usr/bin', '/opt/local/bin'];
      const currentPath = env.PATH || '';
      const pathSet = new Set(currentPath.split(':').filter(p => p));
      extraPaths.forEach(p => pathSet.add(p));
      env.PATH = Array.from(pathSet).join(':');
      
      log(`Spawning Python with PATH: ${env.PATH}`);
      
      pythonProcess = spawn(pythonPath, [scriptPath], { env });

    pythonProcess.stdout.on('data', (data) => {
      log(`Python: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      log(`Python Error: ${data}`);
      
      // Check for missing module errors
      if (data.toString().includes('ModuleNotFoundError')) {
        const { dialog } = require('electron');
        dialog.showErrorBox('Missing Python Dependencies', 
          'Required Python packages are not installed.\n\n' +
          'Please run the following commands in Terminal:\n' +
          'pip3 install flask flask-cors openai-whisper scipy numpy<2\n\n' +
          'Or run: ./setup_dependencies.sh');
        
        // Kill the process
        pythonProcess.kill();
      }
    });

    pythonProcess.on('close', (code) => {
      log(`Python process exited with code ${code}`);
      if (code !== 0 && code !== null) {
        const { dialog } = require('electron');
        dialog.showErrorBox('Python Server Crashed', 
          `The Python server stopped unexpectedly (exit code: ${code}).\n\n` +
          'Check the logs for more details or try restarting the application.');
      }
    });

    pythonProcess.on('error', (error) => {
      log(`Failed to start Python: ${error.message}`);
      // Show error dialog to user
      const { dialog } = require('electron');
      dialog.showErrorBox('Python Server Error', 
        `Failed to start the Python server.\n\n${error.message}\n\nPlease ensure Python 3.11 is installed with required packages.`);
    });
    
  } catch (error) {
    log(`Error spawning Python: ${error.message}`);
  }
}

function createWindow() {
  log('Creating main window...');
  
  mainWindow = new BrowserWindow({
    width: 600,
    height: 500,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    titleBarStyle: 'default',
    alwaysOnTop: true,
    opacity: 0.95
  });

  mainWindow.loadFile('index.html');

  // Open DevTools in development
  if (!app.isPackaged) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  log('Main window created');
}

app.whenReady().then(async () => {
  log('App ready, starting services...');
  
  // Check microphone permissions on macOS
  if (process.platform === 'darwin') {
    const microphoneAccess = systemPreferences.getMediaAccessStatus('microphone');
    log(`Microphone access status: ${microphoneAccess}`);
    
    if (microphoneAccess !== 'granted') {
      // Request microphone access
      const granted = await systemPreferences.askForMediaAccess('microphone');
      log(`Microphone permission request result: ${granted}`);
      
      if (!granted) {
        log('Microphone permission denied!');
        // You might want to show an error dialog here
      }
    }
  }
  
  startPythonBackend();
  
  // Wait longer for Python server to start (server needs time to load model)
  setTimeout(createWindow, 3000);
});

app.on('window-all-closed', () => {
  log('All windows closed');
  if (pythonProcess) {
    pythonProcess.kill();
  }
  app.quit();
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// Handle clipboard operations
ipcMain.handle('copy-to-clipboard', async (event, text) => {
  const { clipboard } = require('electron');
  clipboard.writeText(text);
  log('Text copied to clipboard');
  return true;
});

// Log uncaught errors
process.on('uncaughtException', (error) => {
  log(`Uncaught Exception: ${error.message}\n${error.stack}`);
});

process.on('unhandledRejection', (reason, promise) => {
  log(`Unhandled Rejection at: ${promise}, reason: ${reason}`);
});