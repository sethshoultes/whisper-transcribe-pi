const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const { dialog } = require('electron');

class DependencyManager {
  constructor(logFunction) {
    this.log = logFunction || console.log;
    this.pythonPath = null;
    this.isSetupComplete = false;
  }

  // Find the best Python installation
  async findPython() {
    // First, check for bundled Python
    const bundledPython = path.join(__dirname, 'python-bundle', 'venv', 'bin', 'python');
    if (fs.existsSync(bundledPython)) {
      this.log(`Found bundled Python: ${bundledPython}`);
      this.pythonPath = bundledPython;
      return bundledPython;
    }
    
    // Check for app-packaged Python
    if (process.platform === 'darwin' && process.resourcesPath) {
      const packagedPython = path.join(process.resourcesPath, 'python-bundle', 'venv', 'bin', 'python');
      if (fs.existsSync(packagedPython)) {
        this.log(`Found packaged Python: ${packagedPython}`);
        this.pythonPath = packagedPython;
        return packagedPython;
      }
    }
    
    // Fall back to system Python
    const pythonCandidates = [
      '/usr/local/bin/python3.11',
      '/usr/local/bin/python3.12',
      '/usr/local/bin/python3.10',
      '/usr/local/bin/python3',
      '/usr/bin/python3',
      'python3.11',
      'python3.12',
      'python3.10',
      'python3',
      'python'
    ];

    for (const candidate of pythonCandidates) {
      try {
        const version = await this.checkPythonVersion(candidate);
        if (version && this.isValidPythonVersion(version)) {
          this.log(`Found valid Python: ${candidate} (${version})`);
          this.pythonPath = candidate;
          return candidate;
        }
      } catch (error) {
        // Continue to next candidate
      }
    }

    throw new Error('No suitable Python installation found (requires Python 3.8+)');
  }

  // Check Python version
  checkPythonVersion(pythonPath) {
    return new Promise((resolve, reject) => {
      exec(`"${pythonPath}" --version`, (error, stdout, stderr) => {
        if (error) {
          reject(error);
          return;
        }
        const version = stdout.trim() || stderr.trim();
        resolve(version);
      });
    });
  }

  // Validate Python version is suitable
  isValidPythonVersion(versionString) {
    const match = versionString.match(/Python (\d+)\.(\d+)/);
    if (!match) return false;
    
    const major = parseInt(match[1]);
    const minor = parseInt(match[2]);
    
    // Require Python 3.8+
    return major === 3 && minor >= 8;
  }

  // Check if a package is installed
  checkPackage(packageName) {
    return new Promise((resolve) => {
      exec(`"${this.pythonPath}" -c "import ${packageName}; print('OK')"`, (error, stdout) => {
        resolve(!error && stdout.trim() === 'OK');
      });
    });
  }

  // Install a package using pip
  installPackage(packageName) {
    return new Promise((resolve, reject) => {
      this.log(`Installing ${packageName}...`);
      
      const installProcess = spawn(this.pythonPath, ['-m', 'pip', 'install', '--user', packageName]);
      
      let output = '';
      let errorOutput = '';
      
      installProcess.stdout.on('data', (data) => {
        output += data.toString();
        this.log(`pip: ${data.toString().trim()}`);
      });
      
      installProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
        this.log(`pip error: ${data.toString().trim()}`);
      });
      
      installProcess.on('close', (code) => {
        if (code === 0) {
          this.log(`Successfully installed ${packageName}`);
          resolve();
        } else {
          reject(new Error(`Failed to install ${packageName}: ${errorOutput}`));
        }
      });
    });
  }

  // Check if we're using bundled Python
  isBundledPython() {
    return this.pythonPath && this.pythonPath.includes('python-bundle');
  }

  // Check and install all required packages
  async setupDependencies() {
    if (!this.pythonPath) {
      await this.findPython();
    }

    const requiredPackages = [
      { name: 'flask', importName: 'flask' },
      { name: 'flask-cors', importName: 'flask_cors' },
      { name: 'openai-whisper', importName: 'whisper' },
      { name: 'sounddevice', importName: 'sounddevice' },
      { name: 'scipy', importName: 'scipy' },
      { name: 'numpy', importName: 'numpy' },
      { name: 'psutil', importName: 'psutil' }
    ];

    // If using bundled Python, assume all packages are installed
    if (this.isBundledPython()) {
      this.log('Using bundled Python environment, verifying packages...');
      
      // Quick verification of bundled packages
      for (const pkg of requiredPackages) {
        const isInstalled = await this.checkPackage(pkg.importName);
        if (!isInstalled) {
          throw new Error(`Bundled Python is missing ${pkg.name}. Bundle may be corrupted.`);
        } else {
          this.log(`✓ ${pkg.name} is available in bundle`);
        }
      }
      
      this.log('Bundled Python environment verified!');
      this.isSetupComplete = true;
      return true;
    }

    const missingPackages = [];

    // Check which packages are missing
    for (const pkg of requiredPackages) {
      const isInstalled = await this.checkPackage(pkg.importName);
      if (!isInstalled) {
        missingPackages.push(pkg);
      } else {
        this.log(`✓ ${pkg.name} is installed`);
      }
    }

    if (missingPackages.length === 0) {
      this.log('All dependencies are installed!');
      this.isSetupComplete = true;
      return true;
    }

    // Ask user permission to install missing packages
    const packageNames = missingPackages.map(p => p.name).join(', ');
    const result = await dialog.showMessageBox({
      type: 'question',
      buttons: ['Install Now', 'Cancel'],
      defaultId: 0,
      title: 'Missing Dependencies',
      message: 'Required Python packages are missing',
      detail: `The following packages need to be installed:\n${packageNames}\n\nThis will install them using pip. Continue?`
    });

    if (result.response !== 0) {
      throw new Error('User cancelled dependency installation');
    }

    // Install missing packages
    for (const pkg of missingPackages) {
      try {
        await this.installPackage(pkg.name);
      } catch (error) {
        this.log(`Failed to install ${pkg.name}: ${error.message}`);
        
        // Show specific error dialog
        await dialog.showErrorBox('Installation Failed', 
          `Failed to install ${pkg.name}.\n\n` +
          `Error: ${error.message}\n\n` +
          `You can try installing manually:\n` +
          `${this.pythonPath} -m pip install --user ${pkg.name}`
        );
        
        throw error;
      }
    }

    // Verify installation
    const stillMissing = [];
    for (const pkg of missingPackages) {
      const isInstalled = await this.checkPackage(pkg.importName);
      if (!isInstalled) {
        stillMissing.push(pkg.name);
      }
    }

    if (stillMissing.length > 0) {
      throw new Error(`Installation verification failed for: ${stillMissing.join(', ')}`);
    }

    this.log('All dependencies installed successfully!');
    this.isSetupComplete = true;
    return true;
  }

  // Get Python path for starting the server
  getPythonPath() {
    return this.pythonPath;
  }

  // Create a startup script that ensures dependencies
  async createStartupScript() {
    const scriptPath = path.join(__dirname, 'start_with_deps.js');
    const script = `
// Auto-generated startup script with dependency checking
const { DependencyManager } = require('./dependency-manager');

async function startWithDependencies() {
  const depManager = new DependencyManager(console.log);
  
  try {
    await depManager.setupDependencies();
    console.log('Dependencies ready, starting server...');
    return depManager.getPythonPath();
  } catch (error) {
    console.error('Dependency setup failed:', error.message);
    throw error;
  }
}

module.exports = { startWithDependencies };
`;

    fs.writeFileSync(scriptPath, script);
    return scriptPath;
  }

  // Check if Whisper model needs to be downloaded
  async checkWhisperModel() {
    try {
      const modelCheck = await new Promise((resolve, reject) => {
        exec(`"${this.pythonPath}" -c "import whisper; whisper.load_model('tiny')"`, 
          { timeout: 30000 }, // 30 second timeout
          (error, stdout, stderr) => {
            if (error) {
              reject(error);
            } else {
              resolve(true);
            }
          }
        );
      });
      
      this.log('Whisper model is ready');
      return true;
    } catch (error) {
      this.log('Whisper model needs to be downloaded (this may take a few minutes on first run)');
      return false;
    }
  }

  // Show setup progress to user
  async showSetupProgress() {
    const result = await dialog.showMessageBox({
      type: 'info',
      buttons: ['OK'],
      title: 'First Time Setup',
      message: 'Setting up Whisper Transcribe',
      detail: 'This is the first time running Whisper Transcribe. We need to:\n\n' +
              '1. Check Python installation\n' +
              '2. Install required packages\n' +
              '3. Download AI model (if needed)\n\n' +
              'This may take a few minutes but only happens once.'
    });
  }
}

module.exports = { DependencyManager };